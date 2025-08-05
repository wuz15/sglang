import gc
import logging
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch.nn import Module

from sglang.srt.layers.quantization.deep_gemm import _ENABLE_JIT_DEEPGEMM
from sglang.srt.managers.expert_location import get_global_expert_location_metadata
from sglang.srt.managers.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.managers.schedule_batch import global_server_args_dict

try:
    from deep_gemm import (
        get_col_major_tma_aligned_tensor,
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
        m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    )
    from sgl_kernel import silu_and_mul

    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    use_deep_gemm = True
except ImportError:
    use_deep_gemm = False

from sglang.srt.custom_op import CustomOp
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    ep_gather,
    ep_scatter,
    gelu_and_mul_triton_kernel,
    grouped_gemm_triton,
    post_reorder_triton_kernel,
    pre_reorder_triton_kernel,
    run_moe_ep_preproess,
    silu_and_mul_masked_post_quant_fwd,
    silu_and_mul_triton_kernel,
    tma_align_input_scale,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE, FusedMoEMethodBase
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_kernel import (
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import DeepEPMode, dispose_tensor, is_hip, set_weight_attrs

_is_hip = is_hip()

if _is_hip:
    from vllm._custom_ops import scaled_fp8_quant

logger = logging.getLogger(__name__)


class GroupedGemmRunner(torch.nn.Module):
    flashinfer_gemm_warpper = None

    def __init__(
        self,
        device,
        use_flashinfer: bool = False,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()
        self.device = device
        self.use_flashinfer = use_flashinfer
        self.use_per_token_if_dynamic = use_per_token_if_dynamic
        if self.use_flashinfer and GroupedGemmRunner.flashinfer_gemm_warpper is None:
            GroupedGemmRunner._init_flashinfer_wrapper(device)

    @classmethod
    def _init_flashinfer_wrapper(cls, device):
        from flashinfer import SegmentGEMMWrapper

        workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8, device=device
        )
        cls.flashinfer_gemm_warpper = SegmentGEMMWrapper(workspace_buffer)

    # c = a * b
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        batch_size: int,
        weight_column_major: bool,
        seg_indptr: Optional[torch.Tensor] = None,
        weight_indices: Optional[torch.Tensor] = None,
        use_fp8_w8a8: bool = False,
        scale_a: torch.Tensor = None,
        scale_b: torch.Tensor = None,
        block_shape: Optional[List[int]] = None,
        c_dtype=None,
    ):
        if self.use_flashinfer:
            # TODO: flashinfer
            assert False
            assert GroupedGemmRunner.flashinfer_gemm_warpper is not None
            c = GroupedGemmRunner.flashinfer_gemm_warpper.run(
                x=a,
                weights=b,
                batch_size=batch_size,
                weight_column_major=weight_column_major,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices,
            )
        else:
            assert weight_column_major == True
            c = grouped_gemm_triton(
                a,
                b,
                c,
                batch_size,
                weight_column_major,
                seg_indptr,
                weight_indices,
                use_fp8_w8a8,
                scale_a,
                scale_b,
                block_shape=block_shape,
                c_dtype=c_dtype,
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )
        return c


class EPMoE(torch.nn.Module):
    """
    MoE Expert Parallel Impl


    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
    ):
        super().__init__()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        self.tp_rank = get_tensor_model_parallel_rank()

        self.layer_id = layer_id
        self.num_experts = num_experts
        assert self.num_experts % self.tp_size == 0
        assert (
            num_fused_shared_experts == 0
        ), "num_fused_shared_experts is not supported in EP"
        self.num_experts_per_partition = self.num_experts // self.tp_size
        self.start_expert_id = self.tp_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = top_k
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.correction_bias = correction_bias
        self.custom_routing_function = custom_routing_function
        self.activation = activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_per_token_if_dynamic = use_per_token_if_dynamic

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedEPMoEMethod()
            self.use_fp8_w8a8 = False
            self.use_block_quant = False
            self.block_shape = None
            self.activation_scheme = None
        else:
            if isinstance(self, EPMoESparseCPUInfer):
                self.quant_method: Optional[QuantizeMethodBase] = (
                    Fp8EPMoEMethodCPUInfer(quant_config)
                )
            else:
                self.quant_method: Optional[QuantizeMethodBase] = Fp8EPMoEMethod(
                    quant_config
                )
            self.use_fp8_w8a8 = True
            self.use_block_quant = getattr(self.quant_method, "block_quant", False)
            self.block_shape = (
                self.quant_method.quant_config.weight_block_size
                if self.use_block_quant
                else None
            )
            self.fp8_dtype = torch.float8_e4m3fn
            self.activation_scheme = quant_config.activation_scheme

        self.quant_method.create_weights(
            layer=self,
            num_experts_per_partition=self.num_experts_per_partition,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader,
        )

        self.grouped_gemm_runner = None

    def select_experts(self, **kwargs):
        return select_experts(**kwargs)

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        hidden_states_shape = hidden_states.shape
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None

        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device,
                use_flashinfer=False,  # TODO: use flashinfer
                use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            )

        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
            routed_scaling_factor=self.routed_scaling_factor,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=self.layer_id,
            ),
        )

        reorder_topk_ids, src2dst, seg_indptr = run_moe_ep_preproess(
            topk_ids, self.num_experts
        )

        gateup_input = torch.empty(
            (int(hidden_states.shape[0] * self.top_k), hidden_states.shape[1]),
            device=hidden_states.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states.dtype
            ),
        )
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                max_value = torch.max(hidden_states, dim=1).values.to(torch.float32)
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
            else:
                max_value = (
                    torch.max(hidden_states)
                    .repeat(self.num_experts_per_partition)
                    .to(torch.float32)
                )
                self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max

        # PreReorder
        pre_reorder_triton_kernel[(hidden_states.shape[0],)](
            hidden_states,
            gateup_input,
            src2dst,
            topk_ids,
            self.w13_input_scale,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states.shape[1],
            BLOCK_SIZE=512,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
        )
        dispose_tensor(hidden_states)

        if (
            self.activation_scheme == "dynamic"
            and not self.use_block_quant
            and self.use_per_token_if_dynamic
        ):
            scale = torch.empty(
                hidden_states_shape[0] * self.top_k,
                device=hidden_states_device,
                dtype=torch.float32,
            )
            scale[src2dst] = (
                self.w13_input_scale.unsqueeze(1)
                .expand(hidden_states_shape[0], self.top_k)
                .reshape(-1)
            )
            self.w13_input_scale = scale

        seg_indptr_cur_rank = seg_indptr[self.start_expert_id : self.end_expert_id + 2]
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states_device,
            dtype=torch.int64,
        )
        # GroupGemm-0
        gateup_output = self.grouped_gemm_runner(
            a=gateup_input,
            b=self.w13_weight,
            c=None,
            c_dtype=hidden_states_dtype,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w13_input_scale,
            scale_b=(
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del gateup_input

        # Act
        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            self.w2_input_scale = None
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=hidden_states_dtype,
            )
        else:
            down_input = torch.empty(
                gateup_output.shape[0],
                gateup_output.shape[1] // 2,
                device=gateup_output.device,
                dtype=(
                    self.fp8_dtype
                    if (self.use_fp8_w8a8 and not self.use_block_quant)
                    else hidden_states_dtype
                ),
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        elif self.activation == "gelu":
            gelu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                self.start_expert_id,
                self.end_expert_id,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")
        del gateup_output

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            if self.use_per_token_if_dynamic:
                down_input, self.w2_input_scale = sglang_per_token_quant_fp8(down_input)
            else:
                self.w2_input_scale = torch.ones(
                    self.num_experts_per_partition,
                    dtype=torch.float32,
                    device=hidden_states_device,
                )

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        down_output = self.grouped_gemm_runner(
            a=down_input,
            b=self.w2_weight,
            c=down_output,
            batch_size=self.num_experts_per_partition,
            weight_column_major=True,
            seg_indptr=seg_indptr_cur_rank,
            weight_indices=weight_indices_cur_rank,
            use_fp8_w8a8=self.use_fp8_w8a8,
            scale_a=self.w2_input_scale,
            scale_b=(
                self.w2_weight_scale_inv
                if self.use_block_quant
                else self.w2_weight_scale
            ),
            block_shape=self.block_shape,
        )
        del down_input

        # PostReorder
        output = torch.empty(
            hidden_states_shape, dtype=hidden_states_dtype, device=hidden_states_device
        )
        post_reorder_triton_kernel[(hidden_states_shape[0],)](
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.end_expert_id,
            self.top_k,
            hidden_states_shape[1],
            BLOCK_SIZE=512,
        )
        return output

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
    ) -> List[Tuple[str, str, int, str]]:
        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        physical_expert_ids = (
            get_global_expert_location_metadata().logical_to_all_physical(
                self.layer_id, expert_id
            )
        )
        for physical_expert_id in physical_expert_ids:
            self._weight_loader_physical(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=weight_name,
                shard_id=shard_id,
                expert_id=physical_expert_id,
            )

    def _weight_loader_physical(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if expert_id < self.start_expert_id or expert_id > self.end_expert_id:
            return
        expert_id = expert_id - self.start_expert_id

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Special case for fp8 scales.
        if "scale" in weight_name:
            self._load_fp8_scale(
                param.data,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
            )
            return

        if shard_id == "w2":
            param.data[expert_id] = loaded_weight
        elif shard_id == "w1":
            param.data[expert_id][: self.intermediate_size, :] = loaded_weight
        elif shard_id == "w3":
            param.data[expert_id][self.intermediate_size :, :] = loaded_weight
        else:
            raise ValueError(f"Expected shard_id w1,w2 or w3 but got {shard_id}")

    def _load_fp8_scale(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        if "input_scale" in weight_name:
            if (
                (shard_id == "w1" or shard_id == "w3")
                and param_data[expert_id] != 1
                and (param_data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param_data[expert_id]} "
                    f"vs. {loaded_weight}"
                )
            param_data[expert_id] = loaded_weight
        # Weight scales
        elif "weight_scale" in weight_name:
            if self.use_block_quant:
                block_n, block_k = self.block_shape[0], self.block_shape[1]
                if shard_id == "w1":
                    param_data[expert_id][
                        : (self.intermediate_size + block_n - 1) // block_n, :
                    ] = loaded_weight
                elif shard_id == "w3":
                    param_data[expert_id][
                        (self.intermediate_size + block_n - 1) // block_n :, :
                    ] = loaded_weight
                else:  # w2
                    param_data[expert_id] = loaded_weight
            # If we are in merged column case (gate_up_proj)
            else:
                if shard_id in ("w1", "w3"):
                    # We have to keep the weight scales of w1 and w3 because
                    # we need to re-quantize w1/w3 weights after weight loading.
                    idx = 0 if shard_id == "w1" else 1
                    param_data[expert_id][idx] = loaded_weight

                # If we are in the row parallel case (down_proj)
                else:
                    param_data[expert_id] = loaded_weight


class EPMoESparse(EPMoE):
    # Based on EPMoE, change start_expert_id and end_expert_id to always from 0 to num of expert in this EP rank
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        ep_rank: Optional[Union[int, str]] = None,
    ):
        expert_map = expert_map or self.create_default_expert_map(num_experts, tp_size)

        self.num_experts = num_experts
        if ep_rank is None:
            self.ep_rank = get_tensor_model_parallel_rank()
        else:
            self.ep_rank = ep_rank

        self.expert_id_to_local = [-1] * num_experts
        local_expert_count = 0
        for expert_id in range(num_experts):
            if expert_map.get(expert_id, -1) == self.ep_rank:
                self.expert_id_to_local[expert_id] = local_expert_count
                local_expert_count += 1

        self.expert_id_to_local_tensor = None

        super().__init__(
            local_expert_count,
            top_k,
            hidden_size,
            intermediate_size,
            layer_id,
            params_dtype,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            num_fused_shared_experts,
            topk_group,
            quant_config,
            1,  # create as if tp_size is 1 weights will be created as local_expert_count
            prefix,
            correction_bias,
            custom_routing_function,
            activation,
            routed_scaling_factor,
            use_per_token_if_dynamic,
        )

        self.num_experts = num_experts
        self.start_expert_id = 0
        self.end_expert_id = local_expert_count - 1
        assert local_expert_count == self.num_experts_per_partition

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        local_expert_id = self.expert_id_to_local[expert_id]
        return super().weight_loader(
            param, loaded_weight, weight_name, shard_id, local_expert_id
        )

    def select_experts(self, **kwargs):
        topk_weights, topk_ids = select_experts(**kwargs)

        return topk_weights, self.map_expertid_to_local_experts(topk_ids)

    def map_expertid_to_local_experts(self, topk_ids):
        if self.expert_id_to_local_tensor is None:
            self.expert_id_to_local_tensor = torch.tensor(
                self.expert_id_to_local, device=topk_ids.device
            )

        return self.expert_id_to_local_tensor[topk_ids]

    def create_default_expert_map(self, num_experts, tp_size):
        ep_size = tp_size or get_tensor_model_parallel_world_size()
        gpu_experts_per_rank = num_experts // ep_size
        assert num_experts == gpu_experts_per_rank * ep_size
        expert_map_grouped = {
            e_id: e_id // gpu_experts_per_rank for e_id in range(num_experts)
        }
        expert_map_balanced = {e_id: e_id % ep_size for e_id in range(num_experts)}
        expert_map_imbalanced = {
            e_id: 0 if e_id < 62 else 1 for e_id in range(num_experts)
        }
        expert_map = expert_map_balanced
        return expert_map


class EPMoESparseCPUInterface(EPMoESparse):
    def forward_start(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        raise NotImplementedError()

    def forward_sync(self):
        raise NotImplementedError()

    def forward(self):
        result = self.forward_start()
        self.forward_sync()
        return result


class EPMoESparseCPUEmu(EPMoESparseCPUInterface):
    device = "cuda"

    def forward_start(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            cpu_result = super().forward(
                hidden_states.to(self.device),
                router_logits.to(self.device),
            )
        return cpu_result

    def forward_sync(self):
        torch.cuda.current_stream().wait_stream(self.stream)

    @property
    def stream(self):
        if not hasattr(self, "_stream"):
            self._stream = torch.cuda.Stream()
        return self._stream

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        this_param = [
            p[1] for p in self.named_parameters() if p[0] == weight_name.split(".")[-1]
        ][0]
        return super().weight_loader(
            this_param, loaded_weight, weight_name, shard_id, expert_id
        )


class EPMoESparseCPUInfer(EPMoESparseCPUInterface):
    device = "cpu"
    cpu_infer = None

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        ep_rank: Optional[Union[int, str]] = None,
    ):
        if EPMoESparseCPUInfer.cpu_infer is None:
            import moe_cpu_engine as cpu_engine

            EPMoESparseCPUInfer.cpu_infer = cpu_engine.CPUInfer(0)
            EPMoESparseCPUInfer.cpu_engine = cpu_engine

        with torch.device(self.device):
            super().__init__(
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                layer_id,
                params_dtype,
                renormalize,
                use_grouped_topk,
                num_expert_group,
                num_fused_shared_experts,
                topk_group,
                quant_config,
                tp_size,
                prefix,
                correction_bias,
                custom_routing_function,
                activation,
                routed_scaling_factor,
                use_per_token_if_dynamic,
                expert_map,
                ep_rank,
            )
        if isinstance(self.quant_method, Fp8EPMoEMethodCPUInfer):
            # if no need post processing, then delete to speed up.
            if (
                self.quant_method.quant_config.activation_scheme == "static"
                and self.quant_method.quant_config.is_checkpoint_fp8_serialized
            ):
                del self.quant_method
                self.quant_method = None
        self.moe_config = (
            3,  # dummy layer id
            self.num_experts_per_partition,
            self.top_k,
            hidden_size,
            intermediate_size,
            renormalize,
            num_expert_group,
            topk_group,
        )
        self.cpu_moe_engine = None
        # CPUInfer doesn't need correction_bias but we need to keep the interface consistent
        self.dummy_correction_bias = torch.zeros(
            self.num_experts, device=self.device, dtype=torch.float
        )
        self.cpu_hidden_states = None
        self.cpu_sorted_topk_ids = None
        self.cpu_sorted_topk_weights = None
        self.cpu_result = None
        self.cpu_tensor_tbo_subbatch_id = 0
        self.tensor_tbo_pool_size = 2
        self.cached_tensors_size = 160  # TODO: magic number
        self.create_cpu_tensors_if_needed_from_params(
            hidden_size, top_k, torch.bfloat16, self.cached_tensors_size
        )
        self._create_expected_weights_set()

    def _create_expected_weights_set(self):
        # create a set of weights names to be loaded,
        # so we can check if all weights are loaded
        # after all weights are loaded, we will set weights to CPU
        # so to avoid over occupying memory in one numa node
        self.expected_weights_set = set()
        for expert_id in range(len(self.expert_id_to_local)):
            if self.expert_id_to_local[expert_id] == -1:
                continue
            for shard_id in ["w1", "w2", "w3"]:
                self.expected_weights_set.add((expert_id, shard_id, "weight"))
                self.expected_weights_set.add((expert_id, shard_id, "weight_scale_inv"))

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> None:
        if shard_id == "w2":
            param = (
                self.w2_weight_scale_inv
                if "scale_inv" in weight_name
                else self.w2_weight
            )
        elif shard_id in ["w1", "w3"]:
            param = (
                self.w13_weight_scale_inv
                if "scale_inv" in weight_name
                else self.w13_weight
            )
        else:
            raise ValueError(f"Unknown weight: {weight_name}")

        super().weight_loader(param, loaded_weight, weight_name, shard_id, expert_id)
        weight_key = (
            expert_id,
            shard_id,
            "weight_scale_inv" if "scale_inv" in weight_name else "weight",
        )
        if weight_key in self.expected_weights_set:
            self.expected_weights_set.remove(weight_key)
            if len(self.expected_weights_set) == 0:
                # all weights are loaded, so we can set weights to CPU
                self._create_cpu_moe_engine()

    def _create_cpu_moe_engine(self):
        if self.cpu_moe_engine is not None:
            raise RuntimeError(
                "CPU MoE engine already created, this should not happen."
            )
        else:
            logger.info(f"Creating CPU MoE engine for layer {self.layer_id}")
            self.cpu_moe_engine = self.cpu_engine.moe.MOE(
                self.cpu_engine.moe.MOEConfig(*self.moe_config)
            )
            self.cpu_moe_engine.set_weights(
                self.w13_weight.data.data_ptr(),
                self.w2_weight.data.data_ptr(),
                self.w13_weight_scale_inv.data.data_ptr(),
                self.w2_weight_scale_inv.data.data_ptr(),
                self.dummy_correction_bias.data.data_ptr(),
            )
            del self.w13_weight
            del self.w2_weight
            del self.w13_weight_scale_inv
            del self.w2_weight_scale_inv
            del self.dummy_correction_bias
            gc.collect()

    def _sort_topk_ids(self, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        # Get the sort indices (descending order)
        sorted_topk_ids, sort_indices = torch.sort(topk_ids, dim=1, descending=True)

        # Create batch indices for gathering
        batch_size = topk_ids.shape[0]
        batch_indices = (
            torch.arange(batch_size, device=topk_ids.device)
            .unsqueeze(1)
            .expand_as(sort_indices)
        )

        # Sort both tensors using the same ordering
        sorted_topk_weights = topk_weights[batch_indices, sort_indices]

        return sorted_topk_weights, sorted_topk_ids

    def forward_start(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        self.forward_prepare(hidden_states, router_logits)
        return self.forward_enqueue(hidden_states)

    def forward_prepare(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        topk_weights, topk_ids = self.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            use_grouped_topk=self.use_grouped_topk,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            correction_bias=self.correction_bias,
            custom_routing_function=self.custom_routing_function,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        sorted_topk_weights, sorted_topk_ids = self._sort_topk_ids(
            topk_weights, topk_ids
        )
        n_tokens = hidden_states.shape[0]
        if (
            torch.cuda.is_current_stream_capturing()
            or n_tokens <= self.cached_tensors_size
        ):
            assert n_tokens <= self.cached_tensors_size
            # we are switching to next set, so prepare and enqueue must be together
            # there cannot be other prepare or enqueue in between
            # please make sure check operation strategies won't break this assumption
            self._switch_to_next_tensor_set()
            self.fill_cpu_tensors(hidden_states, sorted_topk_ids, sorted_topk_weights)
        else:
            # these are safe as only decode will run bto heto
            self.adhoc_cpu_result = torch.zeros_like(
                hidden_states, device=self.device, pin_memory=True
            )
            self.adhoc_hidden_states_cpu = hidden_states.to(
                self.device, non_blocking=True
            )
            self.adhoc_sorted_topk_ids_cpu = sorted_topk_ids.to(torch.int).to(
                self.device, non_blocking=True
            )
            self.adhoc_sorted_topk_weights_cpu = sorted_topk_weights.to(
                self.device, non_blocking=True
            )

    def forward_enqueue(self, hidden_states: torch.Tensor):
        if self.cpu_moe_engine is None:
            torch.cuda.current_stream().synchronize()
            self._create_cpu_moe_engine()

        n_tokens = hidden_states.shape[0]
        if (
            torch.cuda.is_current_stream_capturing()
            or n_tokens <= self.cached_tensors_size
        ):
            self.cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
                self.cpu_moe_engine.forward_experts(
                    self.cpu_hidden_states.data_ptr(),
                    self.cpu_sorted_topk_ids.data_ptr(),
                    self.cpu_sorted_topk_weights.data_ptr(),
                    self.cpu_result.data_ptr(),
                    n_tokens,
                ),
            )
            return self.cpu_result[:n_tokens]
        else:
            self.cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream(hidden_states.device).cuda_stream,
                self.cpu_moe_engine.forward_experts(
                    self.adhoc_hidden_states_cpu.data_ptr(),
                    self.adhoc_sorted_topk_ids_cpu.data_ptr(),
                    self.adhoc_sorted_topk_weights_cpu.data_ptr(),
                    self.adhoc_cpu_result.data_ptr(),
                    n_tokens,
                ),
            )
            return self.adhoc_cpu_result

    def forward_sync(self, hidden_states_device):
        self.cpu_infer.sync_with_cuda_stream(
            torch.cuda.current_stream(hidden_states_device).cuda_stream
        )

    def forward_to_gpu(self, hidden_states_device, cpu_result):
        return cpu_result.to(hidden_states_device, non_blocking=True)

    def forward(self):
        raise NotImplementedError("forward Not implemented")

    def create_cpu_tensors_if_needed_from_params(
        self, hidden_size, topk, dtype, batchsize
    ):
        if self.cpu_hidden_states is None:
            self.cpu_hidden_states_pool = [
                torch.empty(
                    (batchsize, hidden_size),
                    dtype=dtype,
                    device=self.device,
                    pin_memory=True,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_sorted_topk_ids_pool = [
                torch.empty(
                    (batchsize, topk),
                    dtype=torch.int32,
                    device=self.device,
                    pin_memory=True,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_sorted_topk_weights_pool = [
                torch.empty(
                    (batchsize, topk),
                    dtype=torch.float32,
                    device=self.device,
                    pin_memory=True,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self.cpu_result_pool = [
                torch.empty(
                    (batchsize, hidden_size),
                    dtype=dtype,
                    device=self.device,
                    pin_memory=True,
                )
                for _ in range(self.tensor_tbo_pool_size)
            ]
            self._switch_to_next_tensor_set()

    def _switch_to_next_tensor_set(self):
        self.cpu_hidden_states = self.cpu_hidden_states_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_sorted_topk_ids = self.cpu_sorted_topk_ids_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_sorted_topk_weights = self.cpu_sorted_topk_weights_pool[
            self.cpu_tensor_tbo_subbatch_id
        ]
        self.cpu_result = self.cpu_result_pool[self.cpu_tensor_tbo_subbatch_id]
        self.cpu_tensor_tbo_subbatch_id = (
            self.cpu_tensor_tbo_subbatch_id + 1
        ) % self.tensor_tbo_pool_size

    def fill_cpu_tensors(self, hidden_states, sorted_topk_ids, sorted_topk_weights):
        bs = hidden_states.shape[0]
        # TODO: cpu_moe_engine will crash if all topk_ids are -1
        # below is a hack to prevent that, but we should fix it in cpu_moe_engine
        # sorted_topk_ids[:, 0] = torch.max(
        #     sorted_topk_ids[:, 0],
        #     torch.zeros_like(sorted_topk_ids[:, 0]),
        # )
        assert bs <= self.cpu_hidden_states.shape[0]
        self.cpu_hidden_states[:bs].copy_(hidden_states, non_blocking=True)
        self.cpu_sorted_topk_ids[:bs].copy_(sorted_topk_ids, non_blocking=True)
        self.cpu_sorted_topk_weights[:bs].copy_(sorted_topk_weights, non_blocking=True)


class EPMoEHeto(EPMoESparse):
    RANK_CPU = "C0"

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        use_per_token_if_dynamic: bool = True,
        expert_map: dict[int, Union[int, str]] = {},
        num_gpu_experts=-1,
    ):
        expert_map = expert_map or self.create_default_expert_map(
            num_experts, tp_size, num_gpu_experts
        )

        self._init_cpu_resources(expert_map)

        super().__init__(
            num_experts,
            top_k,
            hidden_size,
            intermediate_size,
            layer_id,
            params_dtype,
            renormalize,
            use_grouped_topk,
            num_expert_group,
            num_fused_shared_experts,
            topk_group,
            quant_config,
            tp_size,
            prefix,
            correction_bias,
            custom_routing_function,
            activation,
            routed_scaling_factor,
            use_per_token_if_dynamic,
            expert_map,
        )
        if self.is_hosting_cpu():
            self.cpu_moe = EPMoESparseCPUInfer(
                num_experts,
                top_k,
                hidden_size,
                intermediate_size,
                layer_id,
                params_dtype,
                renormalize,
                use_grouped_topk,
                num_expert_group,
                num_fused_shared_experts,
                topk_group,
                quant_config,
                tp_size,
                prefix,
                correction_bias,
                custom_routing_function,
                activation,
                routed_scaling_factor,
                use_per_token_if_dynamic,
                expert_map,
                EPMoEHeto.RANK_CPU,
            )

    def _init_cpu_resources(self, expert_map):
        self.cpu_ep_count = sum(1 for x in expert_map.values() if x == self.RANK_CPU)
        self.cpu_moe = None
        self.cpu_stream = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        op_shared_experts: Optional[Callable] = None,
    ):
        # hidden_states will be destroyed in the process, so we need to save its properties
        # for later use
        hidden_states_shape = hidden_states.shape
        hidden_states_device = hidden_states.device
        hidden_states_dtype = hidden_states.dtype
        self.forward_routed_experts_prepare(hidden_states, router_logits)
        cpu_result = self.forward_routed_experts_enqueue(hidden_states, router_logits)
        if op_shared_experts is not None:
            shared_output = op_shared_experts(hidden_states)
        gpu_result = self.forward_routed_experts_maybe_gpu(
            hidden_states, router_logits  # here hidden_states will be destroyed
        )
        self.forward_routed_experts_sync(
            hidden_states_device,
        )
        result = self.forward_routed_experts_combine(
            hidden_states_shape,
            hidden_states_device,
            hidden_states_dtype,
            gpu_result,
            cpu_result,
        )
        return result, shared_output

    def forward_routed_experts_prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.cpu_moe:
            self.cpu_moe.forward_prepare(hidden_states, router_logits)

    def forward_routed_experts_enqueue(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.cpu_moe:
            return self.cpu_moe.forward_enqueue(hidden_states)
        return None

    def forward_routed_experts_maybe_gpu(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ):
        if self.num_experts_per_partition > 0:
            return super().forward(hidden_states, router_logits)
        return None

    def forward_routed_experts_sync(
        self,
        hidden_states_device,
    ):
        if self.cpu_moe:
            self.cpu_moe.forward_sync(hidden_states_device)

    def forward_routed_experts_combine(
        self,
        hidden_states_shape,
        hidden_states_device,
        hidden_states_dtype,
        gpu_result,
        cpu_result,
    ):
        if self.cpu_moe:
            cpu_result_on_gpu = self.cpu_moe.forward_to_gpu(
                hidden_states_device, cpu_result
            )

        if gpu_result is None:
            if self.cpu_moe:
                result = cpu_result_on_gpu
            else:
                result = torch.zeros(
                    hidden_states_shape,
                    dtype=hidden_states_dtype,
                    device=hidden_states_device,
                )
        else:
            if self.cpu_moe:
                result = gpu_result + cpu_result_on_gpu
            else:
                result = gpu_result

        return result

    def is_hosting_cpu(self):
        return self.ep_rank == 0 and self.cpu_ep_count != 0

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id):
        if self.cpu_moe:
            self.cpu_moe.weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id
            )

        return super().weight_loader(
            param, loaded_weight, weight_name, shard_id, expert_id
        )

    def create_default_expert_map(self, num_experts, tp_size, num_gpu_experts):
        # half on CPU, other half set even on GPUs
        ep_size = tp_size or get_tensor_model_parallel_world_size()
        expert_map_plan = dict()
        # gpu_expert_number = num_experts // 2
        gpu_in_high_part = False
        # if num_gpu_experts == 0:
        #     gpu_expert_total = num_experts // 2
        if num_gpu_experts < 0:
            gpu_in_high_part = True
            gpu_expert_total = -num_gpu_experts
        else:
            gpu_expert_total = num_gpu_experts
        logger.debug(f"create_default_expert_map, gpu_expert_total:{gpu_expert_total}")
        for e_id in range(num_experts):
            if (
                num_experts - 1 - e_id if gpu_in_high_part else e_id
            ) < gpu_expert_total:
                expert_map_plan[e_id] = e_id % ep_size
            else:
                expert_map_plan[e_id] = self.RANK_CPU
        return expert_map_plan


class UnquantizedEPMoEMethod(FusedMoEMethodBase, CustomOp):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        layer.register_parameter("w13_input_scale", None)
        layer.register_parameter("w13_weight_scale", None)

        ones_tensor = torch.ones(num_experts_per_partition, dtype=torch.float32)

        w2_input_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        set_weight_attrs(w2_input_scale, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            ones_tensor,
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethod(Fp8MoEMethod):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                2 * intermediate_size,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts_per_partition,
                hidden_size,
                intermediate_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    2 * ((intermediate_size + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts_per_partition,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
            layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # WEIGHT_SCALES
            # Allocate 2 scales for w1 and w3 respectively.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                )
                w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                    scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class Fp8EPMoEMethodCPUInfer(Fp8MoEMethod):
    """MoE method for FP8 for CPU infer.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Difference to Fp8MoEMethod is to create tensor instead of parameters
    as we are not able to release memory if data are stored in parameters
    for unknown reason.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: Module,
        num_experts_per_partition: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        if self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

        # WEIGHTS
        w13_weight = torch.empty(
            num_experts_per_partition,
            2 * intermediate_size,
            hidden_size,
            dtype=params_dtype,
            device="cpu",
        )
        layer.w13_weight = w13_weight
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.empty(
            num_experts_per_partition,
            hidden_size,
            intermediate_size,
            dtype=params_dtype,
            device="cpu",
        )
        layer.w2_weight = w2_weight
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.block_quant:
            w13_weight_scale = torch.ones(
                num_experts_per_partition,
                2 * ((intermediate_size + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_weight_scale = torch.ones(
                num_experts_per_partition,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            layer.w13_weight_scale_inv = w13_weight_scale
            layer.w2_weight_scale_inv = w2_weight_scale
            assert self.quant_config.activation_scheme == "dynamic"
        else:
            # WEIGHT_SCALES
            # Allocate 2 scales for w1 and w3 respectively.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, 2, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        if self.quant_config.is_checkpoint_fp8_serialized:
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "Found static activation scheme for checkpoint that "
                    "was not serialized fp8."
                )

            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts_per_partition, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: Module) -> None:

        # If checkpoint is fp16, quantize in place.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            # If rocm, use float8_e4m3fnuz as dtype
            fp8_dtype = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w13_weight = torch.empty_like(layer.w13_weight.data, dtype=fp8_dtype)
            w2_weight = torch.empty_like(layer.w2_weight.data, dtype=fp8_dtype)

            layer.w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    layer.num_experts_per_partition,
                    dtype=torch.float32,
                    device=w13_weight.device,
                ),
                requires_grad=False,
            )

            for expert in range(layer.num_experts_per_partition):
                if _is_cuda:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        sgl_scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        sgl_scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                    )
                else:
                    w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                        vllm_ops.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
                    )
                    w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                        vllm_ops.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
                    )
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            return

        # If checkpoint is fp8, we need to handle that the
        # MoE kernels require single activation scale and single weight
        # scale for w13 per expert.
        else:
            if self.quant_config.activation_scheme == "static":
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                layer.w13_weight_scale = torch.nn.Parameter(
                    torch.max(layer.w13_weight_scale, dim=1).values,
                    requires_grad=False,
                )
            return

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class DeepEPMoE(EPMoE):
    """
    MoE Expert Parallel Impl based on DeepEP (https://github.com/deepseek-ai/DeepEP/tree/main)
    """

    _has_printed = False

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        params_dtype: Optional[torch.dtype] = None,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        correction_bias: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        activation: str = "silu",
        routed_scaling_factor: Optional[float] = None,
        deepep_mode: DeepEPMode = DeepEPMode.auto,
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
            params_dtype=params_dtype,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            num_fused_shared_experts=num_fused_shared_experts,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            prefix=prefix,
            correction_bias=correction_bias,
            custom_routing_function=custom_routing_function,
            activation=activation,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.deepep_mode = deepep_mode
        if self.deepep_mode.enable_low_latency():
            assert use_deep_gemm, f"DeepEP {self.deepep_mode} mode requires deep_gemm"
        self.w13_weight_fp8 = (
            self.w13_weight,
            (
                self.w13_weight_scale_inv
                if self.use_block_quant
                else self.w13_weight_scale
            ),
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv if self.use_block_quant else self.w2_weight_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        num_recv_tokens_per_expert: List[int],
        forward_mode: ForwardMode,
    ):
        resolved_deepep_mode = self.deepep_mode.resolve(forward_mode)
        if resolved_deepep_mode == DeepEPMode.normal:
            if _ENABLE_JIT_DEEPGEMM:
                return self.forward_deepgemm_contiguous(
                    hidden_states, topk_idx, topk_weights, num_recv_tokens_per_expert
                )
            else:
                return self.forward_normal(hidden_states, reorder_topk_ids, seg_indptr)
        elif resolved_deepep_mode == DeepEPMode.low_latency:
            return self.forward_deepgemm_masked(hidden_states, masked_m, expected_m)
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        reorder_topk_ids: torch.Tensor,
        seg_indptr: torch.Tensor,
    ):
        hidden_states_dtype = hidden_states.dtype
        hidden_states_device = hidden_states.device

        assert self.quant_method is not None
        assert self.activation == "silu"
        if self.grouped_gemm_runner is None:
            self.grouped_gemm_runner = GroupedGemmRunner(
                hidden_states.device, use_flashinfer=False  # TODO: use flashinfer
            )

        if self.activation_scheme == "dynamic" and not self.use_block_quant:
            max_value = (
                torch.max(hidden_states)
                .repeat(self.num_experts_per_partition)
                .to(torch.float32)
            )
            self.w13_input_scale = max_value / torch.finfo(self.fp8_dtype).max
        weight_indices_cur_rank = torch.arange(
            0,
            self.num_experts_per_partition,
            device=hidden_states.device,
            dtype=torch.int64,
        )

        # GroupGemm-0
        if hidden_states.shape[0] > 0:
            gateup_output = self.grouped_gemm_runner(
                a=hidden_states,
                b=self.w13_weight,
                c=None,
                c_dtype=hidden_states.dtype,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w13_input_scale,
                scale_b=(
                    self.w13_weight_scale_inv
                    if self.use_block_quant
                    else self.w13_weight_scale
                ),
                block_shape=self.block_shape,
            )
        else:
            gateup_output = torch.empty(
                hidden_states.shape[0],
                self.w13_weight.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        # Act
        down_input = torch.empty(
            gateup_output.shape[0],
            gateup_output.shape[1] // 2,
            device=gateup_output.device,
            dtype=(
                self.fp8_dtype
                if (self.use_fp8_w8a8 and not self.use_block_quant)
                else hidden_states_dtype
            ),
        )
        if self.w2_input_scale is None and not self.use_block_quant:
            self.w2_input_scale = torch.ones(
                self.num_experts_per_partition,
                dtype=torch.float32,
                device=hidden_states_device,
            )

        if self.activation == "silu":
            silu_and_mul_triton_kernel[(gateup_output.shape[0],)](
                gateup_output,
                down_input,
                gateup_output.shape[1],
                reorder_topk_ids,
                self.w2_input_scale,
                0,
                self.num_experts_per_partition - 1,
                BLOCK_SIZE=512,
            )
        else:
            raise ValueError(f"Unsupported activation: {self.activation=}")

        del gateup_output

        # GroupGemm-1
        down_output = torch.empty(
            down_input.shape[0],
            self.w2_weight.shape[1],
            device=hidden_states_device,
            dtype=hidden_states_dtype,
        )
        if down_input.shape[0] > 0:
            down_output = self.grouped_gemm_runner(
                a=down_input,
                b=self.w2_weight,
                c=down_output,
                batch_size=self.num_experts_per_partition,
                weight_column_major=True,
                seg_indptr=seg_indptr,
                weight_indices=weight_indices_cur_rank,
                use_fp8_w8a8=self.use_fp8_w8a8,
                scale_a=self.w2_input_scale,
                scale_b=(
                    self.w2_weight_scale_inv
                    if self.use_block_quant
                    else self.w2_weight_scale
                ),
                block_shape=self.block_shape,
            )
        return down_output

    def forward_deepgemm_contiguous(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert: List[int],
    ):
        hidden_states_fp8, hidden_states_scale = hidden_states_fp8
        assert self.quant_method is not None
        assert self.activation == "silu"
        if num_recv_tokens_per_expert is None:
            return hidden_states_fp8.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states_fp8.bfloat16()
        M, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        hidden_states_fp8_dtype = hidden_states_fp8.dtype

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            torch.empty(
                (all_tokens, K // 128),
                device=hidden_states_fp8.device,
                dtype=torch.float32,
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)

        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
        )
        dispose_tensor(hidden_states_fp8)

        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        input_tensor[1] = tma_align_input_scale(input_tensor[1])
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            input_tensor, self.w13_weight_fp8, gateup_output, m_indices
        )
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gateup_output.view(-1, N), down_input)
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(
            down_input, scale_block_size
        )
        del down_input
        down_input_scale = tma_align_input_scale(down_input_scale)
        m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
        )
        del down_input_fp8, down_input_scale

        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)

        return gather_out

    def forward_deepgemm_masked(
        self,
        hidden_states_fp8: Tuple[torch.Tensor, torch.Tensor],
        masked_m: torch.Tensor,
        expected_m: int,
    ):
        assert self.quant_method is not None
        assert self.activation == "silu"

        # GroupGemm-0
        num_groups, m, k = hidden_states_fp8[0].size()
        n = self.w13_weight.size(1)
        expected_m = min(expected_m, m)
        gateup_output = torch.empty(
            (num_groups, m, n), device=hidden_states_fp8[0].device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            hidden_states_fp8, self.w13_weight_fp8, gateup_output, masked_m, expected_m
        )
        dispose_tensor(hidden_states_fp8[0])

        # Act
        down_input = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2,
            ),
            device=gateup_output.device,
            dtype=self.fp8_dtype,
        )
        scale_block_size = 128
        down_input_scale = torch.empty(
            (
                gateup_output.shape[0],
                gateup_output.shape[1],
                gateup_output.shape[2] // 2 // scale_block_size,
            ),
            device=gateup_output.device,
            dtype=torch.float32,
        )
        silu_and_mul_masked_post_quant_fwd(
            gateup_output,
            down_input,
            down_input_scale,
            scale_block_size,
            masked_m,
        )
        del gateup_output

        # GroupGemm-1
        n = self.w2_weight.size(1)
        down_input_fp8 = (
            down_input,
            get_col_major_tma_aligned_tensor(down_input_scale),
        )
        down_output = torch.empty(
            (num_groups, m, n), device=down_input.device, dtype=torch.bfloat16
        )
        m_grouped_gemm_fp8_fp8_bf16_nt_masked(
            down_input_fp8, self.w2_weight_fp8, down_output, masked_m, expected_m
        )

        return down_output


def get_moe_impl_class():
    if global_server_args_dict["enable_deepep_moe"]:
        return DeepEPMoE
    if global_server_args_dict["enable_ep_moe"]:
        if global_server_args_dict["enable_ep_moe_heto"]:
            return EPMoEHeto
        else:
            return EPMoE
    return FusedMoE
