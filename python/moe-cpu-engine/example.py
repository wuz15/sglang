import torch
import moe_cpu_engine as engine
import moe_offload_config as moc
import time
from tqdm import tqdm

# Offload Config (expert_idx list for each layer)
offcfg = moc.create_offload_config()
print(offcfg.get_offload_experts(3))



# Create an instance of CPUInfer
cpu_infer = engine.CPUInfer(0)  # Pass an integer argument to the constructor

# create all layers (3~61) moe, and set weights, store moe for the forward later
layers_moe = []
for layer_id in tqdm(range(3, 7), desc="Loading Layers"):
    moe_config = engine.moe.MOEConfig(
        layer_id,
        256,
        8,
        7168,
        2048,
        True,
        8,
        4
    )

    gate_up_weights = torch.ones((256, 4096, 7168), dtype=torch.float8_e4m3fn)
    down_weights = torch.ones((256, 7168, 2048), dtype=torch.float8_e4m3fn)
    gate_up_scales = torch.full((256, 32, 56), 0.1, dtype=torch.float)
    down_scales = torch.full((256, 56, 16), 0.1, dtype=torch.float)

    e_score_correction_bias = torch.zeros(256, dtype=torch.float)

    # example for sglang
    moe = engine.moe.MOE(moe_config)
    moe.set_weights(gate_up_weights.data_ptr(), down_weights.data_ptr(), gate_up_scales.data_ptr(), down_scales.data_ptr(), e_score_correction_bias.data_ptr())
    layers_moe.append(moe)

n_tokens = 1
gen_tokens = 256

use_gpu = False
dev_str = "cpu"
pin = False
if torch.cuda.is_available():
    use_gpu = True
    dev_str = "cuda"
    pin = True

input_tensor = torch.full((n_tokens, 7168), 0.1, dtype=torch.bfloat16)
gpu_output_tensor = torch.empty((n_tokens, 7168), dtype=torch.bfloat16, device=dev_str)
output_tensor = torch.zeros((n_tokens, 7168), dtype=torch.bfloat16, device="cpu", pin_memory=pin)
# generate rand tmp_ids and tmp_weights
topk_ids = torch.randint(0, 256, (n_tokens, 8), dtype=torch.int)
topk_weights = torch.rand(n_tokens, 8)
for t in tqdm(range(gen_tokens), desc="generating"):
    if t == 1:
        t0 = time.time()
    # inference
    for moe in layers_moe:
        if use_gpu:
            cpu_infer.submit_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream,
                moe.forward_experts(input_tensor.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
                output_tensor.data_ptr(), n_tokens))

            cpu_infer.sync_with_cuda_stream(
                torch.cuda.current_stream().cuda_stream)

            gpu_output_tensor.copy_(output_tensor, non_blocking=True)
            #torch.cuda.current_stream().synchronize()
        else:
            cpu_infer.submit(
                moe.forward_experts(input_tensor.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
                output_tensor.data_ptr(), n_tokens))

            cpu_infer.sync()
if use_gpu:
    print(gpu_output_tensor)
else:
    print(output_tensor)
t1 = time.time()
print("Inference time (ms): ", (t1 - t0) * 1000.0 / (gen_tokens - 1))
