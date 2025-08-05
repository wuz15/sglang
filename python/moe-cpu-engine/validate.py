import torch
import moe_cpu_engine as engine
import moe_offload_config as moc

# Offload Config (expert_idx list for each layer)
offcfg = moc.create_offload_config()
print(offcfg.get_offload_experts(3))



# Create an instance of CPUInfer
cpu_infer = engine.CPUInfer(0)  # Pass an integer argument to the constructor

# layer_id 3~4 example
for layer_id in range(3, 4):
    # Access the `moe` submodule
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

    validate = True
    # Load the expected result
    if validate:
        e_score_correction_bias_path = "test_data/e_score_correction_bias.pt"
        w13_weight_path = "test_data/w13_weight.pt"
        w13_weight_scale_inv_path = "test_data/w13_weight_scale_inv.pt"
        w2_weight_path = "test_data/w2_weight.pt"
        w2_weight_scale_inv_path = "test_data/w2_weight_scale_inv.pt"
        hidden_states_path = "test_data/hidden_states.pt"
        router_logits_path = "test_data/router_logits.pt"
        expected_res_path = "test_data/expected_res.pt"
        topk_ids_path = "test_data/topk_ids.pt"
        topk_weights_path = "test_data/topk_weights.pt"
    
        e_score_correction_bias = (torch.load(e_score_correction_bias_path, map_location=torch.device('cpu'), weights_only=True)).to(torch.float)
        gate_up_weights = torch.load(w13_weight_path, map_location=torch.device('cpu'), weights_only=True)
        gate_up_scales = torch.load(w13_weight_scale_inv_path, map_location=torch.device('cpu'), weights_only=True)
        down_weights = torch.load(w2_weight_path, map_location=torch.device('cpu'), weights_only=True)
        down_scales = torch.load(w2_weight_scale_inv_path, map_location=torch.device('cpu'), weights_only=True)
        input_tensor = torch.load(hidden_states_path, map_location=torch.device('cpu'), weights_only=True)
        router_logits = torch.load(router_logits_path, map_location=torch.device('cpu'), weights_only=True)
        expected_res = torch.load(expected_res_path, map_location=torch.device('cpu'), weights_only=True)
        topk_ids = torch.load(topk_ids_path, map_location=torch.device('cpu'))
        topk_weights = torch.load(topk_weights_path, map_location=torch.device('cpu'))

        n_tokens = input_tensor.shape[0]
    else:
        n_tokens = 1
        e_score_correction_bias = torch.zeros(256, dtype=torch.float)
        gate_up_weights = torch.ones((256, 4096, 7168), dtype=torch.float8_e4m3fn)
        down_weights = torch.ones((256, 7168, 2048), dtype=torch.float8_e4m3fn)
        gate_up_scales = torch.full((256, 32, 56), 0.1, dtype=torch.float)
        down_scales = torch.full((256, 56, 16), 0.1, dtype=torch.float)
        input_tensor = torch.full((n_tokens, 7168), 0.1, dtype=torch.bfloat16)
        router_logits = torch.full((n_tokens, 256), 0.1, dtype=torch.bfloat16)
        tmp_ids = torch.tensor([0,1,2,3,4,5,6,7], dtype=torch.int)
        topk_ids = tmp_ids.repeat(n_tokens, 1)
        topk_weights = torch.full((n_tokens, 8), 0.125, dtype=torch.float)

    output_tensor = torch.zeros((n_tokens, 7168), dtype=torch.bfloat16)

    # example for sglang
    moe = engine.moe.MOE(moe_config)
    moe.set_weights(gate_up_weights.data_ptr(), down_weights.data_ptr(), gate_up_scales.data_ptr(), down_scales.data_ptr(),
            e_score_correction_bias.data_ptr())
    print(input_tensor)
    # Call the `forward` method, routed_scaling_factor = 1.0
    cpu_infer.submit(moe.forward(input_tensor.data_ptr(), router_logits.data_ptr(), output_tensor.data_ptr(), n_tokens))
    cpu_infer.sync()
    print(output_tensor)

    output_tensor = torch.zeros((n_tokens, 7168), dtype=torch.bfloat16)
    cpu_infer.submit(moe.forward_experts(input_tensor.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
            output_tensor.data_ptr(), n_tokens))
    cpu_infer.sync()
    print(output_tensor)
    print(expected_res)
    
    if validate:
        try:
            torch.testing.assert_close(
                output_tensor,
                expected_res,
                rtol=1e-03,
                atol=1e-02,
                msg="Output does not match expected result."
            )
            print("#################### Success #########################")
        except AssertionError as e:
            print("#################### Mismatch #########################")
            print("FusedMoE output: ", output_tensor)
            print("Expected result: ", expected_res)

