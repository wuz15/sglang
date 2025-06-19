/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel_esimd, m) {
  /*
   * From csrc/gemm
   */
  m.def("esimd_mul_lgrf(Tensor a, Tensor b, Tensor c, int flag, int len) -> Tensor");
  m.impl("esimd_mul_lgrf", torch::kXPU, &esimd_kernel_mul_lgrf);

  //   m.def(
  //       "fp8_blockwise_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype,
  //       -> Tensor");
  //   m.impl("fp8_blockwise_scaled_mm", torch::kXPU, &fp8_blockwise_scaled_mm);
}

REGISTER_EXTENSION(common_ops_lgrf)
