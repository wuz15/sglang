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

  m.def("esimd_kernel_uni_lgrf(Tensor t0, Tensor t1, Tensor t2, Tensor t3, Tensor t4, Tensor t5, Tensor t6, Tensor t7, Tensor t8, Tensor t9, \
                          int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7, int i8, int i9, \
                          float f0, float f1, float f2, float f3, float f4) -> Tensor");
  m.impl("esimd_kernel_uni_lgrf", torch::kXPU, &esimd_kernel_uni_lgrf);
}

REGISTER_EXTENSION(common_ops_lgrf)
