# Copyright 2025 SGLang Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
from pathlib import Path

from setuptools import find_packages, setup
# from torch.utils.cpp_extension import BuildExtension, SyclExtension
from torch.utils.cpp_extension import SyclExtension
from esimd_build_extention import BuildExtension

root = Path(__file__).parent.resolve()


def _get_version():
    with open(root / "pyproject_sycl.toml") as f:
        for line in f:
            if line.startswith("version"):
                return line.split("=")[1].strip().strip('"')


operator_namespace = "sgl_kernel_esimd"
include_dirs = [
    root / "include",
    root / "csrc",
]

sources = [
    "csrc/xpu/awq_dequantize.sycl",
    "csrc/xpu/uni_esimd_kernel.sycl",
    "csrc/xpu/torch_extension_sycl.cc",
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "sycl": ["-ffast-math", "-fsycl-device-code-split=per_kernel"],
}

extra_link_args = ["-Wl,-rpath,$ORIGIN/../../torch/lib", "-L/usr/lib/x86_64-linux-gnu"]

ext_modules = []
ext_modules.append(
    SyclExtension(
        name="sgl_kernel_esimd.common_ops",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )
)

### for lgrf esimd kernels
sources = [
    "csrc/xpu/uni_esimd_kernel_lgrf.sycl",
    "csrc/xpu/torch_extension_sycl_lgrf.cc",
]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "sycl": ["-fsycl", "-ffast-math", "-fsycl-device-code-split=per_kernel", "-Xs '-options -doubleGRF'"],
}

ext_modules.append(
    SyclExtension(
        name="sgl_kernel_esimd.common_ops_lgrf",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        py_limited_api=False,
    )
)
### for lgrf esimd kernels

setup(
    name="sgl-kernel-esimd",
    version=_get_version(),
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    options={"bdist_wheel": {"py_limited_api": "cp10"}},
)
