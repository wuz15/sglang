# Need CUDA: https://developer.nvidia.com/cuda-downloads
from setuptools import setup, Extension
import os
from torch.utils import cpp_extension
from setuptools.command.build_ext import build_ext
from distutils.spawn import find_executable
import shutil
import subprocess
from torch.utils.cpp_extension import CppExtension
import importlib.util

def get_module_path(module_name):
    """Get the file path of a module."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ImportError(f"Module {module_name} not found")
    module_path = spec.submodule_search_locations[0]
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module path {module_path} does not exist")
    return module_path

module_path = get_module_path('xfastertransformer-devel')
mlib_path = os.path.join(module_path, 'lib')
minc_path = os.path.join(module_path, 'include')

CUDA_HOME="/usr/local/cuda-12/targets/x86_64-linux/"

# Import cpp_extension for CppExtension
class CustomBuildExt(build_ext):
    def run(self):
        super().run()
        # Compile worker_entry.cpp into an executable during install
        build_dir = os.path.abspath(self.build_lib)
        # Use install_lib if available (for install), else build_lib (for build)
        install_lib = getattr(self, 'install_lib', build_dir)
        bin_dir = os.path.join(install_lib, 'bin')
        os.makedirs(bin_dir, exist_ok=True)
        src_files = [
            os.path.abspath('worker_entry.cpp'),
            os.path.abspath('numa_launcher.cpp'),
            os.path.abspath('utils/shm_ccl.cpp'),
            os.path.abspath('moe.cpp'),
            os.path.abspath('utils/communicator.cpp'),
        ]
        exe_file = os.path.join(bin_dir, 'worker_entry')
        cxx = os.environ.get("CXX", 'g++')
        compile_args = [
            cxx, *src_files, '-o', exe_file,
            '-O3', '-fPIC', '-std=c++17',
            '-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq',
            '-mavx512bf16', '-mavx512vnni', '-fopenmp',
            '-Wl,--allow-shlib-undefined',
            f'-I{os.path.abspath("./common")}',
            f'-I{os.path.abspath("./utils")}',
            f'-I{minc_path}',
            f'-I{CUDA_HOME}/include',
            f'-I{os.environ['CONDA_PREFIX']}/include',
            f'-L{CUDA_HOME}/lib', '-lcudart',
            f'-L{mlib_path}', '-lxfastertransformer', '-lrt',
            '-lnuma'
        ]
        self.announce(f"Compiling {src_files} to {exe_file}", level=3)
        if not shutil.which(cxx):
            raise RuntimeError(f"C++ compiler '{cxx}' not found")
        subprocess.check_call(compile_args)

cmdclass = {'build_ext': CustomBuildExt}

setup(
    name='moe_cpu_engine',
    version='0.1',
    py_modules=['moe_offload_config'],
    ext_modules=[
        cpp_extension.CppExtension(
            'moe_cpu_engine',
            [
                'engine_bindings.cpp',
                'moe.cpp',
                'task_queue.cpp',
                'numa_launcher.cpp',
                'utils/communicator.cpp',
                'utils/numa_detector.cpp',
                'utils/shm_ccl.cpp',
                'utils/logger.cpp',
            ],
            include_dirs=[
                os.path.abspath('./common'),
                os.path.abspath('./utils'),
                minc_path,
                os.path.join(CUDA_HOME, 'include'),
            ],
            extra_compile_args=[
                '-O3',
                '-fPIC',
                '-std=c++17',
                '-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq',
                '-mavx512bf16', '-mavx512vnni',
                '-Wl,--allow-shlib-undefined',
                '-fopenmp'
            ],
            extra_link_args=[
                '-L' + os.path.join(CUDA_HOME, 'lib'), '-lcudart',
                '-L' + mlib_path, '-lxfastertransformer'
            ]
        )
    ],
    cmdclass=cmdclass
)
