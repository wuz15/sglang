cmake_minimum_required(VERSION 3.19.2)

set(CMAKE_C_COMPILER icx CACHE STRING "Intel C compiler")
set(CMAKE_CXX_COMPILER icpx CACHE STRING "Intel C++ compiler")

# —————————————————————————————————————————————
execute_process(
  COMMAND which icpx
  OUTPUT_VARIABLE ICPCX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT ICPCX_PATH)
  message(FATAL_ERROR "icpx not found in PATH; please source oneAPI setvars.sh or adjust your PATH")
endif()
message(STATUS "Found icpx at: ${ICPCX_PATH}")

set(CMAKE_CXX_COMPILER "${ICPCX_PATH}"
    CACHE STRING "Path to DPC++ compiler")

get_filename_component(ICPCX_BIN   "${ICPCX_PATH}" DIRECTORY)  # bin
get_filename_component(ICPCX_LINUX "${ICPCX_BIN}" DIRECTORY)   # root

set(SYCL_CONFIG_DIR "${ICPCX_LINUX}/lib/cmake/IntelSYCL"
    CACHE PATH "Auto-detected SYCLConfig.cmake directory")
if(EXISTS "${SYCL_CONFIG_DIR}/IntelSYCLConfig.cmake")
  message(STATUS "Auto-detected SYCL config dir: ${SYCL_CONFIG_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${SYCL_CONFIG_DIR}")
else()
  message(WARNING "Could not find IntelSYCLConfig.cmake in ${SYCL_CONFIG_DIR}; you may need to install oneAPI or adjust SYCL_CONFIG_DIR")
endif()
# —————————————————————————————————————————————

project(my_sycl_project LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

# Torch
find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE}
          -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PY_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX})
find_package(Torch REQUIRED)

include(FetchContent)

# cutlass
FetchContent_Declare(
    repo-cutlass-fork
    GIT_REPOSITORY https://github.com/codeplaysoftware/cutlass-fork.git
    GIT_TAG        268d735eda8dbd27e12f0596f30be3e6edb2c8db
    GIT_SHALLOW    OFF
)
FetchContent_Populate(repo-cutlass-fork)

# ccache option
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/../../include
    ${PROJECT_SOURCE_DIR}/../../csrc
    ${repo-cutlass-fork_SOURCE_DIR}/include
    ${repo-cutlass-fork_SOURCE_DIR}/tools/util/include
)

set(SOURCES
    "${PROJECT_SOURCE_DIR}/awq_dequantize.cpp"
    "${PROJECT_SOURCE_DIR}/torch_extension_sycl.cc"
)

Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

# SYCL related
string(STRIP "${SYCL_CXX_FLAGS}" SYCL_CXX_FLAGS_STRIPPED)
target_compile_options(common_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${SYCL_CXX_FLAGS_STRIPPED}>)

target_compile_definitions(common_ops PRIVATE PYBIND11_COMPILE_TYPE="_gcc")
target_compile_definitions(common_ops PRIVATE PYBIND11_STDLIB="_libstdcpp")
target_compile_definitions(common_ops PRIVATE PYBIND11_ABI="_cxxabi1016")

target_include_directories(common_ops PRIVATE
    PUBLIC ${SYCL_ROOT}/include/sycl
    PUBLIC ${SYCL_ROOT}/include
)

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} ${SYCL_ROOT}/lib)

# torch related
target_include_directories(common_ops PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})

# install
install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)

install(DIRECTORY "${repo-cutlass-fork_SOURCE_DIR}/include/cute/"
        DESTINATION "deep_gemm/include/cute")

install(DIRECTORY "${repo-cutlass-fork_SOURCE_DIR}/include/cutlass/"
        DESTINATION "deep_gemm/include/cutlass")
