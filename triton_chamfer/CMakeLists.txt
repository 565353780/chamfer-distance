cmake_minimum_required(VERSION 3.14)
project(triton_experiment LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CUDA_ARCHITECTURES 86)
# Enable the generation of compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# As much I prefer FetchContent, it's super annoying to setup up with CUDAExtension
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/cudaKDTree)

set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/third_party/tiny-cuda-nn)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(HDR_DIR include)
set(SRC_DIR src)

set(HDRS
    ${HDR_DIR}/crude_nn.cuh
    ${HDR_DIR}/kd_closest_query.cuh
)

set(SRCS
    ${SRC_DIR}/module.cu    
    ${SRC_DIR}/crude_nn.cu
    ${SRC_DIR}/kd_closest_query.cu
)

add_library(${PROJECT_NAME} ${HDRS} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE ${HDR_DIR})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Python_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${Python_LIBRARIES} ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} cudaKDTree tiny-cuda-nn)

add_executable(${PROJECT_NAME}_debug ${CMAKE_CURRENT_LIST_DIR}/main.cu)
target_include_directories(${PROJECT_NAME}_debug PRIVATE ${HDR_DIR})
target_link_libraries(${PROJECT_NAME}_debug PUBLIC ${PROJECT_NAME})