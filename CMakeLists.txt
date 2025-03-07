cmake_minimum_required(VERSION 3.5)
project(cudaFHE LANGUAGES CXX)

add_library(cudaFHE SHARED IMPORTED)

set_target_properties(cudaFHE PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/libcudaFHE.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

set(cudaFHE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)

install(
    FILES "${CMAKE_CURRENT_SOURCE_DIR}/lib/libcudaFHE.so"
    DESTINATION lib
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION include
)
