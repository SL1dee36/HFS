cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
project(HFS)

add_executable(HFS src/HSF.cpp)

add_subdirectory(external/glfw)
add_subdirectory(external/glad)

target_link_libraries(HFS glfw glad)

include_directories(external/glm)