cmake_minimum_required(VERSION 3.4...3.18)
project(pywhispercpp)

add_subdirectory(pybind11)
add_subdirectory(whisper.cpp)

pybind11_add_module(_pywhispercpp
	src/main.cpp
)

target_link_libraries (_pywhispercpp PRIVATE whisper)

