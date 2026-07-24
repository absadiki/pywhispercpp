cmake_minimum_required(VERSION 3.4...3.18)
project(pywhispercpp)

add_subdirectory(pybind11)
add_subdirectory(whisper.cpp)

pybind11_add_module(_pywhispercpp
	src/main.cpp
	src/parakeet_bindings.cpp
)

# Link both whisper and parakeet (Parakeet is a separate library in whisper.cpp).
# Order can matter for ggml symbols on some linkers: whisper then parakeet.
target_link_libraries(_pywhispercpp PRIVATE whisper parakeet)

# Ensure parakeet.h is visible even if transitive includes change.
target_include_directories(_pywhispercpp PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/include
	${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp/ggml/include
)
