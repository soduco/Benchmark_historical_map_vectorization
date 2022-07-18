project(histmap)
cmake_minimum_required(VERSION 3.14)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(histmapseg
  src/main.cpp)
target_link_libraries(histmapseg PRIVATE CONAN_PKG::pylene)
