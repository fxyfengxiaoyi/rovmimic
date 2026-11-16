cmake_minimum_required(VERSION 3.15)
project(sfenv CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Stonefish REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(sfenv src/bindings.cpp)
target_include_directories(sfenv PRIVATE src)
target_link_libraries(sfenv PRIVATE Stonefish::Stonefish)
# 模块输出到 python/，脚本把这个目录加进 sys.path 即可 import sfenv。
set_target_properties(sfenv PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
  BUILD_RPATH "${Stonefish_DIR}/../..")
