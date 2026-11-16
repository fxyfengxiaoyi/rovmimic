cmake_minimum_required(VERSION 3.10)
project(StonefishBench CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Stonefish REQUIRED)

add_executable(sf_bench sf_bench.cpp)
target_link_libraries(sf_bench PRIVATE Stonefish::Stonefish)
set_target_properties(sf_bench PROPERTIES BUILD_RPATH "${Stonefish_DIR}/../..")
