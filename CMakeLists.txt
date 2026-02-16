cmake_minimum_required(VERSION 3.15)
project(QwenTokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG -flto")
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/reasoners/tokenizer/Qwen/Qwen2.5-1.5B-Instruct)

pybind11_add_module(qwen_tokenizer_cpp 
  reasoners/tokenizer/Qwen/Qwen2.5-1.5B-Instruct/python_bindings.cpp
)

target_link_libraries(qwen_tokenizer_cpp PRIVATE nlohmann_json::nlohmann_json)

