cmake_minimum_required(VERSION 3.10)
project(a-perceptron VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include_directories(include)

set(SOURCES
    src/main.cpp
    src/perceptron.cpp
    src/gradiente-descent.cpp
    src/mlp.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})
