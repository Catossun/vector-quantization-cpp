cmake_minimum_required(VERSION 3.19)
project(VectorQuantization)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} main.cpp VectorQuantizer.cpp VectorQuantizer.h)

set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
