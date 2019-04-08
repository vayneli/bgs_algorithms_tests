cmake_minimum_required(VERSION 3.5)
project(ASBLtest)
set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR /usr/local/opencv-3.1.0/share/OpenCV)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/package_bgs)

AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

add_executable(example ${DIR_SRCS})

find_package(OpenCV REQUIRED)
target_link_libraries(example ${OpenCV_LIBS})