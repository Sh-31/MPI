cmake_minimum_required(VERSION 3.0)
project(main)


find_package(OpenCV REQUIRED)
find_package(MPI REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")

add_executable(main D:/Repo/MPI/main.cpp)

target_link_libraries(main ${OpenCV_LIBS} "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W0)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -w)
endif()