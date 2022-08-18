cmake_minimum_required(VERSION 2.8)

PROJECT(NISwGSP)

FIND_PACKAGE(OpenMP REQUIRED)

#eigen
FIND_PACKAGE(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DEIGEN_DONT_PARALLELIZE -std=c++11")


#opencv
find_package(OpenCV REQUIRED)	
include_directories(${OpenCV_INCLUDE_DIRS})


#vlfeat
FIND_LIBRARY(VLFEAT_LIBRARY libvl)

aux_source_directory(./Debugger DEBUGGER_SRC)
aux_source_directory(./Feature FEATURE_SRC)
aux_source_directory(./Mesh MESH_SRC)
aux_source_directory(./Stitching STITCHING_SRC)
aux_source_directory(./Util UTIL_SRC)

add_executable(NISwGSP Configure.h main.cpp ${DEBUGGER_SRC} ${FEATURE_SRC} ${MESH_SRC} ${STITCHING_SRC} ${UTIL_SRC})
target_link_libraries(NISwGSP ${OpenCV_LIBS} ${VLFEAT_LIBRARY})
