cmake_minimum_required (VERSION 3.1)
include(FetchContent)

add_definitions("-Wall -g")

project (os-elm-zgs)  

set(ALL_SOURCES CACHE INTERNAL "All sources to be compiled in this project" )

set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM ${PROJECT_SOURCE_DIR})

add_subdirectory(src)

add_executable(demo1 demo1.cc ${ALL_SOURCES})
add_executable(demo2 demo2.cc ${ALL_SOURCES})
add_executable(testAgent testAgent.cc ${ALL_SOURCES})
add_executable(testMpi testMpi.cc ${ALL_SOURCES})

target_link_libraries(demo1 oszgs)
target_link_libraries(demo2 oszgs)
target_link_libraries(testAgent oszgs)
target_link_libraries(testMpi oszgs mpi)

file(COPY "./data" DESTINATION "./")
