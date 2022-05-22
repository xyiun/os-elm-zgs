aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} cpps)

add_library (oszgs OBJECT ${cpps})
target_link_libraries(oszgs mpi)
