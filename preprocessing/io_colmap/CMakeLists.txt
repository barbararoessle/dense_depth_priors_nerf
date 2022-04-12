add_library(io_colmap
    include/colmap_reader.h
    src/colmap_reader.cpp
)
target_include_directories(io_colmap PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(io_colmap
    camera
    io
)
