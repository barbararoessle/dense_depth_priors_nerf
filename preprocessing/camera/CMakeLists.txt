add_library(camera
    include/camera.h
    src/camera.cpp
)
target_include_directories(camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
