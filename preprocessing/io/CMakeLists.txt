add_library(io
    include/rgbd.h
    src/rgbd.cpp
    include/file_utils.h
    src/file_utils.cpp
    include/camera_frames.h
    src/camera_frames.cpp
)
target_include_directories(io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(io
    camera
    ${OpenCV_LIBS}
)
