cmake_minimum_required(VERSION 3.11)

project(ogl_renderer LANGUAGES CXX)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)
find_package(glm REQUIRED)
find_package(RapidJSON REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})

add_executable(extract_scannet_scene
    extract_scannet_scene.cpp
)

target_link_libraries(extract_scannet_scene
    io
    io_colmap
    camera
    ${Boost_FILESYSTEM_LIBRARY}
)

add_subdirectory(io)
add_subdirectory(io_colmap)
add_subdirectory(camera)
