#ifndef CAMERA_FRAMES_H
#define CAMERA_FRAMES_H

#include <string>
#include <vector>

#include <camera.h>

struct CameraFrame
{
    std::string rgb_file_path{};
    std::string depth_file_path{};
    Camera camera;
};

void WriteCameraFrames(const std::string& file, const std::vector<CameraFrame>& camera_frames, float camera_near, 
    float camera_far, float depth_scaling_factor);

#endif // CAMERA_FRAMES_H
