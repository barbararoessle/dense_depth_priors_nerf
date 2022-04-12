#ifndef COLMAP_READER_H
#define COLMAP_READER_H

#include <string>
#include <vector>

#include <camera.h>

struct CameraConfig
{
    int id{0};
    std::string rgb_image{};
    int width{0};
    int height{0};
    Camera camera;
};

std::vector<CameraConfig> ReadCameras(const std::string& cameras_txt, const std::string& images_txt, float dist2m);

void ReadSparsePointCloud(const std::string& points3D_txt, float scale, float max_reprojection_error, unsigned int min_track_length,
                          std::vector<glm::vec3>& point_cloud, std::vector<std::vector<int>>& visibility);

#endif // COLMAP_READER_H
