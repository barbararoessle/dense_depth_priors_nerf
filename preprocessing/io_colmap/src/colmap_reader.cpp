#include <iostream>
#include <fstream>
#include <unordered_map>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <colmap_reader.h>
#include <file_utils.h>

std::unordered_map<int, std::pair<std::string, std::array<float, 6>>> ReadIntrinsics(const std::string& cameras_txt)
{
    std::unordered_map<int, std::pair<std::string, std::array<float, 6>>> result{};
    std::ifstream file(cameras_txt);
    std::string line;
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ' ');
        if (elements[0] == std::string("#"))
        {
            continue;
        }
        if ("PINHOLE" == elements[1])
        {
            if (elements.size() == 8)
            {
                // width, height, fx, fy, cx, cy
                result[std::stoi(elements[0])] = std::make_pair(std::string{"PINHOLE"},
                    std::array<float, 6>{std::stof(elements[2]), std::stof(elements[3]), std::stof(elements[4]), 
                    std::stof(elements[5]), std::stof(elements[6]), std::stof(elements[7])});
            }
            else
            {
                std::cout << "Error: Invalid pinhole camera" << std::endl;
            }
        }
        if ("SIMPLE_PINHOLE" == elements[1])
        {
            if (elements.size() == 7)
            {
                // width, height, f, cx, cy
                result[std::stoi(elements[0])] = std::make_pair(std::string{"SIMPLE_PINHOLE"},
                    std::array<float, 6>{std::stof(elements[2]), std::stof(elements[3]), std::stof(elements[4]), 
                    std::stof(elements[5]), std::stof(elements[6]), 0.f});
            }
            else
            {
                std::cout << "Error: Invalid simple pinhole camera" << std::endl;
            }
        }
        else
        {
            std::cout << "Warning: Camera model " << elements[1] << " not implemented." << std::endl;
        }
    }
    file.close();
    if (result.empty())
    {
        std::cout << "Error: Could not read intrinsics" << std::endl;
    }
    return result;
}

std::vector<CameraConfig> ReadCameras(const std::string& cameras_txt, const std::string& images_txt, float dist2m)
{
    std::vector<CameraConfig> result{};

    const auto& intrinsics = ReadIntrinsics(cameras_txt);

    std::ifstream file(images_txt);
    std::string line;
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ' ');
        if (elements.size() != 10 || elements[0] == std::string("#"))
        {
            continue;
        }

        glm::quat quaternion(std::stof(elements[1]), std::stof(elements[2]), std::stof(elements[3]), std::stof(elements[4]));
        // rotation from colmap camera frame (z in positive viewing direction and image origin in top left corner)
        // to internal camera frame (z in negative viewing direction and image origin in bottom left corner)
        const glm::mat4 rot_cam2cam = glm::rotate(glm::mat4(1.0f), glm::radians(180.f), glm::vec3(1.0f, 0.0f, 0.0f));
        glm::mat4 rot_world2cam = glm::mat4_cast(quaternion);
        glm::mat4 tra_world2cam = glm::mat4(1.f);
        tra_world2cam[3] = glm::vec4(dist2m * std::stof(elements[5]), dist2m * std::stof(elements[6]), dist2m * std::stof(elements[7]), 1.f);
        const glm::mat4 world2cam = rot_cam2cam * tra_world2cam * rot_world2cam;

        const auto& model_and_intrinsic = intrinsics.at(std::stoi(elements[8]));
        const auto& model = model_and_intrinsic.first;
        const auto& intrinsic = model_and_intrinsic.second;
        const float w = intrinsic[0];
        const float h = intrinsic[1];
        const float f_x = intrinsic[2];
        float f_y(0.f), c_x(0.f), c_y(0.f);
        if (model == "SIMPLE_PINHOLE")
        {
            f_y = f_x;
            c_x = intrinsic[3];
            c_y = intrinsic[4];
        }
        else if (model == "PINHOLE")
        {
            f_y = intrinsic[3];
            c_x = intrinsic[4];
            c_y = intrinsic[5];
        }

        c_y = h - c_y;
        Camera camera(f_x, f_y, c_x, c_y, world2cam);
        result.emplace_back(CameraConfig{std::stoi(elements[0]), elements[9], static_cast<int>(w), static_cast<int>(h), camera});
    }

    file.close();

    return result;
}

void ReadSparsePointCloud(const std::string& points3D_txt, float scale, float max_reprojection_error, unsigned int min_track_length,
                          std::vector<glm::vec3>& point_cloud, std::vector<std::vector<int>>& visibility)
{
    std::ifstream file(points3D_txt);
    std::string line;
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, ' ');
        if (elements[0] == std::string("#"))
        {
            continue;
        }
        const auto reproj_err = std::stof(elements[7]);
        if (reproj_err <= max_reprojection_error)
        {
            std::vector<int> point_visibility{};
            for (auto it(elements.cbegin() + 8); it < elements.end();)
            {
                point_visibility.emplace_back(std::stoi(*it));
                it = it + 2;
            }
            if (point_visibility.size() >= min_track_length)
            {
                point_cloud.emplace_back(glm::vec3{scale * std::stof(elements[1]), scale * std::stof(elements[2]), scale * std::stof(elements[3])});
                visibility.emplace_back(point_visibility);
            }
        }
    }
}
