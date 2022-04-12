#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>

#include <glm/glm.hpp>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// The camera frame is defined with z pointing in negative viewing direction. This corresponds to the OpenGL camera/ view space
// The origin of the image frame is in the bottom left corner.
class Camera
{
public:
    Camera();
    Camera(float f_x, float f_y, float c_x, float c_y, const glm::mat4& world2cam);
    const glm::mat4& GetWorld2Cam() const;
    const glm::mat4 GetK() const;
    float GetFx() const;
    float GetFy() const;
    float GetCx() const;
    float GetCy() const;
    glm::vec3 GetPose() const;

private:
    float f_x_;
    float f_y_;
    float c_x_;
    float c_y_;
    glm::mat4 world2cam_;
};

int Y2Row(float y, int height);

template <typename Functor>
float ComputeDepth(const std::vector<glm::vec3>& point_cloud, float depth_scaling_factor, float max_depth, const Camera& camera, 
    cv::Mat& depth_map, Functor& Verify)
{
    const auto height = depth_map.rows;
    const auto width = depth_map.cols;

    const auto& K(camera.GetK());
    const auto& world2cam(camera.GetWorld2Cam());
    const auto rot_world2cam = glm::mat3(world2cam);

    int point_idx{0};
    std::size_t count{0};
    for (const auto& point : point_cloud)
    {
        if (Verify(point_idx))
        {
            const glm::vec4 point_world(point, 1.f);
            glm::vec3 point_cam = glm::vec3(world2cam * point_world);
            float z_cam(-point_cam[2]);
            if (z_cam > 0.f)
            {
                glm::vec3 point_img = glm::mat3(K) * point_cam;
                point_img = point_img / point_img[2];
                if (point_img[0] >= 0 && point_img[0] < width && point_img[1] > 0 && point_img[1] <= height)
                {
                    const auto r = Y2Row(point_img[1], height);
                    assert(r >= 0 && r < height);
                    const auto c = static_cast<int>(point_img[0]);
                    assert(c >= 0 && c < width);

                    if (z_cam <= max_depth)
                    {
                        const auto z_cam_scaled = z_cam * depth_scaling_factor;
                        auto z_cam_integer = static_cast<unsigned short>(z_cam_scaled);

                        const auto z_before = depth_map.at<unsigned short>(r, c);
                        if (z_before == 0 || z_cam_integer < z_before)
                        {
                            depth_map.at<unsigned short>(r, c) = z_cam_integer;
                            if (z_before == 0)
                            {
                                ++count;
                            }
                        }
                    }
                    else
                    {
                        std::cout << "Warning: Depth " << z_cam << " is ignored, because it is > maximal depth " << max_depth << std::endl;
                    }
                }
            }
        }
        ++point_idx;
    }
    return static_cast<float>(count) / static_cast<float>(width * height);
}

std::pair<double, std::size_t> ComputeMeanScaling(const cv::Mat& int_matrix_0, const cv::Mat& int_matrix_1);

#endif // CAMERA_H
