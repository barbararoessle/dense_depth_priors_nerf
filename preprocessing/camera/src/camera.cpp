#include <vector>
#include <algorithm>

#include <glm/gtc/matrix_transform.hpp>

#include <camera.h>

Camera::Camera()
    : f_x_(0.f), f_y_(0.f), c_x_(0.f), c_y_(0.f), world2cam_(1.f)
{}

Camera::Camera(float f_x, float f_y, float c_x, float c_y, const glm::mat4 &world2cam)
    : f_x_(f_x), f_y_(f_y), c_x_(c_x), c_y_(c_y), world2cam_(world2cam)
{}

const glm::mat4& Camera::GetWorld2Cam() const
{
    return world2cam_;
}

const glm::mat4 Camera::GetK() const
{
    glm::mat4 K(1.f);
    K[0][0] = f_x_;
    K[1][1] = f_y_;
    K[2][0] = -c_x_;
    K[2][1] = -c_y_;
    K[2][2] = -1.f;
    return K;
}

float Camera::GetFx() const
{
    return f_x_;
}

float Camera::GetFy() const
{
    return f_y_;
}

float Camera::GetCx() const
{
    return c_x_;
}

float Camera::GetCy() const
{
    return c_y_;
}

glm::vec3 Camera::GetPose() const
{
    const glm::mat3 rot_world2cam(world2cam_);
    const glm::mat3 rot_cam2world = glm::transpose(rot_world2cam);
    const glm::vec3 pose(- rot_cam2world * glm::vec3(world2cam_[3]));
    return pose;
}

int Y2Row(float y, int height)
{
    return static_cast<int>(static_cast<float>(height) - y);
}

std::pair<double, std::size_t> ComputeMeanScaling(const cv::Mat& int_matrix_0, const cv::Mat& int_matrix_1)
{
    assert(int_matrix_0.size() == int_matrix_1.size());
    double sum_scaling_factors{0};
    std::size_t valid_count{0};

    cv::MatConstIterator_<unsigned short> it_0(int_matrix_0.begin<unsigned short>()), it_1(int_matrix_1.begin<unsigned short>()), et_0(int_matrix_0.end<unsigned short>());
    for(; it_0 != et_0; ++it_0, ++it_1)
    {
        // both are valid
        if (*it_0 != 0 && *it_1 != 0)
        {
            sum_scaling_factors += static_cast<double>(*it_0) / static_cast<double>(*it_1);
            ++valid_count;
        }
    }
    return std::make_pair(sum_scaling_factors / static_cast<double>(valid_count), valid_count);
}

