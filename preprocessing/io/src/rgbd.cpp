#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <rgbd.h>

cv::Mat ReadRgb(const std::string& file)
{
    cv::Mat image = cv::imread(file, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    return image;
}

cv::Mat ReadDepth(const std::string& file)
{
    return cv::imread(file, cv::IMREAD_ANYDEPTH);
}

void WriteDepth(const cv::Mat& depth_image, const std::string& file)
{
    cv::imwrite(file, depth_image);
}

void WriteRgb(const cv::Mat& image, const std::string& file)
{
    cv::cvtColor(image, image, (image.channels() == 3) ? cv::COLOR_RGB2BGR : cv::COLOR_RGBA2BGRA);
    cv::imwrite(file, image);
}
