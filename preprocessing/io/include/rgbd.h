#ifndef RGBD_WRITER_H
#define RGBD_WRITER_H

#include <string>

#include <opencv2/core/mat.hpp>

cv::Mat ReadRgb(const std::string& file);

cv::Mat ReadDepth(const std::string& file);

void WriteDepth(const cv::Mat& depth_image, const std::string& file);

void WriteRgb(const cv::Mat& image, const std::string& file);

#endif // RGBD_WRITER_H
