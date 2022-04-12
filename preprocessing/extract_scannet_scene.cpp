#include <string>
#include <iostream>
#include <fstream>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>

#include <rgbd.h>
#include <file_utils.h>
#include <camera_frames.h>
#include <colmap_reader.h>

struct VisibilityCheck
{
    bool operator() (int point_idx) const
    {
        const auto& image_idxs = visibility_[point_idx];
        return image_idxs.cend() != std::find(image_idxs.cbegin(), image_idxs.cend(), image_idx_);
    }

    const int image_idx_{0};
    const std::vector<std::vector<int>>& visibility_{};
};

struct ColmapHelper
{
    void Init(const boost::filesystem::path& path2colmap, float dist2m)
    {
        const auto path2sparse_train = path2colmap / "sparse_train" / "0";
        train_camera_configs_ = ReadCameras((path2sparse_train / "cameras.txt").string(), (path2sparse_train / "images.txt").string(), 
            dist2m);
        const auto path2sparse = path2colmap / "sparse" / "0";
        camera_configs_ = ReadCameras((path2sparse / "cameras.txt").string(), (path2sparse / "images.txt").string(), dist2m);
        float max_reproj_error{std::numeric_limits<float>::max()};
        unsigned int min_track_length{0};
        ReadSparsePointCloud((path2sparse_train / "points3D.txt").string(), dist2m, max_reproj_error, min_track_length, 
            train_point_cloud_, train_visibility_);
        ReadSparsePointCloud((path2sparse / "points3D.txt").string(), dist2m, max_reproj_error, min_track_length, 
            point_cloud_, visibility_);
    }

    CameraConfig GetCameraConfig(const std::string& filename, bool test_camera = true) const
    {
        bool found{false};
        CameraConfig camera_config{};
        for (const auto& config : (test_camera ? camera_configs_ : train_camera_configs_))
        {
            if (config.rgb_image == filename)
            {
                found = true;
                camera_config = config;
                break;
            }
        }
        if (!found)
        {
            std::cout << "Error: Camera config " << filename << " was not found in Colmap reconstruction" << std::endl;
        }
        return camera_config;
    }
    void GetSparseDepth(const std::string& filename, const std::string& dataset_type, cv::Mat& sparse_depth, float max_depth, 
        float depth_scaling_factor, const cv::Mat& target_depth)
    {
        // compute sparse depth to determine scaling using sparse reconstruction from all images
        const auto config = GetCameraConfig(filename);
        const auto curr_colmap_id = config.id;
        VisibilityCheck visibility_check{curr_colmap_id, visibility_};
        ComputeDepth(point_cloud_, depth_scaling_factor, max_depth, config.camera, sparse_depth, visibility_check);
        // compute scaling between colmap and target depth
        const auto scaling_result = ComputeMeanScaling(target_depth, sparse_depth);
        const auto scaling = scaling_result.first;
        const auto count = scaling_result.second;
        if (count > 0)
        {
            global_scaling_ = static_cast<double>(global_scaling_count_) / static_cast<double>(global_scaling_count_ + count) * global_scaling_ + 
                static_cast<double>(count) / static_cast<double>(global_scaling_count_ + count) * scaling;
        }
        global_scaling_count_ += count;

        // compute train sparse depth using sparse reconstruction from train images
        if (dataset_type == "train")
        {
            const auto train_config = GetCameraConfig(filename, false);
            const auto train_curr_colmap_id = train_config.id;
            VisibilityCheck train_visibility_check{train_curr_colmap_id, train_visibility_};
            sparse_depth = cv::Mat::zeros(sparse_depth.rows, sparse_depth.cols, CV_16UC1); // reset to zero
            const auto percent = ComputeDepth(train_point_cloud_, depth_scaling_factor, max_depth, train_config.camera, 
                sparse_depth, train_visibility_check);
            sum_percent_ += percent;
            ++count_;
            if (percent < 1e-6)
            {
                std::cout << "Warning: No train sparse depth in " << filename << std::endl;
            }
        }
    }
    float GetPercentValid() const
    {
        return sum_percent_ / static_cast<float>(count_);
    }
    double GetScaling() const
    {
        return global_scaling_;
    }
  private:
    // sparse reconstruction from train images
    std::vector<CameraConfig> train_camera_configs_{};
    std::vector<glm::vec3> train_point_cloud_{};
    std::vector<std::vector<int>> train_visibility_{};
    // sparse reconstruction from all images
    std::vector<CameraConfig> camera_configs_{};
    std::vector<glm::vec3> point_cloud_{};
    std::vector<std::vector<int>> visibility_{};
    int count_{0}; // count processed train files
    float sum_percent_{0.f}; // sum pf percentage of train sparse depth
    double global_scaling_{0.};
    std::size_t global_scaling_count_{0};
};

struct SceneConfig
{
    std::string kName{}; // name of the scene
    float kMaxDepth{}; // maximal depth value in the scene, larger values are invalidated
    float kDist2M{}; // scaling factor that scales the sparse reconstruction to meters
    bool kRgbOnly{}; // write rgb only, f.ex. to get input for colmap
};

SceneConfig LoadConfig(const boost::filesystem::path& path2scene)
{
    std::cout << "Loading config for " << path2scene.string() << std::endl;
    std::ifstream ifs((path2scene / "config.json").string());
    if (!ifs.is_open())
    {
        std::cout << "Error: Could not open config.json" << std::endl;
    }
    rapidjson::IStreamWrapper isw(ifs);

    rapidjson::Document d;
    d.ParseStream(isw);

    return SceneConfig{d["name"].GetString(), static_cast<float>(d["max_depth"].GetDouble()), 
        static_cast<float>(d["dist2m"].GetDouble()), d["rgb_only"].GetBool()};
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ./extract_scannet_scene <path to scene with config.json> <path to ScanNet>" << std::endl;
        return 0;
    }
    boost::filesystem::path path2scene(argv[1]);
    boost::filesystem::path path2scannet(argv[2]);
    const auto& config = LoadConfig(path2scene);
    boost::filesystem::path path2scannetscene(path2scannet / "scans_test" / config.kName);

    // constants across all scenes
    constexpr float kDepthScalingFactor{1000.f};
    constexpr int kWidth{640};
    constexpr int kHeight{480};

    // read reconstruction
    ColmapHelper recon;
    if (!config.kRgbOnly)
    {
        recon.Init(path2scene / "colmap", config.kDist2M);
    }

    float max_depth{0.f};
    std::unordered_map<std::string, std::vector<CameraFrame>> camera_frames{{"train", {}}, {"test", {}}};
    
    for (const std::string& dataset_type : {"train", "test"})
    {
        const boost::filesystem::path path2scene_type(path2scene / dataset_type);
        boost::filesystem::create_directory(path2scene_type);
        for (const std::string& subdir : {"rgb", "depth", "target_depth"})
        {
            const boost::filesystem::path path2subdir(path2scene_type / subdir);
            boost::filesystem::create_directory(path2subdir);
        }
        const boost::filesystem::path path2csv(path2scene / (dataset_type + "_set.csv"));
        std::vector<std::string> filenames = ReadSequence<std::string>(path2csv.string());

        for(const auto& filename : filenames)
        {
            // read rgb
            auto rgb = ReadRgb((path2scannetscene / "color" / filename).string());
            
            // fix different aspect ratio between rgb and depth
            if (rgb.cols == 1296 && rgb.rows == 968)
            {
                int border = 2;
                cv::Mat rgb_padded(rgb.rows + border * 2, rgb.cols, rgb.depth()); // construct with padding
                cv::copyMakeBorder(rgb, rgb_padded, border, border, 0, 0, cv::BORDER_CONSTANT);
                rgb = rgb_padded;
            }

            // resize rgb to depth size
            int orig_width{rgb.cols}, orig_height{rgb.rows};
            cv::resize(rgb, rgb, cv::Size(kWidth, kHeight), 0, 0, cv::INTER_AREA);
            auto depth = ReadDepth((path2scannetscene / "depth" / filename).replace_extension(".png").string());

            // crop black areas from calibration
            int h_crop = 6;
            int w_crop = 8;
            rgb = cv::Mat(rgb, cv::Rect(w_crop, h_crop, rgb.cols - 2 * w_crop, rgb.rows - 2 * h_crop));
            depth = cv::Mat(depth, cv::Rect(w_crop, h_crop, depth.cols - 2 * w_crop, depth.rows - 2 * h_crop));

            // compute maximum
            double curr_min_depth{0.}, curr_max_depth{0.};
            cv::minMaxLoc(depth, &curr_min_depth, &curr_max_depth);
            max_depth = std::max(static_cast<float>(curr_max_depth) / kDepthScalingFactor, max_depth);
            if (max_depth > config.kMaxDepth)
            {
                const auto max_depth_int(static_cast<unsigned short>(config.kMaxDepth * kDepthScalingFactor));
                depth.forEach<unsigned short>([max_depth_int](unsigned short& pixel, const int[]) -> void {
                    pixel = (pixel >= max_depth_int) ? 0 : pixel;
                    });
                std::cout << "Warning: " << filename << " maximal depth " << max_depth << " invalidate values >= " 
                    << config.kMaxDepth <<  std::endl;
                max_depth = config.kMaxDepth;
            }

            // write rgb
            std::string rgb_file_rel{(boost::filesystem::path(dataset_type) / "rgb" / filename).string()};
            WriteRgb(rgb, (path2scene / rgb_file_rel).string());
            if (config.kRgbOnly)
            {
                continue;
            }

            // write sparse depth
            std::string depth_file_rel{(boost::filesystem::path(dataset_type) / "depth" / filename).replace_extension(".png").string()};
            cv::Mat sparse_depth = cv::Mat::zeros(depth.rows, depth.cols, CV_16UC1);
            recon.GetSparseDepth(filename, dataset_type, sparse_depth, config.kMaxDepth, kDepthScalingFactor, depth);
            WriteDepth(sparse_depth, (path2scene / depth_file_rel).string());

            // write target depth
            std::string target_depth_file_rel{(boost::filesystem::path(dataset_type) / "target_depth" / filename).replace_extension(".png").string()};
            WriteDepth(depth, (path2scene / target_depth_file_rel).string());

            // set camera
            auto camera = recon.GetCameraConfig(filename).camera;
            camera_frames[dataset_type].emplace_back(CameraFrame{rgb_file_rel, depth_file_rel, camera});
        }
        std::cout << "Processed " << filenames.size() << " " << dataset_type << " views" << std::endl;
    }
    if (config.kRgbOnly)
    {
        return 0;
    }
    
    // write camera pose files
    const float far = max_depth * 1.025f;
    for (const std::string& dataset_type : {"train", "test"})
    {
        const std::string camera_frame_file((path2scene / ("transforms_" + dataset_type + ".json")).string());
        WriteCameraFrames(camera_frame_file, camera_frames[dataset_type], 0.1f, far, kDepthScalingFactor);
    }
    std::cout << "Set far plane to " << far << std::endl;
    std::cout << "Percent valid depth in train views: " << recon.GetPercentValid() * 100.f << std::endl;
    std::cout << "Scaling between sparse depth and target depth: " << recon.GetScaling() << std::endl;
    
    return 0;
}
