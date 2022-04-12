#include <fstream>
#include <iostream>

#include <rapidjson/document.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>

#include <camera_frames.h>

template<typename T>
void AddFloat(const std::string& name, float value, rapidjson::Document& json, T& object)
{
    rapidjson::Value key(name.c_str(), json.GetAllocator());
    rapidjson::Value val;
    val.SetFloat(value);
    object.AddMember(key, val, json.GetAllocator());
}

template<typename T>
void AddNonEmptyString(const std::string& name, const std::string& value, rapidjson::Document& json, T& object)
{
    if (!name.empty())
    {
        rapidjson::Value key(name.c_str(), json.GetAllocator());
        rapidjson::Value val(value.c_str(), json.GetAllocator());
        object.AddMember(key, val, json.GetAllocator());
    }
}

template<typename T>
void AddMatrix4(const std::string& name, const glm::mat4& matrix, rapidjson::Document& json, T& object)
{
    rapidjson::Value key(name.c_str(), json.GetAllocator());
    rapidjson::Value mat(rapidjson::Type::kArrayType);
    for (int r(0); r != 4; ++r)
    {
        rapidjson::Value mat_row(rapidjson::Type::kArrayType);
        const auto row = glm::row(matrix, r);
        for (int c(0); c != 4; ++c)
        {
            rapidjson::Value element;
            element.SetFloat(row[c]);
            mat_row.PushBack(element, json.GetAllocator());
        }
        mat.PushBack(mat_row, json.GetAllocator());
    }

    object.AddMember(key, mat, json.GetAllocator());
}

void WriteCameraFrames(const std::string& file, const std::vector<CameraFrame>& camera_frames, float camera_near, 
    float camera_far, float depth_scaling_factor)
{
    rapidjson::Document json;
    json.SetObject();

    AddFloat("near", camera_near, json, json);
    AddFloat("far", camera_far, json, json);
    AddFloat("depth_scaling_factor", depth_scaling_factor, json, json);

    // frames
    rapidjson::Value frames(rapidjson::Type::kArrayType);
    for (const auto& camera_frame : camera_frames)
    {
        rapidjson::Value frame(rapidjson::Type::kObjectType);

        // file paths
        AddNonEmptyString("file_path", camera_frame.rgb_file_path, json, frame);
        AddNonEmptyString("depth_file_path", camera_frame.depth_file_path, json, frame);

        // intrinsics
        AddFloat("fx", camera_frame.camera.GetFx(), json, frame);
        AddFloat("fy", camera_frame.camera.GetFy(), json, frame);
        AddFloat("cx", camera_frame.camera.GetCx(), json, frame);
        AddFloat("cy", camera_frame.camera.GetCy(), json, frame);

        // transform matrix
        AddMatrix4("transform_matrix", glm::inverse(camera_frame.camera.GetWorld2Cam()), json, frame);

        frames.PushBack(frame, json.GetAllocator());
    }
    json.AddMember("frames", frames, json.GetAllocator());

    std::ofstream ofs(file);
    rapidjson::OStreamWrapper osw(ofs);

    rapidjson::PrettyWriter<rapidjson::OStreamWrapper> writer(osw);
    json.Accept(writer);
}
