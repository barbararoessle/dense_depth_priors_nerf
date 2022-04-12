#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>

#include <glm/glm.hpp>

std::vector<std::string> SplitByChar(const std::string &s, char c, bool allow_empty = false);

void StringToType(const std::string& s, float& instance);
void StringToType(const std::string& s, int& instance);
void StringToType(const std::string& s, std::string& instance);

template<typename T>
std::vector<T> ReadSequence(const std::string& path, char delimiter = ' ')
{
    std::vector<T> result{};
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line))
    {
        const auto& elements = SplitByChar(line, delimiter);
        for (const auto e : elements)
        {
            T element{};
            StringToType(e, element);
            result.emplace_back(element);
        }
    }
    return result;
}

#endif // FILE_UTILS_H
