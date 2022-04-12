#include <iostream>

#include <file_utils.h>

std::vector<std::string> SplitByChar(const std::string &s, char c, bool allow_empty)
{
    std::vector<std::string> result{};
    std::size_t end = s.find(c);
    std::size_t start = 0;
    for (; end != std::string::npos;)
    {
        if (allow_empty || start != end)
        {
            result.push_back( s.substr(start, end - start));
        }

        start = end + 1;
        end = s.find(c, start);
    }

    // add the rest
    if (start != s.size())
    {
        result.push_back(s.substr(start, s.size() - start));
    }

    return result;
}

void StringToType(const std::string& s, float& instance)
{
    instance = std::stof(s);
}

void StringToType(const std::string& s, int& instance)
{
    instance = std::stoi(s);
}

void StringToType(const std::string& s, std::string& instance)
{
    instance = s;
}
