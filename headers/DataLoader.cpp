#ifndef CLASS_DATA_LOADER
#define CLASS_DATA_LOADER

#include <string>
#include <vector>
#include <filesystem>

class SequentialModel{
    private:
        std::string GetOS(){};

    public:
        std::vector<std::vector<std::string>>  LoadCSV(std::filesystem::path path);

};

#endif CLASS_DATA_LOADER