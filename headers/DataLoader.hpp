#ifndef CLASS_DATA_LOADER
#define CLASS_DATA_LOADER

#include <string>
#include <vector>
#include <filesystem>

class DataLoader{
    public:
        // Constructor
        DataLoader(/* args */);

        // Destructor
        ~DataLoader();

        // Methods
        std::vector<std::vector<std::string>> LoadCSV(std::filesystem::path path);
        void SplitDataset(std::vector<std::vector<std::string>> dataset, double percentage,
                          std::vector<std::vector<std::string>> *set_one,
                          std::vector<std::vector<std::string>> *set_two);
};

#endif CLASS_DATA_LOADER