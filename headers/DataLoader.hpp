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
        std::vector<std::vector<std::string>> LoadCSV(std::filesystem::path path, bool shuffle = true);
        void SplitDataset(std::vector<std::vector<std::string>> dataset, double percentage,
                          std::vector<std::vector<std::string>> *set_one,
                          std::vector<std::vector<std::string>> *set_two);
        void RemoveColumn(std::vector<std::vector<std::string>> &dataset, int index);
        void RemoveColumn(std::vector<std::vector<double>> &dataset, int index);
};

#endif CLASS_DATA_LOADER