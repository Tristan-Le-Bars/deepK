#include <iostream>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "../headers/DataLoader.hpp"

// Constructor
DataLoader::DataLoader(/* args */){}

// Destructor
DataLoader::~DataLoader(){}

// Methods
std::vector<std::vector<std::string>> DataLoader::LoadCSV(std::filesystem::path path){
    std::vector<std::vector<std::string>> data_matrix;
    std::fstream fin;

    fin.open(path);
    std::vector<std::string> row;
    std::string line, word;

    while (getline(fin, line)) {

        row.clear();
        std::stringstream s(line);

        while (getline(s, word, ',')) {
            row.push_back(word);
        }
        
        data_matrix.push_back(row);
    }
    return data_matrix;
}

