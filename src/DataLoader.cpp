#include <iostream>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

#include "../headers/DataLoader.hpp"

// Constructor
DataLoader::DataLoader(/* args */){}

// Destructor
DataLoader::~DataLoader(){}

// Methods
std::vector<std::vector<std::string>> DataLoader::LoadCSV(std::filesystem::path path, bool shuffle){
    /*
    Load the content of a CSV file into a 2D vector of strings.

    Parameters:
        - path: filepath of the CSV file.
        - shuffle: boolean indicating if the data of the CSV must be shuffled during the loading.
    
    Returns:
        - data_matrix: 2D vector of strings containing the CSV's data.
    */

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

    if(shuffle == true)
        std::random_shuffle(data_matrix.begin() + 1, data_matrix.end());

    return data_matrix;
}

void DataLoader::SplitDataset(std::vector<std::vector<std::string>> dataset, double percentage,
                              std::vector<std::vector<std::string>> *set_one,
                              std::vector<std::vector<std::string>> *set_two){

    /*
    Split a dataset into 2.

    Parameters:
        - dataset: dataset to split.
        - percentage: percentage of the original dataset that will be set in the first dataset, the rest will be put in the second dataset.
        - *set_one: adress of the first dataset.
        - *set_two: adress of the second dataset.
    */
    std::vector<std::vector<std::string>> set_one_buffer;
    std::vector<std::vector<std::string>> set_two_buffer;

    int split_point = (int) dataset.size() * percentage;

    for(int i = 0; i < dataset.size(); i++){
        if(i < split_point)
            set_one_buffer.push_back(dataset[i]);
        else
            set_two_buffer.push_back(dataset[i]);
    }
    *set_one = set_one_buffer;
    *set_two = set_two_buffer;
}

void DataLoader::RemoveColumn(std::vector<std::vector<std::string>> &dataset, int index){
    for(auto &v : dataset){
        v.erase(v.begin() + index);
    }
}

void DataLoader::RemoveColumn(std::vector<std::vector<double>> &dataset, int index){
    for(auto &v : dataset){
        v.erase(v.begin() + index);
    }
}

