#include <iostream>
#include <stdio.h>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

class DataLoader{
    private:
        /* data */

        // Methodes
        std::string GetOS(){
            #if defined(_WIN32) || defined(_WIN64)
                return("windows");
            #elif __linux__
                return("linux");
            #elif __unix__
                return("unix");
            #elif TARGET_OS_MAC
                return("macos");
            #else
                return("not found");
            #endif

            return("not found");
        }

    public:
        // Constructor
        DataLoader(/* args */);
        //Destructor
        ~DataLoader();

        // Methods
        std::vector<std::vector<std::string>> LoadCSV(std::filesystem::path path){
            std::vector<std::vector<std::string>> data_matrix;
            std::fstream fin;
  
            // Open an existing file
            fin.open(path);
        
            // Get the roll number
            // of which the data is required
            int rollnum, roll2, count = 0;
            std::cout << "Enter the roll number "
                << "of the student to display details: ";
            std::cin >> rollnum;
        
            // Read the Data from the file
            // as String Vector
            std::vector<std::string> row;
            std::string line, word, temp;

            while (fin >> temp) {
  
                row.clear();
        
                // read an entire row and
                // store it in a string variable 'line'
                getline(fin, line);
        
                std::stringstream s(line);
        
                // read every column data of a row and
                // store it in a string variable, 'word'
                while (getline(s, word, ',')) {
        
                    // add all the column data
                    // of a row to a vector
                    row.push_back(word);
                }
                data_matrix.push_back(row);
            }
            return data_matrix;
        }
        
};

DataLoader::DataLoader(/* args */){}

DataLoader::~DataLoader(){}
