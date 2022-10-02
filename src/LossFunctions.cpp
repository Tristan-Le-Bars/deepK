#include <iostream>
#include <vector>
#include <bits/stdc++.h>

class LossFunctions
{
private:
    /* data */
public:
    // Constructor
    LossFunctions(/* args */);

    // Destructor
    ~LossFunctions();

    // Public methods
    std::vector<double> MeanSquaredError(std::vector<double> labels, std::vector<double> output_values){
        std::vector<double> loss;

        for(int i; i < labels.size(); i++){
            loss.push_back((labels[i] - output_values[i]) * (labels[i] - output_values[i]));
        }
        
        return loss;
    }

    std::vector<double> BinaryCrossEntropy(std::vector<double> labels, std::vector<double> output_values){
        /* Make sure the labels are 0 and 1, else, change them to 0 and 1*/
        std::vector<double> loss;

        for(int i; i < labels.size(); i++){
            if(labels[i] == 1)
                loss.push_back(-std::log10(output_values[i]));
            if(labels[i] == 0)
                loss.push_back(-std::log10(1-output_values[i]));
        }

        return loss;
    }

    std::vector<double> Hinge(std::vector<double> labels, std::vector<double> output_values){
        std::vector<double> loss;
        std::vector<double> buffer;

        for(int i; i < labels.size(); i++){
            if(labels[i] == 0)
                labels[i] = -1;
        }

        for(int i; i < labels.size(); i++){
            buffer = {0, 1 - labels[i] * output_values[i]};
            loss.push_back(*std::max_element(buffer.begin(), buffer.end()));
        }

        return loss;
    }
};

LossFunctions::LossFunctions(/* args */){}

LossFunctions::~LossFunctions(){}
