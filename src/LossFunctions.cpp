#include <iostream>
#include <vector>
#include <bits/stdc++.h>

#include "../headers/LossFunctions.hpp"

// Constructor
LossFunctions::LossFunctions(/* args */){}

// Destructor
LossFunctions::~LossFunctions(){}

// Methods
double LossFunctions::MeanSquaredError(double label, std::vector<double> output_values){

    double error = 0;
    double data_size = output_values.size();

    for (int i = 0; i < data_size; i++) {
        error += std::pow(output_values[i] - label, 2);
    }
    error /= data_size;
    return error;
}

std::vector<double> LossFunctions::BinaryCrossEntropy(std::vector<double> labels, std::vector<double> output_values){
    /* Make sure the labels are 0 and 1, else, change them to 0 and 1*/
    std::vector<double> loss;

    for(int i = 0; i < labels.size(); i++){
        if(labels[i] == 1)
            loss.push_back(-std::log10(output_values[i]));
        if(labels[i] == 0)
            loss.push_back(-std::log10(1-output_values[i]));
    }

    return loss;
}

std::vector<double> LossFunctions::Hinge(std::vector<double> labels, std::vector<double> output_values){
    std::vector<double> loss;
    std::vector<double> buffer;

    for(int i = 0; i < labels.size(); i++){
        if(labels[i] == 0)
            labels[i] = -1;
    }

    for(int i = 0; i < labels.size(); i++){
        buffer = {0, 1 - labels[i] * output_values[i]};
        loss.push_back(*std::max_element(buffer.begin(), buffer.end()));
    }

    return loss;
}
