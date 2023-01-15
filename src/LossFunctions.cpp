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
    double loss = 0;
    double data_size = output_values.size();

    for (int i = 0; i < data_size; i++) {
        loss += std::pow(output_values[i] - label, 2);
    }
    loss /= data_size;
    return loss;
}

double LossFunctions::BinaryCrossEntropy(double label, std::vector<double> output_values){
    /* Make sure the labels are 0 and 1, else, change them to 0 and 1*/
    double loss = 0;
    double data_size = output_values.size();

    for (int i = 0; i < data_size; i++) {
        loss += label * log(output_values[i]) + (1 - label) * log(1 - output_values[i]);
    }
    return -loss / data_size;
}

double LossFunctions::Hinge(double label, std::vector<double> output_values){
    double loss = 0;
    double data_size = output_values.size();

    for (int i = 0; i < data_size; i++) {
        loss += std::max(0.0, 1 - label * output_values[i]);
    }
    return loss / data_size;
}
