#include <iostream>
#include <bits/stdc++.h>
#include <cmath>
#include "../headers/ActivationFunctions.hpp"



// Constructor
ActivationFunctions::ActivationFunctions(/* args */){}

// Destructor
ActivationFunctions::~ActivationFunctions(){}

// Methods
double ActivationFunctions::Logistic(double x){return (1/(1 + exp(-x)));}
double ActivationFunctions::ReLU(double x){return std::max(x, 0.0);}
double ActivationFunctions::Tanh(double x){return std::tanh(x);}

double ActivationFunctions::DerivatedLogistic(double x){
    return (1 / (1 + exp(-x))) * (1.0 - (1 / (1 + exp(-x))));
}

double ActivationFunctions::DerivatedReLU(double x){
    if(x < 0.0)
        return 0.0;
    else
        return 1;
}

double ActivationFunctions::DerivatedTanh(double x){
    double tanh_fraction = (exp(x) - exp(-x) / exp(x) + exp(-x));

    return 1.0 - tanh_fraction * tanh_fraction;
}
