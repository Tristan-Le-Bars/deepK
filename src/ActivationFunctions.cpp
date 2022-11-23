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
