#include <iostream>
#include <bits/stdc++.h>
#include <cmath>

class ActivationFunctions{
    private:
    public:
        // Constructors
        ActivationFunctions(/* args */);
        ~ActivationFunctions();
        // Methods
        double Logistic(double x){return (1/(1 + exp(-x)));}
        double ReLU(double x){return std::max(x, 0.0);}
        double Tanh(double x){return std::tanh(x);}
};

ActivationFunctions::ActivationFunctions(/* args */){}
ActivationFunctions::~ActivationFunctions(){}

