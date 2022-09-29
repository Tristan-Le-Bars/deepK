#include <iostream>

class ActivationFunctions{
    private:
        std::string function;
    public:
        // Constructors
        ActivationFunctions(/* args */);
        ~ActivationFunctions();

        // Seters
        void SetFunction(std::string f){function = f;}

        // Geters
        std::string GetFunction(){return function;}

        // Methods
        double Logistic(){}
        double ReLU(){}
        double Tanh(){}
};

ActivationFunctions::ActivationFunctions(/* args */){}
ActivationFunctions::~ActivationFunctions(){}

