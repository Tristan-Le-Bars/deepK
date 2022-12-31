#ifndef CLASS_ACTIVATION_FUNCTIONS
#define CLASS_ACTIVATION_FUNCTIONS

class ActivationFunctions {
    public:
        // Constructor
        ActivationFunctions();

        // Destructor
        ~ActivationFunctions();

        // Methods
        double Logistic(double x);
        double ReLU(double x);
        double Tanh(double x);
        double DerivatedLogistic(double x);
        double DerivatedReLU(double x);
        double DerivatedTanh(double x);
};

#endif CLASS_ACTIVATION_FUNCTIONS