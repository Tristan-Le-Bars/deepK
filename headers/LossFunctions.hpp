#ifndef CLASS_LOSS_FUNCTIONS
#define CLASS_LOSS_FUNCTIONS

class LossFunctions {
    public:
        // Constructor
        LossFunctions();

        // Destructor
        ~LossFunctions();

        // Methods
        double MeanSquaredError(double label, std::vector<double> output_values);
        double BinaryCrossEntropy(double label, std::vector<double> output_values);
        double Hinge(double label, std::vector<double> output_values);
};

#endif