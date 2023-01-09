#ifndef CLASS_LOSS_FUNCTIONS
#define CLASS_LOSS_FUNCTIONS

class LossFunctions {
    public:
        // Constructor
        LossFunctions();

        // Destructor
        ~LossFunctions();

        // Methods
        void MeanSquaredError(double label, std::vector<double> output_values, std::vector<double> *loss);
        std::vector<double> BinaryCrossEntropy(std::vector<double> labels, std::vector<double> output_values);
        std::vector<double> Hinge(std::vector<double> labels, std::vector<double> output_values);
};

#endif CLASS_LOSS_FUNCTIONS