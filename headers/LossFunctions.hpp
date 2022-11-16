#ifndef CLASS_LOSS_FUNCTIONS
#define CLASS_LOSS_FUNCTIONS

class LossFunctions {
    public:
        std::vector<double> MeanSquaredError(std::vector<double> labels, std::vector<double> output_values);
        std::vector<double> BinaryCrossEntropy(std::vector<double> labels, std::vector<double> output_values);
        std::vector<double> Hinge(std::vector<double> labels, std::vector<double> output_values);
};

#endif CLASS_LOSS_FUNCTIONS