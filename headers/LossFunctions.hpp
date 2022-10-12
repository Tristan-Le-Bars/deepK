class LossFunctions {
public:
    std::vector<double> MeanSquaredError(std::vector<double> labels, std::vector<double> output_values);
    std::vector<double> BinaryCrossEntropy(std::vector<double> labels, std::vector<double> output_values);
    std::vector<double> Hinge(std::vector<double> labels, std::vector<double> output_values);
};