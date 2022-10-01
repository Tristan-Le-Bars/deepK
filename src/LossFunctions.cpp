#include <iostream>
#include <vector>

class LossFunctions
{
private:
    /* data */
public:
    // Constructor
    LossFunctions(/* args */);

    // Destructor
    ~LossFunctions();

    // Public methods
    std::vector<double> MeanSquaredError(std::vector<double> label, std::vector<double> output_values){
        std::vector<double> loss;
        for(int i; i < label.size(); i++){
            loss.push_back((label[i] - output_values[i]) * (label[i] - output_values[i]));
        }
        return loss;
    }
};

LossFunctions::LossFunctions(/* args */){}

LossFunctions::~LossFunctions(){}
