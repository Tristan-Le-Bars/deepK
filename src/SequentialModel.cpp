#include <iostream>

class SequentialModel
{
private:
    int layers_nbr;
    std::vector<std::vector<int>> neural_matrix;
    std::vector<std::string> activation_functions_matrix;


public:
    // Constructors
    SequentialModel(/* args */);
    ~SequentialModel();

    // Methods
    AddLayer(int neurons_nbr, std::string activation_function);
};

SequentialModel::SequentialModel(/* args */){}
SequentialModel::~SequentialModel(){}


// seters
void SetLayersNbr(int l){layer_nbr = l;};
void SetNeuralMatrix(std::vector<std::vector<int>> n){neural_matrix = n;};
void SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};


// geters
int GetLayerNbr(){return layer_nbr};
int GetNeuralMatrix(){return neural_matrix};
int GetActivationFunctionMatrix(){return activationFunctionMatrix}

// methodes
SequentialModel::AddLayer(int neurons_nbr, std::string activation_function) {
    neural_matrix.
}
