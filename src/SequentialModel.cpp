#include <iostream>
#include <vector>
#include <string>
#include <random>

class SequentialModel
{
private:
    int layers_nbr;
    std::vector<std::vector<double>> neural_matrix;
    std::vector<std::vector<double>> synaptic_matrix;
    std::vector<std::string> activation_functions_matrix;


public:
    // Constructors
    SequentialModel(/* args */);
    ~SequentialModel();

    // Methods
    // AddLayer(int neurons_nbr, std::string activation_function);

    SequentialModel::SequentialModel(/* args */){}
    SequentialModel::~SequentialModel(){}


    // seters
    void SetLayersNbr(int l){layers_nbr = l;};
    void SetNeuralMatrix(std::vector<std::vector<double>> n){neural_matrix = n;};
    void SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};


    // geters
    int GetLayerNbr(){return layers_nbr;}
    std::vector<std::vector<double>> GetNeuralMatrix(){return neural_matrix;}
    std::vector<std::string> GetActivationFunctionMatrix(){return activation_functions_matrix;}

    // methodes
    void AddLayer(int neurons_nbr, std::string activation_function) {
        std::vector<double> neurons(neurons_nbr, 0);
        std::vector<double> synapses;

        if(neural_matrix.size() >= 1){
            for(int i; i < neural_matrix.back().size() * neurons_nbr; i++){
                srand(time(0)); 
                double r = rand() % 100;  
                synapses.push_back(r);
            }
        }
        neural_matrix.push_back(neurons);
    }

};