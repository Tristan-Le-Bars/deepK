#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>

#include <../headers/ActivationFunctions.hpp>

class SequentialModel{
    private:
        // Variables
        int layers_nbr;
        std::vector<std::vector<double>> neural_matrix;
        std::vector<std::vector<double>> synaptic_matrix;
        std::vector<double> bias;
        std::vector<std::string> activation_functions_matrix;

        void ForwardPropagation(double bias, int network_position){
            ActivationFunctions activation_function = ActivationFunctions();
            std::vector<double> weights = this->synaptic_matrix[network_position];
            std::vector<double> inputs = this->neural_matrix[network_position]; 
            std::string choosen_function = this->activation_functions_matrix[network_position];

            std::vector<double> output = std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0);

            for(int i; i < output.size(); i++){
                output[i] += bias;
                if(choosen_function == "Logistic")
                    output[i] = activation_function.Logistic(output[i]);
                if(choosen_function == "ReLU")
                    output[i] = activation_function.ReLU(output[i]);
                if(choosen_function == "Tanh")
                    output[i] = activation_function.Tanh(output[i]);
            }
            this->neural_matrix[network_position] = output;
        }

    public:
        // Constructor
        SequentialModel(/* args */);
        // Destructor
        ~SequentialModel();


        // Seters
        void SetLayersNbr(int l){layers_nbr = l;};
        void SetNeuralMatrix(std::vector<std::vector<double>> n){neural_matrix = n;};
        void SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};


        // Geters
        int GetLayerNbr(){return layers_nbr;}
        std::vector<std::vector<double>> GetNeuralMatrix(){return neural_matrix;}
        std::vector<std::string> GetActivationFunctionMatrix(){return activation_functions_matrix;}

        // Methodes
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
            
            activation_functions_matrix.push_back(activation_function);
        }
};

SequentialModel::SequentialModel(/* args */){}
SequentialModel::~SequentialModel(){}