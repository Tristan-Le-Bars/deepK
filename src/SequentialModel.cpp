#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>

#include <../headers/ActivationFunctions.hpp>
#include <../headers/LossFunctions.hpp>

class SequentialModel{
    private:
        // Variables
        int layers_nbr;
        std::vector<std::vector<double>> neural_matrix;
        std::vector<std::vector<double>> synaptic_matrix;
        std::vector<double> bias;
        std::vector<std::string> activation_functions_matrix;
        std::string loss_function;


        // Methods
        void ForwardPropagation(double bias, int network_position){
            ActivationFunctions function = ActivationFunctions();
            std::vector<double> weights = this->synaptic_matrix[network_position];
            std::vector<double> inputs = this->neural_matrix[network_position]; 
            std::string activation_function = this->activation_functions_matrix[network_position];

            std::vector<double> output = std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0);

            for(int i; i < output.size(); i++){
                output[i] += bias;
                if(activation_function == "Logistic")
                    output[i] = function.Logistic(output[i]);
                if(activation_function == "ReLU")
                    output[i] = function.ReLU(output[i]);
                if(activation_function == "Tanh")
                    output[i] = function.Tanh(output[i]);
            }
            this->neural_matrix[network_position] = output;
        }

        void BackwardPropagation(std::vector<double> labels){
            LossFunctions function = LossFunctions();
            std::vector<double> outputs = this->neural_matrix[neural_matrix>.size() - 1];
            std::vector<double> loss;
            std::string loss_function = this->loss_function;

            if(loss_function == "mean_quared")
                loss = function.MeanSquaredError(labels, outputs);
            if(loss_function == "binary_cross_entropy")
                loss = function.BinaryCrossEntropy(labels, outputs);
            if(loss_function == "hinge")
                loss = function.Hinge(labels, outputs);
            

            for(int i = this->neural_matrix.size(); i > 0; i--){
                if(this->activation_functions_matrix[i] == "Logistic")
                    // matrix product of ouput x synaptic weight * derivated function of the neurones values
                if(this->activation_functions_matrix[i] == "ReLU")
                    // matrix product of ouput x synaptic weight * derivated function of the neurones values
                if(this->activation_functions_matrix[i] == "Tanh")
                    // matrix product of ouput x synaptic weight * derivated function of the neurones values 
            }
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
        void SetLossFunction(LossFunctions loss_function)

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