#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>

#include "../headers/ActivationFunctions.hpp"
#include "../headers/LossFunctions.hpp"
#include "../headers/SequentialModel.hpp"

// Constructor
SequentialModel::SequentialModel(/* args */){}

// Destructor
SequentialModel::~SequentialModel(){}

// Methods
std::vector<std::vector<double>> SequentialModel::MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix){
    /*mn * nk*/
    int first_matrix_rows_num = first_matrix.size();
    int first_matrix_columns_num = first_matrix[0].size();
    int second_matrix_rows_num = second_matrix.size();
    int second_matrix_columns_num = second_matrix[0].size();
    std::vector<std::vector<double>> result_matrix(first_matrix_rows_num);


    for(int i = 0; i < first_matrix_rows_num; ++i)
        for(int j = 0; j < second_matrix_columns_num; ++j){
            result_matrix[i].push_back(0.00);
        }
    
    for(int i = 0; i < first_matrix_rows_num; i++)
        for(int j = 0; j < second_matrix_columns_num; j++)
            for(int k = 0; k < first_matrix_columns_num; k++){
                result_matrix[i][j] += first_matrix[i][k] * second_matrix[k][j];
            }

    return result_matrix;
}

void SequentialModel::ForwardPropagation(double bias, int network_position){
    ActivationFunctions activation_function = ActivationFunctions();
    std::vector<std::vector<double>> weights = synaptic_matrix[network_position];
    std::vector<std::vector<double>> inputs = neural_matrix[network_position];
    std::string choosen_function = activation_functions_matrix[network_position];
    double sum = 0.0;
    std::vector<double> sum_matrix;

    std::vector<std::vector<double>> output = MatrixMultiplication(inputs, weights);

    for(int i = 0; i < output[0].size; i++){
        for(int j = 0; i < output.size; j++){
            sum += output[i][j];
        }
            sum_matrix.push_back(sum);
            sum = 0.0;
    }

    for(int i = 0; i < sum_matrix.size(); i++){
        sum_matrix[i] += bias;
        if(activation_function == "Logistic")
            sum_matrix[i] = function.Logistic(sum_matrix[i]);
        if(activation_function == "ReLU")
            sum_matrix[i] = function.ReLU(sum_matrix[i]);
        if(activation_function == "Tanh")
            sum_matrix[i] = function.Tanh(sum_matrix[i]);
    }
    neural_matrix[network_position] = sum_matrix;
}

void SequentialModel::BackwardPropagation(std::vector<double> labels){
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
    neural_matrix[network_position] = output;
}


// Seters
void SequentialModel::SetLayersNbr(int l){layers_nbr = l;};
void SequentialModel::SetNeuralMatrix(std::vector<std::vector<std::vector<double>>> n){neural_matrix = n;};
void SequentialModel::SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};
void SequentialModel::SetLossFunction(std::string l){loss_function = l;}

// Geters
int SequentialModel::GetLayerNbr(){return layers_nbr;}
std::vector<std::vector<std::vector<double>>> SequentialModel::GetNeuralMatrix(){return neural_matrix;}
std::vector<std::string> SequentialModel::GetActivationFunctionMatrix(){return activation_functions_matrix;}

// Methodes
void SequentialModel::AddLayer(int neurons_nbr, std::string activation_function) {
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