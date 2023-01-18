#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cmath>
#include <algorithm>

#include "../headers/matplotlibcpp.h"
#include "../headers/ActivationFunctions.hpp"
#include "../headers/LossFunctions.hpp"
#include "../headers/SequentialModel.hpp"

namespace plt = matplotlibcpp;

// Constructor
SequentialModel::SequentialModel(int input_s){
    std::vector<double> input_layer(input_s, 0.0);

    neural_matrix.push_back(input_layer);

    input_size = input_s;
}

// Destructor
SequentialModel::~SequentialModel(){}

// Seters
void SequentialModel::SetLayersNbr(int l){layers_nbr = l;};
void SequentialModel::SetNeuralMatrix(std::vector<std::vector<double>> n){neural_matrix = n;};
void SequentialModel::SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};
void SequentialModel::SetLossFunction(std::string l){loss_function = l;}
void SequentialModel::SetLearningRate(double lr){learning_rate = lr;}

// Geters
int SequentialModel::GetLayerNbr(){return layers_nbr;}
std::vector<std::vector<double>> SequentialModel::GetNeuralMatrix(){return neural_matrix;}
std::vector<std::string> SequentialModel::GetActivationFunctionMatrix(){return activation_functions_matrix;}
int SequentialModel::GetInputSize(){return input_size;}

// Methods
std::vector<std::vector<double>> SequentialModel::MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix){
    /*mn * nk = mk*/
    int first_matrix_rows_num = first_matrix.size();
    int first_matrix_columns_num = first_matrix[0].size();
    int second_matrix_rows_num = second_matrix.size();
    int second_matrix_columns_num = second_matrix[0].size();

    std::vector<std::vector<double>> result_matrix(first_matrix_rows_num, std::vector<double>(second_matrix_columns_num, 0.0));

    for(int i = 0; i < first_matrix_rows_num; i++){
        for(int j = 0; j < second_matrix_columns_num; j++){
            for(int k = 0; k < first_matrix_columns_num; k++){
                result_matrix[i][j] += first_matrix[i][k] * second_matrix[k][j];
            }
        }
    }

    return result_matrix;
}

std::vector<std::vector<double>> SequentialModel::MatrixTransposition(std::vector<std::vector<double>> matrix){
    int row_size = matrix.size();
    int col_size = matrix[0].size();
    std::vector<std::vector<double>> transposed(col_size, std::vector<double>(row_size));

    for(int i = 0; i < row_size; i++){
        for(int j = 0; j < col_size; j++)
        {
            transposed[j][i] = matrix[i][j];
        }
    }
    return transposed;
}

double SequentialModel::NormalDistribution(){
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0,1.0);

    return distribution(generator);
}

void SequentialModel::ForwardPropagation(int network_position,
                                         std::vector<std::vector<double>> test_set,
                                         std::vector<double> labels_test_set){
    ActivationFunctions function; 
    std::string activation_function = activation_functions_matrix[network_position];
    double sum = 0.0;
    std::vector<double> sum_matrix;
    // create buffer variables to set input and weights to the good number of dimensions to use*
    // the matrix multiplication methodes
    std::vector<std::vector<double>> weights = synaptic_matrix[network_position]; // Shape = number of neurones * number of connection per neuron
    std::vector<std::vector<double>> inputs;
    // weights.push_back(synaptic_matrix[network_position]);
    inputs.push_back(neural_matrix[network_position]);


    std::vector<std::vector<double>> output = MatrixMultiplication(inputs, weights);

    sum_matrix = output[0];

    /* Apply the activation function*/
    for(int i = 0; i < sum_matrix.size(); i++){
        sum_matrix[i] += bias[network_position];
        if(activation_function == "Logistic")
            sum_matrix[i] = function.Logistic(sum_matrix[i]);
        if(activation_function == "ReLU")
            sum_matrix[i] = function.ReLU(sum_matrix[i]);
        if(activation_function == "Tanh")
            sum_matrix[i] = function.Tanh(sum_matrix[i]);
    }
    neural_matrix[network_position + 1] = sum_matrix;
}

// AJOUTER LA GESTION DES NEURONES DE BIAIS
void SequentialModel::BackwardPropagation(double label,
                                          std::vector<std::vector<double>> test_set,
                                          std::vector<double> labels_test_set){
    LossFunctions function;
    ActivationFunctions ActivationFunctions;
    std::vector<double> outputs = neural_matrix.back();
    double loss;
    double accuracy;
    double output_error;

    std::cout << "output = " << outputs[0] << "; label = " << label << "; ";
    accuracy = TestAccuracy(test_set, labels_test_set);
    std::cout << "accuracy = " << accuracy << std::endl;
    accurarcies_history.push_back(accuracy);

    if(loss_function == "mean_squared")
        loss = function.MeanSquaredError(label, outputs); //Etotal

    if(loss_function == "binary_cross_entropy")
        loss = function.BinaryCrossEntropy(label, outputs);

    if(loss_function == "hinge")
        loss = function.Hinge(label, outputs);

    losses_history.push_back(loss);

    output_error = outputs[0] - label;


    
    std::vector<std::vector<double>> hidden_errors(synaptic_matrix.size() - 1);

    for(int i = synaptic_matrix.size() - 1; i > 0; i--){
        hidden_errors[i-1].resize(synaptic_matrix[i].size());
        for (int j = 0; j < synaptic_matrix[i].size(); j++) {
            double error = 0;
            for (int k = 0; k < synaptic_matrix[i][j].size(); k++) {
                error += synaptic_matrix[i][k][j] * output_error;
            }
            hidden_errors[i-1][j] = error;
        }
        // if(activation_functions_matrix[i] == "Logistic"){
        //     std::cout << "not implemented" << std::endl;
        // }


        // if(activation_functions_matrix[i] == "ReLU"){
        //     std::cout << "not implemented" << std::endl;
        // }

        // if(activation_functions_matrix[i] == "Tanh"){
        //     std::cout << "not implemented" << std::endl;
        // }
    }

    for (int i = synaptic_matrix.size() - 1; i >= 0; i--) {
        for (int j = 0; j < synaptic_matrix[i].size(); j++) {
            double delta = 0;
            if (i == synaptic_matrix.size() - 1) {
                delta = output_error * ActivationFunctions.DerivatedLogistic(outputs[0]);
            } else {
                delta = hidden_errors[i][j] * ActivationFunctions.DerivatedLogistic(outputs[0]);
            }
            bias[i] -= learning_rate * delta;
            for (int k = 0; k < synaptic_matrix[i][j].size(); k++) {
                synaptic_matrix[i][j][k] -= learning_rate * delta * neural_matrix[0][k];
            }
        }
    }
}

double SequentialModel::TestAccuracy(std::vector<std::vector<double>> test_set,
                                     std::vector<double> test_labels_set){
    int correct_prediction = 0;
    double accuracy;
    // std::cout << std::endl;
    for(int i = 0; i < test_set.size(); i++){
        // std::cout << "neural_matrix[0] = " << std::endl;

        neural_matrix[0] = test_set[i];
        // for(int j = 0; j < neural_matrix[0].size(); j++){
        //     std::cout << neural_matrix[0][j] << " "; 
        // }
        // std::cout << std::endl;

        for(int j = 0; j < neural_matrix.size() - 1; j++){
            ForwardPropagation(j, test_set, test_labels_set);
        }
        // std::cout << "output = " << neural_matrix.back()[0] << std::endl;
        // std::cout << "label = " << test_labels_set[i] << std::endl;
        if((neural_matrix.back()[0] < 0.45 && test_labels_set[i] == 0.0) ||
           (neural_matrix.back()[0] > 0.55 && test_labels_set[i] == 1.0)){
            correct_prediction++;
            // std::cout << "yes" <<std::endl;
           }
    // std::cout << std::endl;
    }
    // std::cout << std::endl;
    // for(int i = 0;i < neural_matrix.back().size(); i++){
    //     std::cout << neural_matrix.back()[i] << " ";
    // } 
    // std::cout << std::endl;
    // std::cout << "test_labels_set size = " << test_set.size() << std::endl;
    // std::cout << "correct prediction = " << correct_prediction << std::endl;
    accuracy = (correct_prediction * 100.0) / test_set.size();

    return accuracy;
}


void SequentialModel::AddLayer(int neurons_nbr, std::string activation_function){
    std::vector<double> neurons(neurons_nbr, 0);
    std::vector<std::vector<double>> synapses; // Shape = (neurons_nbr, neurons_nbr)

    if(neural_matrix.size() > 1){
        for(int i = 0; i < neural_matrix.back().size(); i++){
            synapses.push_back(std::vector<double>());
            for(int j = 0; j < neurons_nbr; j++){
                synapses[i].push_back(0.0);
            }
        }
    }
    else{
        for(int i = 0; i < input_size; i++){
            synapses.push_back(std::vector<double>());
            for(int j = 0; j < neurons_nbr; j++){
                synapses[i].push_back(0.0);
            }
        }
    }
    
    neural_matrix.push_back(neurons);
    synaptic_matrix.push_back(synapses);
    bias.push_back(0.0);
    
    activation_functions_matrix.push_back(activation_function);
}

void SequentialModel::Compile(){
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for(int i = 0; i < synaptic_matrix.size(); i++){
        for(int j = 0; j < synaptic_matrix[i].size(); j++){
            for(int k = 0; k < synaptic_matrix[i][j].size(); k++){
                synaptic_matrix[i][j][k] = distribution(generator);
            }
        }
    }

    for(int i = 0; i < neural_matrix.size() - 2; i++){
        bias.push_back(NormalDistribution());
    }
}

void SequentialModel::Train(std::vector<std::vector<double>> training_set, std::vector<double> train_labels_set,
                            std::vector<std::vector<double>> test_set, std::vector<double> test_labels_set,
                            int epochs){
    std::cout << "neural map:" << std::endl;
    for(int i = 0;i<neural_matrix.size(); i++){
        for(int j = 0;j<neural_matrix[i].size(); j++){
            std::cout << neural_matrix[i][j];
        }
        std::cout << std::endl;
    }

    std::cout << "synaptic map:" << std::endl;
    for(int i = 0;i<synaptic_matrix.size(); i++){
        for(int j = 0;j<synaptic_matrix[i].size(); j++){
            for(int k = 0;k<synaptic_matrix[i][j].size();k++){
                std::cout << synaptic_matrix[i][j][k] << " ";
            }
            std::cout << "     ";
        }
        std::cout << std::endl;
    }


    for(int i = 0; i < epochs; i++){
        neural_matrix[0] = training_set[i];

        /*Display values in the neural network*/
        std::cout << "bias map:" << std::endl;
        for(int j = 0;j < bias.size(); j++){
            std::cout << bias[j] << " ";    
        }
        std::cout << std::endl;

        std::cout << "neural map:" << std::endl;
        for(int j = 0; j < neural_matrix.size(); j++){
            for(int k = 0; k < neural_matrix[j].size(); k++){
                std::cout << neural_matrix[j][k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "synaptic map:" << std::endl;
        for(int j = 0; j < synaptic_matrix.size(); j++){
            for(int k = 0; k < synaptic_matrix[j].size(); k++){
                for(int l = 0; l < synaptic_matrix[j][k].size(); l++){
                    std::cout << synaptic_matrix[j][k][l] << " ";
                }
                std::cout << "     ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;



        std::cout << "Epoch: " << i << " ---> ";
        for(int j = 0; j < neural_matrix.size() - 1; j++){
            ForwardPropagation(j, test_set, test_labels_set);
        }
        BackwardPropagation(train_labels_set[i], test_set, test_labels_set);


    }
}

void SequentialModel::DisplayAccuraciesHistory(){
    plt::figure();
    plt::plot(accurarcies_history);
    plt::xlabel("Epoch");
    plt::ylabel("accuracy");
    plt::title("history of accuracies obtained during the training");
    plt::savefig("accurarcies_history.pdf");
}
void SequentialModel::DisplayLossesHistory(){
    plt::figure();
    plt::plot(losses_history);
    plt::xlabel("Epoch");
    plt::ylabel("loss");
    plt::title("history of losses obtained during the training");
    plt::savefig("losses_history.pdf");
}
