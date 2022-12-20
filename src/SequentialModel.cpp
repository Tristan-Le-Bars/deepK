#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <numeric>
#include <cmath>

#include "../headers/ActivationFunctions.hpp"
#include "../headers/LossFunctions.hpp"
#include "../headers/SequentialModel.hpp"

// Constructor
SequentialModel::SequentialModel(/* args */){}

// Destructor
SequentialModel::~SequentialModel(){}

// Seters
void SequentialModel::SetLayersNbr(int l){layers_nbr = l;};
void SequentialModel::SetNeuralMatrix(std::vector<std::vector<std::vector<double>>> n){neural_matrix = n;};
void SequentialModel::SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};
void SequentialModel::SetLossFunction(std::string l){loss_function = l;}
void SequentialModel::SetLearningRate(double lr){learning_rate = lr;}

// Geters
int SequentialModel::GetLayerNbr(){return layers_nbr;}
std::vector<std::vector<double>> SequentialModel::GetNeuralMatrix(){return neural_matrix;}
std::vector<std::string> SequentialModel::GetActivationFunctionMatrix(){return activation_functions_matrix;}

// Methods
std::vector<std::vector<double>> SequentialModel::MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix){
    /*mn * nk = mk*/
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

std::vector<std::vector<double>> SequentialModel::MatrixTransposition(std::vector<std::vector<double>> matrix){
    int row_size = matrix.size();
    int col_size = matrix[0].size();
    std::vector<std::vector<double>> transposed;

    for(int i = 0; i < row_size; ++i){
        for(int j = 0; j < col_size; ++j)
        {
            transposed[j][i] = matrix[i][j];
        }
    }
}

// AJOUTER LA GESTION DES NEURONES DE BIAIS
void SequentialModel::ForwardPropagation(int network_position){
    ActivationFunctions function; 
    std::string activation_function = activation_functions_matrix[network_position];
    double sum = 0.0;
    std::vector<double> sum_matrix;
    // create buffer variables to set input and weights to the good number of dimensions to use*
    // the matrix multiplication methodes
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> inputs;
    weights.push_back(synaptic_matrix[network_position]);
    inputs.push_back(neural_matrix[network_position]);
    

    std::vector<std::vector<double>> output = MatrixMultiplication(inputs, weights);

    /* Use only if the ouput matrix have a dim1 size above 1*/
    // for(int i = 0; i < output[0].size; i++){
    //     for(int j = 0; i < output.size; j++){
    //         sum += output[i][j];
    //     }
    //         sum_matrix.push_back(sum);
    //         sum = 0.0;
    // }

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
void SequentialModel::BackwardPropagation(std::vector<double> labels){
    LossFunctions function;
    ActivationFunctions ActivationFunctions;
    std::vector<double> outputs = neural_matrix[neural_matrix.size() - 1];
    std::vector<double> loss;
    std::vector<double> labels_buffer = labels;
    std::string loss_function = loss_function;
    std::vector<double> output_layer_errors;
    std::vector<double> difference;
    std::vector<std::vector<double>> impact;
    std::vector<std::vector<double>> bias_modif;
    std::vector<std::vector<double>> difference_buffer;
    std::vector<std::vector<double>> synaptic_matrix_buffer;
    std::vector<std::vector<double>> neural_matrix_buffer;
    std::vector<std::vector<double>> labels_range;
    double total_error;
    std::vector<std::vector<double>> der_prev_input;
    std::vector<double> impact_rate;
    
    labels_range[0].push_back(1.0 / labels.size());

    if(loss_function == "mean_quared")
        loss = function.MeanSquaredError(labels, outputs);
    if(loss_function == "binary_cross_entropy")
        loss = function.BinaryCrossEntropy(labels, outputs);
    if(loss_function == "hinge"){
        for(int i = 0; i < labels_buffer.size(); i++){
            if(!labels_buffer[i] == 0.0)
                labels_buffer[i] = -1;
        }
        loss = function.Hinge(labels_buffer, outputs);
    }
    
    /* Update weights at the output */
    // get the error of each ouput neuron
    for(int j = 0; j < outputs.size(); j++){
        output_layer_errors.push_back(1 / 2 * pow(labels[j] - outputs[j], 2));
        difference.push_back(outputs[j] - labels[j]);
    }

    // sum the error of each output neuron to get the total error
    total_error = std::accumulate(output_layer_errors.begin(), output_layer_errors.end(), 0);

    // Using buffer to give the appropriate number of dimensions to the difference and neural matrix vectors 
    difference_buffer.push_back(difference);
    neural_matrix_buffer.push_back(neural_matrix[neural_matrix.size() - 2]);
    impact = MatrixMultiplication(labels_range, MatrixMultiplication(difference_buffer, MatrixTransposition(neural_matrix_buffer)));
    bias_modif = MatrixMultiplication(labels_range, difference_buffer);



    for(int i = neural_matrix.size(); i > 0; i--){
        if(activation_functions_matrix[i] == "Logistic")
            synaptic_matrix_buffer.push_back(synaptic_matrix[i]);
            der_prev_input = MatrixMultiplication(synaptic_matrix_buffer, difference_buffer);
            for(int j = 0; j < der_prev_input[0].size(); j++){
                der_prev_input[0][j] = der_prev_input[0][j] * ActivationFunctions.DerivatedLogistic(neural_matrix[i][j]);
            }
        /* Mettre les autre dérivée de fonction d'activation
            // matrix product of ouput x synaptic weight * derivated function of the neurones values
        if(activation_functions_matrix[i] == "ReLU")
            // matrix product of ouput x synaptic weight * derivated function of the neurones values
        if(activation_functions_matrix[i] == "Tanh")
            // matrix product of ouput x synaptic weight * derivated function of the neurones values 
        */
        
        // met à jour tout les neurones de la couche
        for(int j = 0; j < impact[0].size(); j++){
            impact_rate.push_back(learning_rate * impact[0][j]);
        }

        for(int j = 0; j < synaptic_matrix[i].size(); j++){
            synaptic_matrix[i][j] = synaptic_matrix[i][j] - impact_rate[j];
        }
    }
    // neural_matrix[network_position] = output;
}


void SequentialModel::AddLayer(int neurons_nbr, std::string activation_function){
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

void SequentialModel::Train(std::vector<std::vector<double>> training_set, std::vector<double> labels_set, int epochs){
    // INITIALIZE WEIGHTS WITH RANDOM VALUES

    for(int i = 0; i < epochs; i++){
        for(int j = 0; j < neural_matrix.size(); j++){
            ForwardPropagation(j);
        }
        BackwardPropagation(labels_set);
    }
}