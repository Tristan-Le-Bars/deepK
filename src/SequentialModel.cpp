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

    // std::cout << "inputs_matrix =" << std::endl; 
    // for (int i = 0; i < first_matrix_rows_num; i++) {
    //     for (int j = 0; j < first_matrix_columns_num; j++) {
    //         std::cout << first_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;
    // std::cout << "weights_matrix =" << std::endl; 
    // for (int i = 0; i < second_matrix_rows_num; i++) {
    //     for (int j = 0; j < second_matrix_columns_num; j++) {
    //         std::cout << second_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << std::endl;


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

double SequentialModel::GaussianRand(){
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0,1.0);

    return distribution(generator);
}

void SequentialModel::ForwardPropagation(int network_position){
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
    
    for(int i = 0;i<synaptic_matrix.size(); i++){
        for(int j = 0;j<synaptic_matrix[i].size(); j++){
            for(int k = 0;k<synaptic_matrix[i][j].size();k++){
            }
        }
    }


    std::vector<std::vector<double>> output = MatrixMultiplication(inputs, weights);

    sum_matrix = output[0];

    // std::cout << "sum_matrix =" << std::endl;
    // for (int i = 0; i < sum_matrix.size(); i++) {
    //     std::cout << sum_matrix[i];
    // }

    // std::cout << std::endl;

    /* Apply the activation function*/
    for(int i = 0; i < sum_matrix.size(); i++){
        // sum_matrix[i] += bias[network_position]; // ICI AJOUTER LA GESTION DES BIAIS
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
    // std::string loss_function = loss_function;
    std::vector<double> output_layer_errors;
    std::vector<double> difference;
    std::vector<std::vector<double>> impact;
    std::vector<std::vector<double>> bias_modif;
    std::vector<std::vector<double>> difference_buffer;
    // std::vector<std::vector<double>> synaptic_matrix_buffer;
    std::vector<std::vector<double>> neural_matrix_buffer;
    std::vector<std::vector<double>> labels_range;
    double total_error;
    std::vector<std::vector<double>> der_prev_input;
    std::vector<double> impact_rate;
    
    labels_range.push_back(std::vector<double>());
    labels_range[0].push_back(1.0 / labels.size());
    if(loss_function == "mean_squared")
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
    /* get the error of each ouput neuron */
    for(int j = 0; j < outputs.size(); j++){
        output_layer_errors.push_back(1 / 2 * pow(labels[j] - outputs[j], 2));
        difference.push_back(outputs[j] - labels[j]);
    }


    // sum the error of each output neuron to get the total error
    total_error = std::accumulate(output_layer_errors.begin(), output_layer_errors.end(), 0);

    // Using buffer to give the appropriate number of dimensions to the difference and neural matrix vectors 
    difference_buffer.push_back(difference);
    neural_matrix_buffer.push_back(neural_matrix[neural_matrix.size() - 1]);
    impact = MatrixMultiplication(labels_range, MatrixMultiplication(difference_buffer, MatrixTransposition(neural_matrix_buffer)));

    for(int i = neural_matrix.size() - 2; i >= 0; i--){
        if(activation_functions_matrix[i] == "Logistic")
            der_prev_input = MatrixMultiplication(difference_buffer, synaptic_matrix[i]);
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
            for(int k = 0; k < synaptic_matrix[i][j].size(); k++)
            synaptic_matrix[i][j][k] = synaptic_matrix[i][j][k] - impact_rate[j];
        }

        /*GESTION DES BIAS À FAIRE PLUS TARD*/
        // bias_modif = MatrixMultiplication(MatrixMultiplication(labels_range, difference_buffer), der_prev_input);
        // bias[i] = bias[i] - (learning_rate * bias_modif);
    }
    // neural_matrix[network_position] = output;
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

// void SequentialModel::Compile(){
//     // FILL THE SYNAPTIC MATRIX WITH RANDOM VALUES AND THE NEURAL MATRIX WITH 0
// }

void SequentialModel::Train(std::vector<std::vector<double>> training_set, std::vector<double> labels_set, int epochs){
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
                std::cout << synaptic_matrix[i][j][k];
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "neural_matrix size = " << neural_matrix.size() << std::endl;

    for(int i = 0; i < epochs; i++){
        std::cout << "Epoch: " << i << std::endl;
        for(int j = 0; j < neural_matrix.size() - 1; j++){
            ForwardPropagation(j);
        }
        BackwardPropagation(labels_set);
    }
}