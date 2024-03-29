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
/*
void SequentialModel::SetLayersNbr(int l){layers_nbr = l;};
void SequentialModel::SetNeuralMatrix(std::vector<std::vector<double>> n){neural_matrix = n;};
void SequentialModel::SetActivationFuntionMatrix(std::vector<std::string> a){activation_functions_matrix = a;};
*/

void SequentialModel::SetLossFunction(std::string l){
    try{
        if(l == "mean_squared" || l == "binary_cross_entropy" || l == "hinge")
            loss_function = l;
        else
            throw(l);
    }
    catch(std::string str){
        std::cout << std::endl << "ERROR 1: Loss function must be mean_squared, binary_cross_entropy or hinge." << std::endl;
        std::cout << "Loss function is " << str << "." << std::endl;
        exit(1);
    }
}

void SequentialModel::SetLearningRate(double lr){learning_rate = lr;}


// Geters
int SequentialModel::GetLayerNbr(){return layers_nbr;}
std::vector<std::vector<double>> SequentialModel::GetNeuralMatrix(){return neural_matrix;}
std::vector<std::string> SequentialModel::GetActivationFunctionMatrix(){return activation_functions_matrix;}
int SequentialModel::GetInputSize(){return input_size;}


// Methods
std::vector<std::vector<double>> SequentialModel::MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix){
    /*
    Performe a matrix multiplication  (mn * nk = mk).

    Parameters:
        - first_matrix: a 2 dimension vector of double.
        - second_matrix: a 2 dimension vector of double.

    Returns:
        - result_matrix: result of the matrix multiplication of first_matrix and second_matrix.
    */

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
    /*
    Transpose a matrix.

    Parameters:
        - matrix: a 2 dimension vector of double.
    
    Returns:
        - transposed: the transposed version of the matrix.
    */
   
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
    /*
    Create a normal distribution random double generator 

    Returns:
        - distribution(generator): a normal distribution random double generator
    */

    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(0.0,1.0);

    return distribution(generator);
}

void SequentialModel::ForwardPropagation(int network_position){
    /*
    Perform forward propagation for one layer of the neural network.

    Parameters:
        - network_position: integer indicating at which layer of the neural network forward propagation must be performed.
    */

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
        try{
            if(activation_function == "Logistic")
                sum_matrix[i] = function.Logistic(sum_matrix[i]);
            else if(activation_function == "ReLU")
                sum_matrix[i] = function.ReLU(sum_matrix[i]);
            else if(activation_function == "Tanh")
                sum_matrix[i] = function.Tanh(sum_matrix[i]);
            else{
                throw(activation_function);
            }
        }
        catch(std::string str){
            std::cout << std::endl << "ERROR 2: Activation function must be Logistic, ReLU or Tanh." << std::endl;
            std::cout << "Activation function is " << str << "." << std::endl;
            exit(2);
        }
    }
    neural_matrix[network_position + 1] = sum_matrix;
}

void SequentialModel::BackwardPropagation(double label,
                                          std::vector<std::vector<double>> test_set,
                                          std::vector<double> labels_test_set){
    /*
    Perform the back propagation on the neural network

    Parameters:
        - label: ground truth of the input given in the previous forward propagation
        - test_set: test dataset used to evaluate the neural network with the current weights.
        - labels_test_set: labels of the test dataset used to evaluate the neural network with the current weights.

    */
    LossFunctions function;
    ActivationFunctions ActivationFunctions;
    std::vector<double> outputs = neural_matrix.back();
    double loss;
    double accuracy;
    double output_error;

    // changer le calcul de l'output
    std::cout << "output = " << outputs[0] << "; label = " << label << "; ";
    accuracy = TestAccuracy(test_set, labels_test_set);
    if(accuracy > accuracy_historical_bests.back()){
        accuracy_historical_bests.push_back(accuracy);
    }

    if(loss_function == "mean_squared")
        loss = function.MeanSquaredError(label, outputs);

    if(loss_function == "binary_cross_entropy")
        loss = function.BinaryCrossEntropy(label, outputs);

    if(loss_function == "hinge")
        loss = function.Hinge(label, outputs);

    losses_history.push_back(loss);
    
    output_error = loss;

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
    }

    for (int i = synaptic_matrix.size() - 1; i >= 0; i--) {
        for (int j = 0; j < synaptic_matrix[i].size(); j++) {
            double delta = 0;
            if (i == synaptic_matrix.size() - 1) {
                if(activation_functions_matrix[i] == "Logistic")
                    delta = output_error * ActivationFunctions.DerivatedLogistic(outputs[0]);
                if(activation_functions_matrix[i] == "ReLU")
                    delta = output_error * ActivationFunctions.DerivatedReLU(outputs[0]);
                if(activation_functions_matrix[i] == "Tanh")
                    delta = output_error * ActivationFunctions.DerivatedTanh(outputs[0]);

            } else {
                if(activation_functions_matrix[i] == "Logistic")
                    delta = hidden_errors[i][j] * ActivationFunctions.DerivatedLogistic(outputs[0]);
                if(activation_functions_matrix[i] == "ReLU")
                    delta = hidden_errors[i][j] * ActivationFunctions.DerivatedReLU(outputs[0]);
                if(activation_functions_matrix[i] == "Tanh")
                    delta = hidden_errors[i][j] * ActivationFunctions.DerivatedTanh(outputs[0]);
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
    /*
    Compute different metrics to evaluate the model.

    Parameters:
        - test_set: test dataset used to evaluate the model.
        - labels_test_set: labels of the test dataset used to evaluate the model.

    Returns:
        - accuracy: return the accuracy obtained by the model on the test set.
    */

    int correct_prediction = 0;
    double accuracy;
    double precision;
    double recall;
    double f1;
    double sensivity;
    double specificity;
    double tp = 0;
    double tn = 0;
    double fp = 0;
    double fn = 0;

    for(int i = 0; i < test_set.size(); i++){
        neural_matrix[0] = test_set[i];

        for(int j = 0; j < neural_matrix.size() - 1; j++){
            ForwardPropagation(j);
        }

        if(neural_matrix.back()[0] >= 0.50 && test_labels_set[i] == 1.0)
            tp++;
        if(neural_matrix.back()[0] < 0.50 && test_labels_set[i] == 0.0)
            tn++;
        if(neural_matrix.back()[0] >= 0.50 && test_labels_set[i] == 0.0)
            fp++;
        if(neural_matrix.back()[0] < 0.50 && test_labels_set[i] == 1.0)
            fn++;
    }
    
    accuracy = (tn + tp) / (tn + fp + fn + tp);
    precision = tp / (tp + fp);
    recall = tp / (tp+fn);
    specificity = tn / (tn + fp);

    f1 = 2 * ((precision * recall) / (precision + recall));
    std::cout << " precision = " << precision << ";";
    std::cout << " recall = " << recall << ";";
    // std::cout << " specificity = " << specificity << ";";
    std::cout << " f1 = " << f1 << ";";
    std::cout << " accuracy = " << accuracy << std::endl;

    precision_history.push_back(precision);
    recall_history.push_back(recall);
    f1_history.push_back(f1);
    accurarcy_history.push_back(accuracy);

    return accuracy;
}


void SequentialModel::AddLayer(int neurons_nbr, std::string activation_function){
    /*
    Add a layer in the neural network.

    Parameters:
        - neurrons_nbr: number of neurones in the new layer.
        - activation_function: activation function of the neurons in the new layer.
    */

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

void SequentialModel::UniDistribInit(){
    /*
    Initiate the weights of the neural network with an uniform distribution between -1 and 1.
    */

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

void SequentialModel::HeInit(){
    /*
    Initiate the weights of the neural network with an uniform distribution between 1 and the scale of the input.
    */
    double scale = sqrt(2.0 / input_size);
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<> distribution(1, scale);

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
                            int epochs, unsigned int early_stopping){
    /*
    Train the neural network.

    Parameters:
        - training_set: dataset containing the inputs used to train the neural network.
        - train_labels_set: dataset containing the ground truth of the inputs in the training dataset. 
        - test_set: dataset containing the inputs used to test the neural network.
        - test_labels_set: dataset containing the ground truth of the inputs in the testing dataset.
        - epochs: number of training iteration of the neural network.
        - early_stopping: maximum number of training iteration without performances improvement accepted before stopping the training.
    */
    int stopping_range = 0;

    if(epochs == 0){
        epochs = training_set.size();
    }

    accuracy_historical_bests.push_back(0.0);

    for(int i = 0; i < epochs; i++){
        neural_matrix[0] = training_set[i];
        std::cout << "Epoch: " << i << " ---> ";
        for(int j = 0; j < neural_matrix.size() - 1; j++){
            ForwardPropagation(j);
        }
        if(early_stopping != 0 && i > early_stopping){
            for(int j = 0; j < early_stopping; j--){
                if(accurarcy_history[accurarcy_history.size() - 1 - j] <=
                *accuracy_historical_bests.end()){
                    stopping_range++;
                }
                else{
                    stopping_range = 0;
                    break;
                }
            }
            if(stopping_range == early_stopping){
                std::cout << "Model's accuracy haven't improved for "
                << early_stopping << " iterations. Stopping the training." << std::endl;
                break;
            }
        }

        BackwardPropagation(train_labels_set[i], test_set, test_labels_set);
    }
}

double SequentialModel::Predict(std::vector<double> input){
    /*
    Performe a prediction.

    Parameters:
        - input: input data.

    Returns:
        - neural_matrix.back(): output of the neural network.
    */

    neural_matrix[0] = input;

    for(int j = 0; j < neural_matrix.size() - 1; j++){
        ForwardPropagation(j);
    }
    // ICI modifier la prediction
    return neural_matrix.back()[0];
}


void SequentialModel::DisplayAccuraciesHistory(){
    plt::figure();
    plt::plot(accurarcy_history);
    plt::xlabel("Epoch");
    plt::ylabel("accuracy");
    plt::title("history of accuracy obtained during the training");
    plt::savefig("accurarcy_history.pdf");
    // plt::show();
}

void SequentialModel::DisplayPrecisionHistory(){
    plt::figure();
    plt::plot(precision_history);
    plt::xlabel("Epoch");
    plt::ylabel("precision");
    plt::title("history of precision obtained during the training");
    plt::savefig("precision_history.pdf");
    // plt::show();
}

void SequentialModel::DisplayRecallHistory(){
    plt::figure();
    plt::plot(recall_history);
    plt::xlabel("Epoch");
    plt::ylabel("recall");
    plt::title("history of recall obtained during the training");
    plt::savefig("recall_history.pdf");
    // plt::show();
}

void SequentialModel::DisplayF1History(){
    plt::figure();
    plt::plot(f1_history);
    plt::xlabel("Epoch");
    plt::ylabel("F1");
    plt::title("history of f1 obtained during the training");
    plt::savefig("f1_history.pdf");
    // plt::show();
}

void SequentialModel::DisplayAccuraciesHistoricalBests(){
    accuracy_historical_bests.erase(accuracy_historical_bests.begin());
    plt::figure();
    plt::plot(accuracy_historical_bests);
    plt::xlabel("Epoch");
    plt::ylabel("accuracy");
    plt::title("Evolution of the best accuracies obtained during training");
    plt::savefig("accurarcies_historical_bests.pdf");
    // plt::show();
}

void SequentialModel::DisplayLossesHistory(){
    plt::figure();
    plt::plot(losses_history);
    plt::xlabel("Epoch");
    plt::ylabel("loss");
    plt::title("history of losses obtained during the training");
    plt::savefig("losses_history.pdf");
    // plt::show();
}
