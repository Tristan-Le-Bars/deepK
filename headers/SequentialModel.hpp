#ifndef CLASS_SEQUENTIAL_MODEL
#define CLASS_SEQUENTIAL_MODEL

#include <string>
#include <vector>

class SequentialModel{
    private:
        int input_size;
        int layers_nbr;
        std::vector<std::vector<double>> neural_matrix;
        std::vector<std::vector<std::vector<double>>> synaptic_matrix;
        std::vector<double> bias;
        std::vector<std::string> activation_functions_matrix;
        std::string loss_function;
        double learning_rate;

        std::vector<std::vector<double>> MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix);
        std::vector<std::vector<double>> MatrixTransposition(std::vector<std::vector<double>> matrix);
        void ForwardPropagation(int network_position);
        void BackwardPropagation(std::vector<double> labels);

    public:
        // Constructor
        SequentialModel(int input_size);

        // Destructor
        ~SequentialModel();

        // Methods
        void SetLayersNbr(int l);
        void SetNeuralMatrix(std::vector<std::vector<std::vector<double>>> n);
        void SetActivationFuntionMatrix(std::vector<std::string> a);
        void SetLossFunction(std::string l);
        void SetLearningRate(double lr);

        int GetLayerNbr();
        std::vector<std::vector<double>> GetNeuralMatrix();
        std::vector<std::string> GetActivationFunctionMatrix();
        void AddLayer(int neurons_nbr, std::string activation_function);
        void Train(std::vector<std::vector<double>> training_set, std::vector<double> labels_set, int epochs);

};

#endif CLASS_SEQUENTIAL_MODEL