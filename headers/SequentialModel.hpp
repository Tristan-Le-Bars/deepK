#ifndef CLASS_SEQUENTIAL_MODEL
#define CLASS_SEQUENTIAL_MODEL

#include <string>
#include <vector>

class SequentialModel{
    private:
        int input_size;
        int layers_nbr;
        std::vector<std::vector<std::vector<double>>> neural_matrix;
        std::vector<std::vector<std::vector<double>>> synaptic_matrix;
        std::vector<double> bias;
        std::vector<std::string> activation_functions_matrix;
        std::string loss_function;

        double MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix);
        void ForwardPropagation(double bias, int network_position);
        void BackwardPropagation(std::vector<double> labels);

    public:
        // Constructor
        SequentialModel(/* args */);

        // Destructor
        ~SequentialModel();

        // Methods
        void SetLayersNbr(int l);
        void SetNeuralMatrix(std::vector<std::vector<double>> n);
        void SetActivationFuntionMatrix(std::vector<std::string> a);
        void SetLossFunction(std::string l);

        int GetLayerNbr();
        std::vector<std::vector<double>> GetNeuralMatrix();
        std::vector<std::string> GetActivationFunctionMatrix();
        void AddLayer(int neurons_nbr, std::string activation_function);

};

#endif CLASS_SEQUENTIAL_MODEL