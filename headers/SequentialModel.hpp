#ifndef CLASS_SEQUENTIAL_MODEL
#define CLASS_SEQUENTIAL_MODEL

#include <string>
#include <vector>

class SequentialModel{
    private:
        int layers_nbr;
        std::vector<std::vector<double>> neural_matrix;
        std::vector<std::vector<double>> synaptic_matrix;
        std::vector<double> bias;
        std::vector<std::string> activation_functions_matrix;

        void ForwardPropagation(double bias, int network_position);

    public:
        void SetLayersNbr(int l);
        void SetNeuralMatrix(std::vector<std::vector<double>> n);
        void SetActivationFuntionMatrix(std::vector<std::string> a);

        int GetLayerNbr();
        std::vector<std::vector<double>> GetNeuralMatrix();
        std::vector<std::string> GetActivationFunctionMatrix();
        void AddLayer(int neurons_nbr, std::string activation_function);

};

#endif CLASS_SEQUENTIAL_MODEL