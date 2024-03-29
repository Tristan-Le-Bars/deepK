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
        std::vector<double> accurarcy_history;
        std::vector<double> precision_history;
        std::vector<double> recall_history;
        std::vector<double> f1_history;
        std::vector<double> accuracy_historical_bests;
        std::vector<double> losses_history;

        std::vector<std::vector<double>> MatrixMultiplication(std::vector<std::vector<double>> first_matrix, std::vector<std::vector<double>> second_matrix);
        std::vector<std::vector<double>> MatrixTransposition(std::vector<std::vector<double>> matrix);
        double NormalDistribution();
        void ForwardPropagation(int network_position);
        void BackwardPropagation(double label,
                                 std::vector<std::vector<double>> test_set,
                                 std::vector<double> labels_test_set);
        double TestAccuracy(std::vector<std::vector<double>> test_set, std::vector<double> test_labels_set);

    public:
        // Constructor
        SequentialModel(int input_size);

        // Destructor
        ~SequentialModel();

        // Seters
        /*
        void SetLayersNbr(int l);
        void SetNeuralMatrix(std::vector<std::vector<double>> n);
        void SetActivationFuntionMatrix(std::vector<std::string> a);
        */
        void SetLossFunction(std::string l);
        void SetLearningRate(double lr);

        // Methods
        int GetLayerNbr();
        std::vector<std::vector<double>> GetNeuralMatrix();
        std::vector<std::string> GetActivationFunctionMatrix();
        int GetInputSize();

        void AddLayer(int neurons_nbr, std::string activation_function);
        void UniDistribInit();
        void HeInit();
        void Train(std::vector<std::vector<double>> training_set,
                   std::vector<double> labels_set,
                   std::vector<std::vector<double>> test_set,
                   std::vector<double> test_labels_set, int epochs = 0,
                   unsigned int early_stopping = 0);
        double Predict(std::vector<double> input);
        void DisplayAccuraciesHistory();
        void DisplayPrecisionHistory();
        void DisplayRecallHistory();
        void DisplayF1History();
        void DisplayAccuraciesHistoricalBests();
        void DisplayLossesHistory();

};

#endif