// Your First C++ Program

#include <iostream>

#include "neuralNetworkTrainer.hpp"

int main() {
    std::vector<std::vector<double>> inputs(
    {
        std::vector<double>({0.0,0.0}),
        std::vector<double>({0.0,1.0}),
        std::vector<double>({1.0,0.0}),
        std::vector<double>({1.0,1.0})
    });
    std::vector<std::vector<double>> outputs(
    {
        std::vector<double>({0.0}),
        std::vector<double>({1.0}),
        std::vector<double>({1.0}),
        std::vector<double>({1.0})
    });

    std::cout << "------------------Begin Valuation------------------" << std::endl;
    NeuralNetworkTrainer trainer;
    srand ( time(NULL) );
    trainer.addLayer(2,2);
    trainer.addLayer(2);
    trainer.addLayer(1);
    trainer.setDataSplitRatio(1.0);

    trainer.setTrainingCycles(50000);

    std::vector<double> example = {1.1, 2.2};
    clock_t time_req = clock();

    trainer.trainTestLoop(inputs, outputs);
    time_req = clock() - time_req;

    std::cout << "Processor time taken for training: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;


    std::cout << "------------------End Valuation------------------" << std::endl;
    
    return 0;
}