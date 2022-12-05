#ifndef NEURALNETWORK_H
#include "neuralNetwork.hpp"
#endif

/*********************** CONSTRUCTORS ******************************/

// Default
NeuralNetwork::NeuralNetwork()
{
    this->initialized = false;
    this->finalized = false;
    this->inputCount = 0;

    // Initialize the network ratings to null values, these are otherwise only set by a trainer
    this->trainedAccuracy = -1.0;
    this->trueAccuracy = -1.0;
}

// Constructor with defined layers
NeuralNetwork::NeuralNetwork(std::vector<NeuralLayer> *network): NeuralNetwork()
{
    // Check that the vector is empty or null
    if (!network || network->size() == 0)
    {
        std::cout << "Error: network was empty or pointer null" << std::endl;
        return;
    }

    // Clone the vector
    this->network = *network;
    this->inputCount = this->network.at(0).getInputCount();

    // If we made it here, should be good
    this->initialized = true;
}

/*********************** DESTRUCTORS *******************************/

NeuralNetwork::~NeuralNetwork()
{
    // This object maintains ownership of data, no pointers to clean up
} 

/*********************** SETTERS ***********************************/

// Finalize Network
void NeuralNetwork::finalize()
{
    this->finalized = true;
}

/*********************** GETTERS ***********************************/

// Is Initialized?
bool NeuralNetwork::isInitialized()
{
    return this->initialized;
}

// Is Finalized?
bool NeuralNetwork::isFinalized()
{
    return this->finalized;
}

// Get Training Accuracy
double NeuralNetwork::getTrainedAccuracy()
{
    return this->trainedAccuracy;
}

// Get True Accuracy
double NeuralNetwork::getTrueAccuracy()
{
    return this->trueAccuracy;
}


// Returns Network Expected Input
unsigned int NeuralNetwork::getInputCount()
{
    return this->inputCount;
}

// Get Network Memory
std::vector<double> NeuralNetwork::getNetworkMemory()
{
    return this->networkMemory;
}

/*********************** FUNCTIONAL ********************************/

// Add Defaul Layer to Network
void NeuralNetwork::addLayer(unsigned int neuronCount, unsigned int inputCount)
{
    // Check that the first layer has a defined input count
    if (this->layerCount() == 0 && inputCount == 0)
    {
        std::cout << "Error:  First layer must be initialized with input count" << std::endl;
        return;
    }
    // If not the first layer, always retrieve it ourselves
    else if (this->layerCount() == 0)
    {
        this->inputCount = inputCount;
    }
    else
    {
        inputCount = this->network[this->layerCount() - 1].neuronCount();
        this->network[this->layerCount() - 1].finalize();
    }

    // Finalize the previous layer

    // Add a new layer to the network
    NeuralLayer layer(neuronCount, inputCount);
    this->network.push_back(layer);
}

// Add Layer to Network
void NeuralNetwork::addLayer(NeuralLayer layer)
{
    // Check if the network is locked
    if(this->isFinalized())
    {
        std::cout << "Warning: network is finalized, skipping add" << std::endl;
        return;
    }

    // Check if the incoming layer is initialized
    if (!layer.isInitialized())
    {
        std::cout << "Error: Layer not initialized, skipping add" << std::endl;
        return;
    }

    // If this is the first layer, capture the input count
    if (this->layerCount() == 0)
    {
        this->inputCount = layer.getInputCount();
    }
    // If it's not the first layer, check that the input count is correct
    else if (layer.getInputCount() != this->network[this->layerCount() - 1].neuronCount())
    {
        std::cout << "Error: Layer input size does not match previous layer output size, skipping add" << std::endl;
        return;
    }
    // If this is not the first layer
    else
    {
        this->network[this->layerCount() - 1].finalize();
    }
    
    // Add the layer
    this->network.push_back(layer);
}

// Get Layer Count
unsigned int NeuralNetwork::layerCount()
{
    return (unsigned int)this->network.size();
}

// Network Recall
std::vector<double> NeuralNetwork::recall(std::vector<double> *input)
{
    std::vector<double> * layerInput = input;
    std::vector<double> layerOutput;

    // We look through the layers, and forward feed the inputs
    for (NeuralLayer &layer : this->network)
    {
        // Loop through and recall each layer.
        layerOutput = layer.recall(layerInput);

        layerInput = &layerOutput;
    }
    
    // Network output is the output of the last layer
    this->networkMemory = layerOutput;
    return this->networkMemory;
}