#ifndef NEURALLAYER_H
#include "neuralLayer.hpp"
#endif

/*********************** CONSTRUCTORS ******************************/

// Default
NeuralLayer::NeuralLayer()
{
    this->initialized = false;
    this->finalized = false;
    this->activated = false;
    this->inputCount = 0;
}

// Constructor with inputCount and neuronCount - unassigned weights
NeuralLayer::NeuralLayer(unsigned int neuronCount, unsigned int inputCount): NeuralLayer()
{
    // Check if an input count was given
    if (inputCount == 0)
    {
        std::cout << "Error: inputCount required for layer on initialization" << std::endl;
        return;
    }

    // Check if a neuron count was given
    if (neuronCount == 0)
    {
        std::cout << "Error: neuronCount required for layer on initialization" << std::endl;
        return;
    }
    
    // Input count set
    this->inputCount = inputCount;

    // Create neurons based on size (this can be zero)
    for (unsigned short i = 0; i < neuronCount; i++)
    {
        // This creates a neuron with random weights and default assignment
        Neuron neuron(inputCount);

        // Adds the weight to the vector
        this->layer.push_back(neuron);

        // At least one neuron was added, it is now officially initialized
        this->initialized = true;
    }
}

// Constructor with defined neurons
NeuralLayer::NeuralLayer(std::vector<Neuron> *neurons): NeuralLayer()
{
    // Check that the vector is empty or null
    if (!neurons || neurons->size() == 0)
    {
        std::cout << "Error: neuron vector was empty or pointer null" << std::endl;
        return;
    }

    // Check the the supplied layer has the same expected input
    unsigned int inputSizeCheck = 0;
    for ( Neuron neuron : *neurons)
    {
        // Fetch the first input count
        if (inputSizeCheck == 0)
        {
            inputSizeCheck = neuron.getInputCount();
        }
        else
        {
            // If the input count is different across the neurons...
            if (inputSizeCheck != neuron.getInputCount())
            {
                std::cout << "Error: neuron vector contained neurons of differing expected inputs" << std::endl;
                return;
            }
        }
    }

    // Clone the vector
    this->layer = *neurons;
    this->inputCount = inputSizeCheck;
    this->initialized = true;
}

/*********************** DESTRUCTORS *******************************/

// Default Decontructor
NeuralLayer::~NeuralLayer()
{
    // This object maintains ownership of data, no pointers to clean up
} 

/*********************** SETTERS ***********************************/

// Finalize Layer
void NeuralLayer::finalize()
{
    this->finalized = true;
}

/*********************** GETTERS ***********************************/

// Is Initialized?
bool NeuralLayer::isInitialized()
{
    return this->initialized;
}

// Is Finalized?
bool NeuralLayer::isFinalized()
{
    return this->finalized;
}

// Has activated?
bool NeuralLayer::hasActivated()
{
    return this->activated;
}

// Get Layer Input Count
unsigned int NeuralLayer::getInputCount()
{
    return this->inputCount;
}

// Get Layer Memory
std::vector<double> NeuralLayer::getLayerMemory()
{
    return this->layerMemory;
}

// Get Pointer to Neuron in Layer 
Neuron * NeuralLayer::getNeuron(unsigned int neuronIdx)
{   
    // We only set this if our object is initialized
    if (!this->isInitialized())
    {
        std::cout << "Error: Layer not initialized" << std::endl;
        return nullptr;
    }
    if (neuronIdx < (unsigned int)this->layer.size())
    {
        Neuron * returnPtr = &this->layer[neuronIdx];
        return returnPtr;
    }
    return nullptr;
}

/*********************** FUNCTIONAL ********************************/

// Clearn Layer
void NeuralLayer::clearLayer()
{
    this->activated = false;
    this->layerMemory = std::vector<double>();
}

// Add a neuron to the layer
void NeuralLayer::addNeuron(Neuron neuron)
{
    // Check if the layer is locked
    if(this->isFinalized())
    {
        std::cout << "Warning: layer is finalized, skipping add" << std::endl;
        return;
    }

    // Check that the incoming neuron is initialized
    if (!neuron.isInitialized())
    {
        std::cout << "Error: neuron not initialized, skipping add" << std::endl;
        return;
    }

    // Check that the input count aligns with that defined in the layer
    if (neuron.getInputCount() != this->getInputCount())
    {
        std::cout << "Error: neuron input size doesn't match layer input count" << std::endl;
        return;
    }

    // Set the initialization, incase constructed with neuronCount = 0
    this->initialized = true;

    // Add the neuron to the layer
    this->layer.push_back(neuron);
}

// Layer Recall
std::vector<double> NeuralLayer::recall(std::vector<double> *inputs)
{
    // This resets the layer memory
    this->layerMemory = std::vector<double>();

    // Loop through each neuron and recall
    for(Neuron &neuron : this->layer)
    {
        // Add the neurons memory into a vector for easy access
        this->layerMemory.push_back(neuron.recall(inputs));
    }
    this->activated = true;

    // Return the results
    return this->layerMemory;
}


// Neuron Count in Layer
unsigned int NeuralLayer::neuronCount()
{
    return (unsigned int)this->layer.size();
}