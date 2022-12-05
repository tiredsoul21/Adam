#ifndef NEURON_H
#include "neuron.hpp"
#endif

/*********************** CONSTRUCTORS ******************************/

// Default
Neuron::Neuron()
{
    // inputCount is initialized to 0 (uninitialized neuron)
    this->initialized = false;
    this->activated = false;
    this->neuronMemory = -12345678.9;
    this->inputCount = 0;
    this->activationType = NeuralActivationType::SIGMOID;
}

// Constructor with inputCount - unassigned weights
Neuron::Neuron(unsigned int inputCount): Neuron()
{
    // Preliminary checks
    if ( inputCount == 0 )
    {
        std::cout << "Error: inputCount must be greater than 0" << std::endl;
        return;
    }
    
    // Set up new assessment
    this->inputCount = inputCount;
    this->weightSize = (inputCount + 1);
    
    // Generate random weights by default
    for (unsigned int i = 0; i < this->weightSize; i++)
    {
        this->weights.push_back(.6*(double)rand() / RAND_MAX - 0.3);
    }
    
    // Set status as initialized
    this->initialized = true;
}

// Constructor with assigned weights
Neuron::Neuron(std::vector<double> *weights): Neuron()
{
    // Preliminary checks
    if ( weights->empty() )
    {
        std::cout << "Error: weights cannot be empty" << std::endl;
        return;
    }
    this->inputCount = weights->size() - 1;
    if (inputCount == 0 )
    {
        std::cout << "Error: Failed to initialize perceptron:: "
                  << "Invalid weightArray size (size >= 2)" << std::endl;
        return;
    }
    
    // Set up new assessment
    this->inputCount = inputCount;
    this->weights = *weights;
    this->weightSize = inputCount + 1;
    
    // Set status as initialized
    this->initialized = true;
}

/*********************** DESTRUCTORS *******************************/

Neuron::~Neuron()
{
    // This object maintains ownership of data, no pointers to clean up
} 

/*********************** SETTERS ***********************************/

// Set Weights
void Neuron::setWeights(std::vector<double> *weights)
{
    // Check that the vector is empty or null
    if (!weights || weights->size() == 0)
    {
        std::cout << "Error: weights vector was empty or pointer null" << std::endl;
        return;
    }

    // We only set this if our object is initialized
    if (!this->isInitialized())
    {
        std::cout << "Error: Neuron not initialized" << std::endl;
        return;
    }

    // Check that the weight size is what we're expecting
    if ((unsigned int)weights->size() < this->weightSize )
    {
        std::cout << "Error: Failed to set weights:: "
                  << "Invalid weightArray size" << std::endl;
        return;
    }
    
    // Should be safe for assignment
    this->weights = *weights;
}

// Set Activation Type
void Neuron::setActivationType(NeuralActivationType type)
{
    this->activationType = type;
}

/*********************** GETTERS ***********************************/

// Is Initialized?
bool Neuron::isInitialized()
{
    return this->initialized;
}

// Has activated?
bool Neuron::hasActivated()
{
    return this->activated;
}

// Get Weights
std::vector<double> Neuron::getWeights()
{
    return this->weights;
}

// Get Input Count
unsigned int Neuron::getInputCount()
{
    return this->inputCount;
}

// Get Neuron Memory
double Neuron::getNeuronMemory()
{
    return this->neuronMemory;
}

// Get Activation Type
NeuralActivationType Neuron::getActivationType()
{
    return this->activationType;
}

/*********************** FUNCTIONAL ********************************/

// Clear Neuron
void Neuron::clearNeruon()
{
    this->activated = false;
    this->neuronMemory = -12345678.9;
}

// Neuron Recall Method
double Neuron::recall(std::vector<double> *inputs)
{
    // Check that the vector is empty or null
    if (!inputs || inputs->size() == 0)
    {
        std::cout << "Error: inputs vector was empty or pointer null" << std::endl;
        return -12345678.9;
    }

    // We only set this if our object is initialized
    if(!this->isInitialized())
    {
        std::cout << "Error: Neuron not initialized" << std::endl;
        return -12345678.9;
    }

    // Check that the input size is what we're expecting
    if(inputs->size() < inputCount)
    {
        std::cout << "Error: invalid input size, expected " << inputCount << std::endl;
        return -12345678.9;
    }

    // Get some iterators
    std::vector<double>::iterator weightsItr = this->weights.begin();
    std::vector<double>::iterator inputsItr = inputs->begin(); 

    // Calculate sum with the bias first multiplied by one
    double localSum = *(weightsItr);
    weightsItr++;

    // Sum the input*weight vectors
    for (; inputsItr != inputs->cend(); ++inputsItr, ++weightsItr)
    {
        localSum += (*weightsItr) * (*inputsItr);
    }

    // Let's assume that we activated first
    this->activated = true;

    // Return value based upon activation type
    switch(this->activationType)
    {
        // SWITCH
        case NeuralActivationType::SWITCH:
            this->neuronMemory = (localSum > 0.0)? 1.0 : 0.0;
            break;

        // SIGMOID
        case NeuralActivationType::SIGMOID:
            this->neuronMemory = 1/ ( 1 + std::exp(-localSum));
            break;

        // HYPERBOLIC TANGENT
        case NeuralActivationType::HYPERBOLIC_TANGENT:
            this->neuronMemory = std::tanh(localSum);
            break;

        // RAW
        case NeuralActivationType::RAW:
            this->neuronMemory = localSum;
            break;

        // CATEGORICAL
        case NeuralActivationType::CATEGORICAL:
            this->neuronMemory = std::floor(localSum);
            break;

        // ??
        default:
            // Somehow we got here, unset neuron
            this->neuronMemory = -12345678.9;
            this->activated = false;
            break;
    }

    return this->neuronMemory;
}
