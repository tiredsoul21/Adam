#ifndef NEURALNETWORKTRAINER_H
#include "neuralNetworkTrainer.hpp"
#endif

/*********************** CONSTRUCTORS ******************************/

// Constructor with unassigned weights
NeuralNetworkTrainer::NeuralNetworkTrainer(): NeuralNetwork()
{
    // Configure defaults
    this->trainingCycles = this->MAX_TRAINING_CYCLES;
    this->convergenceCount = 0;
    this->convergenceMargin = 0.0;
    this->dataSplitRatio = 0.6f;
    this->learnRate = 0.5f;
    this->currentCycle = 0;
    this->currentConvergenceCount = 0;
}

// Constructor with assigned weights
NeuralNetworkTrainer::NeuralNetworkTrainer(std::vector<NeuralLayer> *network): NeuralNetwork(network)
{
    // Configure defaults
    this->trainingCycles = this->MAX_TRAINING_CYCLES;
    this->convergenceCount = 0;
    this->convergenceMargin = 0.0;
    this->dataSplitRatio = 0.6f;
    this->learnRate = 0.5f;
    this->currentCycle = 0;
    this->currentConvergenceCount = 0;
}

/*********************** DESTRUCTORS *******************************/

NeuralNetworkTrainer::~NeuralNetworkTrainer()
{
    // This object maintains ownership of data, no pointers to clean up
} 

/*********************** SETTERS ***********************************/

// Set Training Cycles
void NeuralNetworkTrainer::setTrainingCycles(unsigned int cycles)
{
    // If the given cycles exceeds the max -- cap them
    if (cycles > this->MAX_TRAINING_CYCLES)
    {
        std::cout << "Warning: cycles were capped to " << this->MAX_TRAINING_CYCLES << std::endl;
        this->trainingCycles = this->MAX_TRAINING_CYCLES;
        return;
    }

    // Set the number of training cycles
    this->trainingCycles = cycles;
}

// Set Convergence Count
void NeuralNetworkTrainer::setConvergenceFactors(unsigned int count, double margin)
{
    // Check if the inputs are 0
    if (count == 0 || margin == 0)
    {
        // Warn if only one is 0 -- are these are a joint function
        if (count != 0 || margin != 0)
        {
            std::cout << "Warning: convergence margin or count 0, both set to 0" << std::endl;
        }
        this->convergenceCount = 0;
        this->convergenceMargin = 0.0;
        return;
    }

    // If the margin is 0, we need to flip this, as the delta is an absolute
    if (margin < 0)
    {
        std::cout << "Warning: convergence margin < 0, this is an absolute.  Changing sign" << std::endl;
        margin = -1*margin;
    }

    // Set the values
    this->convergenceMargin = margin;
    this->convergenceCount = count;
}

// Set Dataset Split Ratio
void NeuralNetworkTrainer::setDataSplitRatio(float ratio)
{
    // Check that the ratio is in range
    if (ratio < 0 || ratio > 1)
    {
        std::cout << "Error: Ratio must be in the range of [0,1], ratio not set" << std::endl;
        return;
    }

    // Set the ratio
    this->dataSplitRatio = ratio;
}

// Set Learning Rate
void NeuralNetworkTrainer::setLearnRate(float rate)
{
    // Check that the ratio is in range
    if (rate <= 0)
    {
        std::cout << "Error: rate must be in the range of (0,inf), ratio not set" << std::endl;
        return;
    }

    // Set the ratio
    this->learnRate = rate;
}

/*********************** GETTERS ***********************************/

// Get Training Cycles
unsigned int NeuralNetworkTrainer::getTrainingCycles()
{
    return this->trainingCycles;
}

// Get Convergence Count
unsigned int NeuralNetworkTrainer::getConvergenceCount()
{
    return this->convergenceCount;
}

// Get Convergence Margin
double NeuralNetworkTrainer::getConvergenceMargin()
{
    return this->convergenceMargin;
}

// Get Dataset Split Ratio
float NeuralNetworkTrainer::getDataSplitRatio()
{
    return this->dataSplitRatio;
}

// Get Learn Rate
float NeuralNetworkTrainer::getLearnRate()
{
    return this->learnRate;
}

// Get Base Network
NeuralNetwork NeuralNetworkTrainer::getNetwork()
{
    return (NeuralNetwork)(*this);
}

/*********************** FUNCTIONAL ********************************/

void NeuralNetworkTrainer::trainTestLoop(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> truths)
{
    // Let's make sure the size of the parameters is the same
    if (inputs.size() != truths.size())
    {
        std::cout << "Error: The size of the inputs does not match size of the truth values" << std::endl;
        return;
    }

    // We grab these here, so we don't have to get this every time in the loop.
    unsigned int networkInputCount = this->getInputCount();
    unsigned int networkOutputCount = this->network[this->layerCount() -1].neuronCount();

    // Purge the data of unexpected inputs
    for(int i = inputs.size() - 1; i >= 0; i--)
    {
        bool pop = false;
        if (inputs[i].size() != networkInputCount)
        {
            std::cout << "Error: The size of the inputs on index: " << i << " not correct, removing" << std::endl;
            pop = true;
        }
        else if (truths[i].size() != networkOutputCount)
        {
            std::cout << "Error: The size of the truths (output) on index: " << i << " not correct, removing" << std::endl;
            pop = true;
        }
        if (pop)
        {
            std::swap(inputs[i], inputs.back());
            inputs.pop_back();
            std::swap(truths[i], truths.back());
            truths.pop_back();
        }
    }

    // We will loop across the dataset for the portion of the data
    unsigned int trainingSize = std::ceil(inputs.size()*this->dataSplitRatio);

    // Create a shuffled index 
    std::vector<int> indexes;
    indexes.reserve(trainingSize - 1);
    for (unsigned int i = 0; i < trainingSize; ++i)
    {
        indexes.push_back(i);
    }

    // Loop across each training cycle
    for(this->currentCycle = 0; this->currentCycle < this->trainingCycles; this->currentCycle++)
    {
        // Shuffle the index
        std::random_shuffle(indexes.begin(), indexes.end());

        // Set up the data index
        unsigned int dataIdx = 0;
        unsigned int averageCount = 0;

        // best neural change is the sum of the cost gradients over a large set
        std::map<std::string, double> cost_gradient;

        double loopCost = 0.0;

        // Loop over the first N*splitRatio dataPoints
        for (;dataIdx < trainingSize; dataIdx++)
        {
            int localIdx = indexes[dataIdx];
            
            // Place the network call (outputs are stored in memory)
            this->recall(&(inputs[localIdx]));

            // Update the cost value for sample
            for (unsigned int i = 0; i < (unsigned int)truths[0].size(); i++)
            {
                double diff = this->network[this->layerCount() - 1].getNeuron(i)->getNeuronMemory() - truths[localIdx][i];
                loopCost += diff*diff;
            }

            std::map<std::string, double> return_cost_gradient = this->backPropigation(&(inputs[localIdx]), &(truths[localIdx]));

            // If the cost_gradient is empty, just copy
            if (cost_gradient.size() == 0)
            {
                cost_gradient = return_cost_gradient;
            }
            // Else sum the two vectors
            else
            {
                std::map<std::string, double>::iterator i = cost_gradient.begin();
                for (; i != cost_gradient.cend(); i++)
                {
                    i->second += return_cost_gradient[i->first];
                }
            }
            averageCount++;

            // Every 100 data samples or final sample, adjust weights
            if (averageCount == 100 || dataIdx == trainingSize -1)
            {
                updateNetworkWeights(cost_gradient, averageCount);
                //std::cout << loopCost / averageCount / 2 << std::endl;
                cost_gradient = std::map<std::string,double>();
                averageCount=0;
            }
        }
        
        //std::cout << "----------------------------------" << std::endl;
    }
}

void NeuralNetworkTrainer::updateNetworkWeights(std::map<std::string, double> weight_change, int count)
{
    // Loop through the network and update the weights
    int layerIdx = 0;
    for (NeuralLayer &layer : this->network)
    {
        for (unsigned int neuronIdx = 0; neuronIdx < layer.neuronCount(); neuronIdx++)
        {
            Neuron *neuron = layer.getNeuron(neuronIdx);
            std::vector<double> newWeights = neuron->getWeights();

            // This string stream used to fetch map keys
            std::stringstream key;

            // Set the new bias weight first
            key << "bias_" << layerIdx << "_" << neuronIdx;
            //std::cout << "bias: " << newWeights[0] << "   " << weight_change.find(key.str())->second << "   ";
            newWeights[0] -= this->learnRate * weight_change[key.str()] / count;
            //std::cout << newWeights[0] << std::endl;
            // Loop though each oh the weights and calculate the weight gradients
            for (int i = 1; i < (int)neuron->getWeights().size(); i++)
            {
                key.str("");
                key << "weight_" << layerIdx << "_" << neuronIdx << "_" << i-1;
                //std::cout << key.str() << ": " << newWeights[i] << "   " << weight_change.find(key.str())->second << "   ";
                newWeights[i] -= this->learnRate * weight_change[key.str()] / count;
                //std::cout << newWeights[i] << std::endl;
            }

            // Set the neuron weights
            neuron->setWeights(&newWeights);
            newWeights = neuron->getWeights();
        }
        layerIdx++;
    }
    
}

std::map<std::string, double> NeuralNetworkTrainer::backPropigation(std::vector<double> * inputs, std::vector<double> * truths)
{
    // This is this run's cost gradient
    std::map<std::string, double> cost_gradient;

    // We start at the outter most layer and calculate backwards
    for (std::vector<NeuralLayer>::reverse_iterator layerIter = this->network.rbegin(); 
         layerIter != this->network.rend(); ++layerIter )
    {
        // We use this for creating maps
        int layerIdx = this->network.rend() - layerIter - 1;

        // We setup the inputs that feed the layer we're working on
        std::vector<double> layerInputs;
        if (layerIdx == 0)
        {
            // First layer the inputs are the feed
            layerInputs = *inputs;
        }
        else
        {
            // Any other layer, the inputs are the previous layer's activation
            layerInputs = this->network[layerIdx -1].getLayerMemory();
        }

        for (unsigned int neuronIdx = 0; neuronIdx < layerIter->neuronCount(); neuronIdx++ )
        {
            // This string stream used to calculate map keys
            std::stringstream key;

            Neuron neuron = *layerIter->getNeuron(neuronIdx);
            double d_activation = 0.0;

            // The activation is different for the output (last) layer.
            if (layerIter == this->network.rbegin())
            {
                //dC_N/da_L; Cost function = 1/2*(a - y)^2
                d_activation = neuron.getNeuronMemory() - (*truths)[neuronIdx];
            }
            else
            {
                // The new cr_activation is the sum of the activation impacts on the next layer
                NeuralLayer previousLayer = this->network[layerIdx + 1];
                for (unsigned int i = 0; i < previousLayer.neuronCount(); i++)
                {
                    // Get the neruon that the activation its in the next layer this is dz^n/da^(n-1)
                    double activationWeight = previousLayer.getNeuron(i)->getWeights()[neuronIdx+1];

                    // Times the previous bias (dC_dZ) terms
                    key.str("");
                    key << "bias_" << layerIdx+1 << "_" << i;
                    d_activation += activationWeight * cost_gradient[key.str()];
                }
            }
            // std::cout << "d_activation: " << d_activation << std::endl;

            // We leverage the fact that the cr_summation == cr_bias
            key.str("");
            key << "bias_" << layerIdx << "_" << neuronIdx;
            double d_bias = d_activation_fun(neuron)*d_activation;
            cost_gradient.insert({key.str(), d_bias});

            // std::cout << "d_bias: " << d_bias << std::endl;

            // Loop though each oh the weights and calculate the weight gradients
            for (int i = 0; i < (int)neuron.getWeights().size() - 1; i++)
            {
                key.str("");
                key << "weight_" << layerIdx << "_" << neuronIdx << "_" << i;
                cost_gradient.insert({key.str(), d_bias*layerInputs[i]});
                // std::cout << key.str() << ": " << d_bias*layerInputs[i] << std::endl;
            }
        }
    }
    return cost_gradient;
}

// d_sigmoid/d_x = (sigmoid(x)*(1-sigmoid(x)))
double NeuralNetworkTrainer::d_activation_fun(Neuron neuron)
{
    switch(neuron.getActivationType())
    {
        case NeuralActivationType::SIGMOID:
            return neuron.getNeuronMemory()*(1-neuron.getNeuronMemory());
        default:
            return 0.0;
    }
}

