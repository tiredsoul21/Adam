#ifndef NEURALLAYER_H
#define NEURALLAYER_H

#ifndef NEURALTYPES_H
#include "neuralTypes.hpp"
#endif

#ifndef NEURON_H
#include "neuron.hpp"
#endif

// #include <vector> // Sourced from neuron.hpp
// #include <iostream> // Sourced from neuron.hpp

/**
 * This class forms an interactive container of sorts, to house and expose
 * individual neurons in a layer.  This layer is the effective owner of a
 * neuron, and will pass pointers to the network and trainer classes.
 * 
 * The neural layer class allows for a random generation of neurons to a set
 * count.  To do this, it is necessary to define define the number of expected
 * inputs for each of the neurons in the layer.  Neurons are untrained. Each 
 * neuron must be manually configured, as they are initialized with defaults.
 * @see Neuron.hpp for default conigurations
 *
 * The alternative is to provide a neuron vector containing all the defined
 * neurons.  The configuration is assumed complete as is, but manual layer
 * configuration / training is still allowed after-the-fact.
 */
class NeuralLayer
{
public: // Public Members
    
public: // Public Methods
    
    /*********************** CONSTRUCTORS ******************************/

    /**
     * Constructor without pre-defined neurons.  This will create randomized
     * weights for neurons.  Be sure to call SRAND external to the network
     * and neuron generation process (EX: in main.cpp)
     */ 
    NeuralLayer(unsigned int neuronCount, unsigned int inputCount);

    /// Constructor with pre-defined neurons
    NeuralLayer(std::vector<Neuron> *layer);

    /*********************** DESTRUCTORS *******************************/

    /// Default
    ~NeuralLayer();
    
    /*********************** SETTERS ***********************************/

    /**
     * This method locks the layer for adding more neurons;
     */
    void finalize();

    /*********************** GETTERS ***********************************/
    
    /**
     * Is the internal mechanism to identify if the object was properly
     * configures and initialized.
     * 
     * @return true - if initialized properly
     * @return false - if not initialized / initialized improperly
     */
    bool isInitialized();

    /**
     * Is the internal mechanism to identify if the object is allowed
     * to be edited with adding neurons.  This is a need for networks, as 
     * adding neurons to a layer already nested impacts expected inputs
     * of other layers.  This is used to prevent adding neurons.
     * 
     * @return true - if layer is finalized
     * @return false - if layer can be modified
     */
    bool isFinalized();

    /**
     * Is the internal mechanism to identify if the layer has been used.
     * This mechanic is used to safeguard the layerMemory.
     * 
     * @return - true - if has activated
     * @return - false - if has not activated (no memory yet)
     */
    bool hasActivated();

    /**
     * This reutrns the nuber of expected inputs into the network
     * 
     * @return - Input size requirements
     */
    unsigned int getInputCount();

    /**
     * This method returns the last activation state for the neurons in
     * this layer
     * 
     * @return - The last activation values of the layer
     */
    std::vector<double> getLayerMemory();

    /**
     * This method exposes the neuron to external changes and read 
     * 
     * @param neuronIdx - the index of the neuron to fetch
     * @return - A pointer to the neuron of interest
     */
    Neuron * getNeuron(unsigned int neuronIdx);

    /*********************** FUNCTIONAL ********************************/
    
    /**
     * This method adds a neuron to the layer.  The order that it
     * is added is the order that is persisted.  If you add a neuron, the
     * input size is compared to the first element for validity.
     * This method is mainly here for deliberate layer building.
     * @param neuron - a neuron to add to the network
     */
    void addNeuron(Neuron neuron);

    /**
     * This method simply sets the 'activated' field to false.  This is to
     * protect callers to the layer to get memory from a previous call.
     * Simple mechanic to protect truth.  This should be called, as a
     * cleanup after layer information is no longer needed.
     */
    void clearLayer();

    /**
     * This method returns the current number of neurons in the layer
     * @returns - number of neurons in the layer
     */
    unsigned int neuronCount();

    /**
     * This method receives an input vector and feeds it through
     * the various layers and neurons.  At the first layer, the inputs
     * are fed to produce ouputs.  From there outputs from successive
     * layers feed the next layer.  The return the the output from the 
     * final layer.
     * The output from the network doesn't regard as trained or untrained...
     * It just fires based on current values.
     * 
     * @param inputs - a vector of double containing the expected inputs
     * @return - the output from the network
     */
    std::vector<double> recall(std::vector<double> *inputs);

private: // Private Members

    /// Valuation of if layer is initialized - default: false
    bool initialized;

    /// This Layer is locked from adding neurons - used in network builds
    bool finalized;

    /// If the layer has been activated
    bool activated;

    /// Retains the the output of the layer after firing
    std::vector<double> layerMemory;

    /// Expected input count for the layer
    unsigned int inputCount;

    /// The layer of neurons
    std::vector<Neuron> layer;

private: // Private Methods
    
    /*********************** CONSTRUCTORS ******************************/

    /// Default contructer is privatized due to initilization requirements
    NeuralLayer();

    /*********************** FUNCTIONAL ********************************/    

};

#endif