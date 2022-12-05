#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#ifndef NEURALTYPES_H
#include "neuralTypes.hpp"
#endif

#ifndef NEURALLAYER_H
#include "neuralLayer.hpp"
#endif

#include <map> 
// #include <vector> // Sourced from neuron.hpp
// #include <iostream> // Sourced from neuron.hpp

/**
 * This class forms an interactive container of sorts, and is the primary
 * component for neural network functionality.  THis is used to house and
 * expose individual layers for recall and training.  The trainer call can
 * extend / wrap this call to facilitate neural training.
 * 
 * The neural network class allows for a random generation of layers.  To do
 * this, it is necessary to define define the number of expected inputs for 
 * each of the neurons in the layer.  Neurons are untrained. Each neuron 
 * must be manually configured, as they are initialized with defaults.
 * @see Neuron.hpp for default conigurations
 *
 * The alternative is to provide a neural layer OR a network containing
 * all the defined neurons.  The configuration is assumed complete as is,
 * but manual layer configuration is still allowed after-the-fact.
 */
class NeuralNetwork
{
public: // Public Members
    
public: // Public Methods
    
    /*********************** CONSTRUCTORS ******************************/

    /// Default
    NeuralNetwork();

    /// Constructor with pre-defined neural network
    NeuralNetwork(std::vector<NeuralLayer> *network);

    /*********************** DESTRUCTORS *******************************/

    /// Default
    ~NeuralNetwork();
    
    /*********************** SETTERS ***********************************/

    /**
     * This method locks the network for adding more layers
     */
    void finalize();

    /*********************** GETTERS ***********************************/
    
    /**
     * This is the internal mechanism to identify if the network was properly
     * configures and initialized.
     * 
     * @return - true - if initialized properly
     * @return - false - if not initialized / initialized improperly
     */
    bool isInitialized();

    /**
     * This is the internal mechanism to identify if the network is allowed
     * to be edited with adding layers.  This is for network fidelity.  Once
     * training begins, the relationship ought not be modified.  If you're
     * truely wanting to modify the network, you will need to manually 
     * re-create the network.
     * 
     * @return - true - if layer is finalized
     * @return - false - if layer can be modified
     */
    bool isFinalized();

    /**
     * This method returns the value of the accuracy against the trained
     * data set.  This rate has some value, but is not a true measure
     * of the networks ability successfully predict outputs
     * 
     * @return - accuracy rating against trained data
     */
    double getTrainedAccuracy();

    /**
     * This method returns the value of the accuracy against a blind data
     * set.  This rate is assumed the standard for accuracy of the networks
     * ability sucessfully predict outputs given an input.
     * 
     * @return - accuracy rating against a blind dataset
     */
    double getTrueAccuracy();

    /**
     * This reutrns the nuber of expected inputs into the network
     * 
     * @return - Input size requirements
     */
    unsigned int getInputCount();

    /**
     * This method returns the last output state for the network
     * 
     * @return - The last output values of the network
     */
    std::vector<double> getNetworkMemory();

    /*********************** FUNCTIONAL ********************************/
    
    /**
     * This method adds a layer to the network.  The order that it
     * is added is the order that is observed from input --> output.
     * If you add a layer, the input size is compared to the last layer 
     * for validity.
     * 
     * The last added layer is assumed the output layer.  Input count is
     * defaulted, and will only use a value if it's the first layer.
     *
     * This will create randomized weights for neurons.  Be sure to call
     * SRAND external to the network and neuron generation process
     * (EX: in main.cpp)
     * @param neuronCount - the number of neurons to add
     * @param inputCount - the number of inputs expected on the layer
     */ 
    void addLayer(unsigned int neuronCount, unsigned int inputCount = 0);

    /**
     * This method adds a layer to the network.  The order that it
     * is added is the order that is observed from input --> output.
     * If you add a layer, the input size is compared to the last layer 
     * for validity.
     * 
     * The last added layer is assumed the output layer.  Input count must
     * match the previous layer's neuon count (if not first layer)
     * @param layer - the Neural Layer to add to the network
     */
    void addLayer(NeuralLayer layer);

    /**
     * This method returns the number of layers in the network
     * 
     * @return - Integer number of layers
     */
    unsigned int layerCount();

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

protected: // Protected Members

    /// The network of neural layers
    std::vector<NeuralLayer> network;

    // //////////////////////////////////////////////////////////////////////////////////////
    // The variables below are configured and set by the trainer, and cannot be set otherwise
    // //////////////////////////////////////////////////////////////////////////////////////

    /// Accuracy against the last training cycle
    double trainedAccuracy;

    /// Accuracy of trained network againsted a blind dataset
    double trueAccuracy;

private: // Private Members

    /// Valuation of if object is initialized - default: false
    bool initialized;

    /// This network is locked from adding layers
    bool finalized;

    /// Network input size
    unsigned int inputCount;

    /// Retains the the output of the network (last layer) after firing
    std::vector<double> networkMemory;


private: // Private Methods
    
    /*********************** FUNCTIONAL ********************************/    

};

#endif