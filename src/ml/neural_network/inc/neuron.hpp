#ifndef NEURON_H
#define NEURON_H

#ifndef NEURALTYPES_H
#include "neuralTypes.hpp"
#endif

#include <time.h>
#include <math.h>
#include <vector>
#include <iostream>

/**
 * This class contains the necessary components for a functional neuron,
 * the inputs coupled with weights and activation method create a neuron.
 * This class also contains the recall function and calculates the activation
 * method.  In order to keep this class lightweight, this is decoupled from the 
 * training mechanics.
 * 
 * The number of weights (n + 1) is determined by the number of inputs (n).  The
 * additional weight is to form a bias in the case of the zero solution in the
 * training set.  If all the inputs are 0, there is a bias factor w_0 (an
 * additional weight) that is always multiplied by a constant -1 to prevent an
 * indeterminate zero sum.
 * 
 * The outputs are dependant upon the activation method given.
 * SWITCH - always 1 or 0 (fire / not fire respectively).
 * SIGMOID - returns a value between 0 and 1 (smooth switch)
 * HYPERBOLIC_TANGENT - returns a value between -1 and 1 (smooth switch)
 * CATEGORICAL - This is a floor function, which maps to a round number
 *      this is likely best used as an output for a cataforgical determination
 *      Note: This is not an activation, but an evaluation
 * RAW - Returns the weighted sum, this likely best served as an output layer neuron
 *      Note: This is not an activation, but an evaluation
 */
class Neuron
{
public: // Public Members
    
public: // Public Methods
    
    /*********************** CONSTRUCTORS ******************************/

    /**
     * Constructor without pre-defined weights.  This will create randomized
     * weights for neurons.  Be sure to call SRAND external to the network
     * and neuron generation process (EX: in main.cpp)
     */ 
    Neuron(unsigned int inputCount);

    /// Constructor with pre-defined weights
    Neuron(std::vector<double> *weights);

    /*********************** DESTRUCTORS *******************************/

    /// Default
    ~Neuron();
    
    /*********************** SETTERS ***********************************/
    
    /**
     * Set the Weights vector.  This is the set of weights the pair 
     * with the inputs to calculate the activation.  This will also include
     * the bias weight in the first position, and therefore the number of 
     * weights is = (n + 1), where n is the number of inputs.
     * 
     * @param weights - a vector of weights for inputs and bias
     */
    void setWeights(std::vector<double> *weights);

    /**
     * Set the Activation Type.  This is for the congiuration for how the 
     * neuron will respond to the inputs.
     * @see NeuralActivationType
     * 
     * @param type - the neuron response mechanic
     */
    void setActivationType(NeuralActivationType type);

    /*********************** GETTERS ***********************************/

    /**
     * This is the internal mechanism to identify if the neuron was properly
     * configures and initialized.
     * 
     * @return - true - if initialized properly
     * @return - false - if not initialized / initialized improperly
     */
    bool isInitialized();

    /**
     * Is the internal mechanism to identify if the neuron has been used.
     * This mechanic is used to safeguard the neuronMemory.
     * 
     * @return - true - if has activated
     * @return - false - if has not activated (no memory yet)
     */
    bool hasActivated();
    
    /**
     * Get the Weights vector.  This allows the caller to store the weights 
     * for trainingless configuration in future.
     * 
     * @return - a copy of the weights
     */
    std::vector<double> getWeights();

    /**
     * Get the Input Count based off of initialization
     * 
     * @return - The number of inputs expected
     */
    unsigned int getInputCount();

    /**
     * Get the Activation Type currently used on this neuron
     * 
     * @return - Current set activation method 
     */
    NeuralActivationType getActivationType();

    /**
     * This method returns the last neural activaty.  If the neuron
     * hasn't previously activated, will return a dummy value of -12345678.9
     * It is best to wrap any calls to this function with
     * a check of neuron->hasActivated()
     * 
     * @return - The neural activation value 
     * @return - (-12345678.9) if neuron hasn't activated yet
     */
    double getNeuronMemory();

    /*********************** FUNCTIONAL ********************************/

    /**
     * This method takes the input values, and calculated the activation
     * method against the weights.  This value can be retrieved later with
     * neuron->getNeuonMemory()
     * 
     * @param inputs - input values to evaluate the neuron against
     * @return - activate return value
     */
    double recall(std::vector<double> *inputs);

    /**
     * This method simply sets the 'activated' field to false.  This is to
     * protect callers to the neuron to get memory from a previous call.
     * Simple mechanic to protect truth.  This should be called, as a
     * cleanup after neuron information is no longer needed.
     */
    void clearNeruon();

private: // Private Members

    /// Valuation of if neuron is initialized - default: false
    bool initialized;

    /// Valuation of if neuron has activated
    bool activated;
    
    /// Short value containing number of inputs - default: 0
    unsigned int inputCount;
    
    /// Retrains the last neural stimulus - default (dummy): -12345678.9
    double neuronMemory;

    /**
     * Contains the weights for all n inputs including the additional bias 
     * input weight, and so the total size of this vector is (n + 1)
     * default: empty vector
     */
    std::vector<double> weights;
    
    /// The total size of the weight vector: (inputSize + 1)
    unsigned int weightSize;

    /// Used to define the activation function (output) - default: SIGMOID
    NeuralActivationType activationType;

private: // Private Methods

    /*********************** CONSTRUCTORS ******************************/

    /// Default contructer is privatized due to initilization requirements
    Neuron();

    /*********************** FUNCTIONAL ********************************/    

};

#endif