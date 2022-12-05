#ifndef NEURALNETWORKTRAINER_H
#define NEURALNETWORKTRAINER_H

#ifndef NEURALTYPES_H
#include "neuralTypes.hpp"
#endif

#ifndef NEURALLAYER_H
#include "neuralLayer.hpp"
#endif

#ifndef NEURALNETWORK_H
#include "neuralNetwork.hpp"
#endif

#include <mutex>
#include <thread>
#include <sstream>
#include <algorithm>

// #include <map> // Sourced from neuralNetwork.hpp

/**
 * This class is a heavy weight wrapper to the Neural Nerwork class.  The
 * primary purpose of this class is to train the base class network. A
 * Neural Network that no longer needs to learn or train should be exported
 * to the base class.
 * @see NeuralNetwork.hpp
 * 
 * Whereas the trainer does consumer more resources for RAM and CPU time,
 * It is not unreasonable to simply leave the trainer for production runs.
 * Other reasons why one might desire to leave the trainer in place would
 * be for feedback purposes, as well as continuous learning.  A continuous
 * learning use provides no assurances to the accuracy of the network beyond
 * initial training.  It would be recommended retrain the network from scratch
 * to include the new data points.  By doing so a valid accuracy determination
 * can be made. The recall and the backward propigation are decoupled to allows
 * for this use case.
 */
class NeuralNetworkTrainer: public NeuralNetwork
{
public: // Public Members

    /// This defines a threshold for the maxium number of training cylces on training data
    const static int MAX_TRAINING_CYCLES = 1000000;

public: // Public Methods
    
    /*********************** CONSTRUCTORS ******************************/

    /// Trainer with unassigned weights
    NeuralNetworkTrainer();

    /// Trainer with assigned weights
    NeuralNetworkTrainer(std::vector<NeuralLayer> *network);

    /*********************** DESTRUCTORS *******************************/

    /// Default
    ~NeuralNetworkTrainer();
    
    /*********************** SETTERS ***********************************/

    /**
     * This method sets the current setting for the number of training
     * cycles before a a network training concludes.  If this number is 
     * reached in a training loop, it will break automatically.  If however,
     * the convergence factors are set, then the training cycles will
     * automatically break if defined convergence is achieved.
     * 
     * @param cycles - the number of training cycles before training stop
     */
    void setTrainingCycles(unsigned int cycles);

    /**
     * This method returns sets the convergence factors for the training
     * loop.  If the number or training cycles is achieved before this,
     * than the training will stop.  The convergence factors are determined
     * as follows.  If the training accuracy doesn't change by a factor
     * greater than 'margin' for 'count' number of times, convergence is
     * achieved.  This will stop training even if the number of cycles
     * hasn't been achieved.
     * 
     * If either value is 0, convergence is ignored, and the training will
     * continue until max training cycles is reached.  Margin should be
     * positive, since the training difference is absolute.
     * 
     * @param count - the number of times the result must be less than margin
     *      This is a consecutive count.
     * @param margin - the delta between consecutive training accuracies must
     *      be less than this value.
     */
    void setConvergenceFactors(unsigned int count, double margin);

    /**
     * This method sets the ratio at which the dataset will be split into
     * two datasets.  The value entered will be the portion of the data
     * dedicated to training., the ratio should be between [0,1] inclusive
     * EX: Ratio = .6 --> 60% training data 40% blind test data
     * 
     * @param ratio - the portion of input data used to train the network
     */
    void setDataSplitRatio(float ratio);

    /**
     * This method sets the learning rate for the network.  This could cause
     * the network to converge more slowly, or provide momentum to 'jump'
     * past local minima.  Can play with this to have differing impacts on
     * solutions.  Note, a rate too large could cause the network to fluctuate
     * without convergence.
     * 
     * @param rate - the learn rate multiplier
     */
    void setLearnRate(float rate);

    /*********************** GETTERS ***********************************/

    /** 
     * This method returns the currently set number of training cycles.  This
     * is the number of times the network will be trained against the dataset
     * 
     * @return - number of training cycles
     */
    unsigned int getTrainingCycles();

    /**
     * This returns the current convergence count.
     * 
     * @return - the definition of convergence (# of times)
     */
    unsigned int getConvergenceCount();

    /**
     * This returns the currenct convergence margin
     * 
     * @return - the definition of convergence (consecutive delta tolerance)
     */
    double getConvergenceMargin();

    /**
     * This returns the current setting for the portion of data dedicated
     * to training the network
     * 
     * @return - the training dataset ratio
     */
    float getDataSplitRatio();

    /**
     * TThis returns the current learning rate for the network.  This will 
     * slow / speed up the changes in the weights.
     * 
     * @return - the current learning rate
     */
    float getLearnRate();

    /**
     * This returns the the base class of the training network (which is simply
     * a NeuralNetwork).  This object will only have the necessary pieces
     * needed to fire a network as has already been trained against.
     * 
     * @return - base class of trainer
     */
    NeuralNetwork getNetwork();

    /*********************** FUNCTIONAL ********************************/

    void trainTestLoop(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> truths);

private: // Private Members

    /// The number of training cycles before training stops - default: 1,000,000
    unsigned int trainingCycles;

    /// The number of times difference between successive accuracies < margin - default: 0 (off)
    unsigned int convergenceCount;

    /// The threshold for the difference between successive accuracies - default: 0.0 (off)
    double convergenceMargin;

    /// Ratio of data set aside for strict training - default: 0.6f (60%)
    float dataSplitRatio;

    /// Learning rate for the network - default 0.2f
    float learnRate;

    /// Global index for current training cycle
    unsigned int currentCycle;

    /// Global index keeping track of number of times accuracy deltas < margin
    unsigned int currentConvergenceCount;

    std::map<std::string, double> dataMap;

private: // Private Methods
    
    /*********************** FUNCTIONAL ********************************/    

    void train(std::vector<double> inputs, std::vector<double> truths);

    void updateNetworkWeights(std::map<std::string, double> weight_change, int count);

    std::map<std::string, double> backPropigation(std::vector<double> * inputs, std::vector<double> * truths);

    double d_activation_fun(Neuron neuron);

    double d_sigmoid(double value, bool recalc = false);

};

#endif