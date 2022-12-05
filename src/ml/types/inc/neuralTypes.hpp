#ifndef NEURALTYPES_H
#define NEURALTYPES_H

/**
 * This enumeration is used for the switching of the neuron /
 * trainer for activation type
 * RAW - The releases the output of the neuron without activation
 * SWITCH - 0.0 or 1.0 (false / true respectively) - discontinuous
 * SIGMOID - this function  bound between 0 & 1, and is continuous
 * CATEGORICAL - This is a floor function evaluation
 * HYPERBOLIC_TANGENT - this function  bound between 0 & 1, and is continuous
 * 
 */
enum class NeuralActivationType
{
    /** Returns the value as summed, is an evaluation (best use is output) */
    RAW,

    /**
     * Returns a 0.0 or 1.0 if the threshold was breached.  Not for practical use
     * as the function is not differentable, and thus cannot readily be trained.
     * This could theoretically be plugged into a sigmoid output once it has been
     * properly trained
     */
    SWITCH,

    /**
     * Returns a value between (0,1), better use as a switch as this method
     * is differentiable. Rounding or external processing would help to make the
     * output more defined...I.E. .09756... should be evaluated as 1.0.
     */
    SIGMOID,

    /**
     * Returns the Floor of the summed value, is an evaluation (best used is output)
     * This method is not differntiable, and so should not be used during training,
     * This actication type coule be substituted for output trigger for raw, once it
     * has been trained properly.
     */
    CATEGORICAL,

    /**
     * Returns a value between (-1,1), usesable as a switch as this method
     * is differentiable. Rounding or external processing would help to make the
     * output more defined...I.E. .09756... should be evaluated as 1.0.
     */
    HYPERBOLIC_TANGENT
};

/**
 * Enumeration to distinguish between different datasets
 */
enum class DatasetType
{
    /** Binary Classifiers for Blind Dataset Statistics */
    TRUE,

    /** Binary Classifiers for Training Dataset Statistics */
    TRAINING
};

/**
 * Enum to distinguish between different dataset classifiers statistics
 */
enum class BinaryClassifierType
{
    /** Classifiers for a result that was falsely predicted Positive */
    FALSE_POSITIVE,

    /** Classifiers for a result that was falsely predicted Negative */
    FALSE_NEGATIVE,

    /** Classifiers for a result that was correctly predicted Positive*/
    TRUE_POSITIVE,

    /** Classifiers for a result that was correctly predicted Negative */
    TRUE_NEGATIVE
};

#endif