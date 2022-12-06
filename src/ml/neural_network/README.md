# Neural Networks
## Up Front Assumptions
* This is not a full explaination of Neural Networks or Neurons
* Perceptrons will be referred to as 'Neurons' (same meaning)
* General knowledge on Neurons should be formally sourced and studied independantly
* Multi-Layer Perceptrons (MPL) will be referred to as 'Neural Networks' (same meaning)
* General knowledge on Neural Networks should be formally sourced and studied independantly
* The solutions proposed here are an implementation, but no necessarily an optimized / perfect

## External Resources
* [Perceptrons](https://www.simplilearn.com/tutorials/deep-learning-tutorial/perceptron)
* [Backwards Propagation](https://www.youtube.com/watch?v=tIeHLnjs5U8)
* [Explicit Solutions - to Test Against](https://blog.demofox.org/2017/03/09/how-to-train-neural-networks-with-backpropagation/)
* [Depictive Illustration](http://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Spring.2018/www/slides/lec2.universal.pdf)
* [Book: Machine Learning: An Algorithmic Perspective](https://www.amazon.com/Machine-Learning-Algorithmic-Perspective-Recognition-dp-1466583282/dp/1466583282/ref=dp_ob_title_bk)
* There are a plethera of other examples on the internet, but these are the primary proponents to generate this solution.  Aside from this, 

## Design Concepts
The general structure that we wish to re-create is: 
![Image Missing](../../../artifacts/ml/neural_network/NeuralNetworks-General.png?raw=true "General")
* N - The number of defined inputs for a the network as a whole
* K - The number of layers in the network
  * The first K - 1 layers are 'hidden', and the error cannot be directly assessed.  The Kth is the observable layer in terms of error.
* Each layer is allowed to have any assortment of of neurons that is deemed as necessary
* The number of outputs is the same as the number of neurons in the Kth layer
* All the inputs feed into every neuron in the 1st layer
* All the outputs of a given layer feed into every neuron of the downstream layer
* A neuron is an independent object that knows about itself only
* A neural layer, is an object that contains a vector of neurons, and knows only about itself, and operates against the neural vector
* A neural network, is an object that contains a vector of neural layers, is the primary unit.  It operates against the neural layer vector
* A neural network trainer, is a wrapper that adds onto the the neural network, and provides the training mechanisms to allow the network to optimize against training data.
  * The trainer should allow the neural network to be exportable / importable around training.

A Class level definition is:<br>
![Image Missing](../../../artifacts/ml/neural_network/NeuralNetworks-Classes.png?raw=true "Classes")

## Backwards Propagation Elaboration
The best way to explain the implementation of the backwards propagation is via a simple example.<br>
Setup:
* 2 Inputs
* 3 Layers, with 2 neurons each
* Each neuron has 1 bias and 2 weights (for each input)

and a definition as to why we choose the loop structure we did is demonstrated below:

![Image Missing](../../../artifacts/ml/neural_network/NeuralNetworks-BackwardsProp.png?raw=true "BackwardsProp")

Primary goals of backward propagation:
* Minimize the cost function (Derivatives)
  * Find an optimal amount to change each of the weights by
  * Find an optimal amount to change each of the biases by

The arrow breakout of the network outlines the chain rule terms that are used to calculate the needed terms.

### General Formulas

Cost $=\frac{1}{2}\sum_{i=0}^{1}( a - y )^2,$ <br>
&emsp; &emsp; where y is expected value, a is an Activation Function

Activation Functions:
* $Sigmoid = a = 1/(1-e^\sigma)$<br>

Summation Equation $ =  \sigma =\sum_{m=0}^N (w^l_{nm}*a^{l-1}_{m}) + b^l_n,$ <br>
&emsp; &emsp; where l = layer, m = neuron index in previous layer, n = neuron index in current layer, N = number of neurons, b = bias

### Derivative Formulas
$\frac{\delta C_0}{\delta a^l} = a^l - y$ <br><br>
$\frac{\delta a^l}{\delta z^l} = \sigma'(z^l)$ <br>
&emsp; &emsp; where $\sigma'$ is the derivative of $\sigma$ and is evaluated at the numberic sum of $z^l$ <br><br>
$\frac{\delta z^l}{\delta w^l_{nm}} = a^{l-1}_{m}$<br><br>
$\frac{\delta z^l}{\delta b^l_{n}} = 1$<br>

### Chain Rule Accumulations
This is where the Arrow chart starts to help us.  We will see that terms begin to accumulate and are reuseable

$
\frac{\delta C_0}{\delta a^l_n} = a^l_n - y
$<br><br>

$
\frac{\delta C_0}{\delta z^l_n} = \frac{\delta a^l_n}{\delta z^l_n} \cdot \frac{\delta C_0}{\delta a^l_n}
$<br><br>

$
\frac{\delta C_0}{\delta b^l_{n}} = \frac{\delta z^l_n}{\delta b^l_{n}} \cdot \frac{\delta a^l_n}{\delta z^l_n} \cdot \frac{\delta C_0}{\delta a^l_n} = \frac{\delta z^l_n}{\delta b^l_{n}} \cdot \frac{\delta C_0}{\delta z^l_n} = 1 \cdot \frac{\delta C_0}{\delta z^l_n}
$<br><br>

$
\frac{\delta C_0}{\delta w^l_{nm}} = \frac{\delta z^l_n}{\delta w^l_{nm}} \cdot \frac{\delta a^l_n}{\delta z^l_n} \cdot \frac{\delta C_0}{\delta a^l_n} = \frac{\delta z^l_n}{\delta w^l_{nm}} \cdot \frac{\delta C_0}{\delta z^l_n} = \frac{\delta z^l_n}{\delta w^l_{nm}} \cdot \frac{\delta C_0}{\delta z^l_n}
$<br><br>

$
\frac{\delta C_0}{\delta a^{l-1}_m} = \sum_{i=0}^1 \frac{\delta z^l_n}{\delta a^{l-1}_m} \cdot \frac{\delta a^l_n}{\delta z^l_n} \cdot \frac{\delta C_0}{\delta a^l_n} = \sum_{i=0}^1 \frac{\delta z^l_n}{\delta a^{l-1}_m} \cdot \frac{\delta C_0}{\delta z^l_n}
$

For down stream computations, the same pattern continues, with the activation just calculated will be the common term.  As an exercise, I'd recommend you calculate each term in the given example

See the [External Resources](#external-resources) for further / more detailed discussions.

## Example of Neuron output
Reference the picture below for the following discussion:

![Image Missing](../../../artifacts/ml/neural_network/network_example.png?raw=true "BackwardsProp")

Here, there are 5 neurons in the first layer, and one in the last.  The 5 neurons create a region by intersection by lines.  Now, since the output is only one neuron, we would have only a true or false (in the most simple case).  So the output would be either within the intersection of the 5 lines, or outside.  In a counter example, if we had 5 output neurons, we could begin to classify upto ($2^{5-1}$)=16 different regions of data.

See the [External Resources > Depictive Illustration](#external-resources) for further / more detailed discussions.