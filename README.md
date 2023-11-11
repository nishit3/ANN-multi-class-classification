# ANN-multi-class-classification
## Multi-class classification on iris dataset
ANN contains 4 neurons in the input layer, 24 neurons in the hidden layer, and 3 neurons in the output layer. ReLU is used as an activation function and CrossEntropy as a loss function. Stochastic Gradient Descent(SGD) is used. 

## With vs Without Softmax function  (Experiment)
Randomly generated 2D data is classified into 3 classes, Once using Softmax() as the activation function of the output layer and once no activation function was set for the output layer. Every other parameter is the same. Results deduce that, without using the softmax function the model performance is significantly stable, and mean accuracy is better with a slight margin, although in both cases model got >97% accuracy.
(PyTorch documentation recommends not using Softmax as the CrossEntropy loss function handles it, it now becomes evident from the experiment results)

![without_softmax](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/fb18bdfe-45f0-4a29-a575-fa53893112a4)


![with_softmax](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/a11009f7-877c-4002-869a-302fe3bf07ca)
