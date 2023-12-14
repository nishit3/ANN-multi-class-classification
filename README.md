# ANN-multi-class-classification
## Multi-class classification on iris dataset
ANN contains 4 neurons in the input layer, 24 neurons in the hidden layer, and 3 neurons in the output layer. ReLU is used as an activation function and CrossEntropy as a loss function. Stochastic Gradient Descent(SGD) is used. 

## Accuracy as a function of model width and depth  (Experiment)
ANN models are trained for classification on the iris dataset each with a different set of (depth, width). With depth (1 to 5) (depth = no. of hidden layers) and width (1 to 100) (width = no. of neurons in each hidden layer). It can be deduced that accuracy is not directly proportional to the depth of model architecture, but less dense models are outperforming in discrete ranges. Although irrespective of depth, accuracy is always improving with more width.

![depth and width impact](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/a0fe7f01-ad7d-4004-8803-9f160639ab97)

## With vs Without Softmax function  (Experiment)
Randomly generated 2D data is classified into 3 classes, Once using Softmax() as the activation function of the output layer and once no activation function was set for the output layer. Every other parameter is the same. Results deduce that, without using the softmax function the model performance is significantly stable, and mean accuracy is better with a slight margin, although in both cases model got >97% accuracy.
(PyTorch documentation recommends not using Softmax as the CrossEntropy loss function handles it, it now becomes evident from the experiment results)

![without_softmax](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/fb18bdfe-45f0-4a29-a575-fa53893112a4)


![with_softmax](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/a11009f7-877c-4002-869a-302fe3bf07ca)

### With vs Without Batch Normalization (With Softmax)
It stabilized model performance and reduced accuracy dips.
![Figure_2](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/e0dc6fdb-5487-4bb0-8fde-618af3644176)

## Digit classification on MNIST dataset
Optimizer:- Adam  
Loss Function:- Negative Log Likelihood  
Learning Rate:- 0.01  
Epochs:- 100  
[Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/)

![Figure_1](https://github.com/nishit3/ANN-multi-class-classification/assets/90385616/c60b76e6-8e07-46aa-9828-50ccee5d45d6)
