# Neural-Networks: MLPs-and-CNNs


In recent years, neural networks have been one of the most powerful machine learning models.
Many toolboxes/platforms (e.g., TensorFlow, PyTorch, Torch, Theano, MXNet, Caffe, CNTK)
are publicly available for efficiently constructing and training neural networks. The core idea of
these toolboxes is to treat a neural network as a combination of data transformation modules.
Now we will provide more information on modules for this project. Each module has its
own parameters (but note that a module may have no parameters). Moreover, each module can
perform a forward pass and a backward pass. The forward pass performs the computation of the
module, given the input to the module. The backward pass computes the partial derivatives of
the loss function w.r.t. the input and parameters, given the partial derivatives of the loss function
w.r.t. the output of the module. Consider a module \<module name\>. Let \<module name\>.forward
and \<module name\>.backward be its forward and backward passes, respectively.

# Dataset

We will use mnist subset (images of handwritten digits from 0 to 9). As before, the dataset is stored in a JSON-formated
file mnist subset.json. You can access its training, validation, and test splits using the keys
‘train’, ‘valid’, and ‘test’, respectively. For example, suppose we load mnist subset.json to the
variable x. Then, x\['train'\] refers to the training set of mnist subset. This set is a list with two
elements: x\['train'\] \[0\] containing the features of size N (samples) ×D (dimension of features), and
x\['train'\]\[1\] containing the corresponding labels of size N.


# Cautions

Please do not import packages that are not listed in the provided code. Follow the instructions
in each section strictly to code up your solutions. Do not change the output format. Do
not modify the code unless we instruct you to do so.


# Implementing Modules

The task is to finish the implementation of several modules, where these modules are elements of a multi-layer perceptron (MLP) or a convolutional neural network (CNN). We will apply these models to the same 10-class classification problem.
We will train the models using stochastic gradient descent with minibatch, and explore how different hyperparameters of optimizers and regularization techniques affect training and validation accuracies over training epochs. 

Please read through dnn_mlp.py and dnn_cnn.py. Both files will use modules defined in
dnn_misc.py (which you will modify). The idea is to understand how modules are created, how
they are linked to perform the forward and backward passes, and how parameters are updated based
on gradients (and momentum). The dnn_misc.py file defines all modules that you will need to construct
the MLP and CNN in dnn mlp.py and dnn cnn.py, respectively. You have three tasks. First,
finish the implementation of forward and backward functions in class linear layer. Second, finish the implementation of forward and backward functions in class relu. Third, finish the the implementation of backward function in class dropout.


# Multi-layer Perceptron

**1.** Running q53.sh will run python3 dnn mlp.py with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.0. The output file stores the training and validation accuracies over 30 training epochs. It will output MLP lr0.01 m0.0 w0.0 d0.0.json.

**2.** Running q54.sh will run python3 dnn mlp.py --dropout rate 0.5 with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs. It will output MLP lr0.01 m0.0 w0.0 d0.5.json.

**3.** Running q55.sh will run python3 dnn mlp.py --dropout rate 0.95 with learning rate 0.01,
no momentum, no weight decay, and dropout rate 0.95. The output file stores the training and
validation accuracies over 30 training epochs. It will output MLP lr0.01 m0.0 w0.0 d0.95.json.
You will observe that the model in 2. will give better validation accuracy (at epoch 30)
compared to 1. Specifically, dropout is widely-used to prevent over-fitting. However, if we use
a too large dropout rate (like the one in 3.), the validation accuracy (together with the training
accuracy) will be relatively lower, essentially under-fitting the training data

**4.** Running q56.sh will run python3 dnn mlp nononlinear.py with learning rate 0.01, no
momentum, no weight decay, and dropout rate 0.0. The output file stores the training and validation
accuracies over 30 training epochs. It will output LR lr0.01 m0.0 w0.0 d0.0.json.
The network has the same structure as the one in 1., except that we remove the relu (nonlinear)
layer. You will see that the validation accuracies drop significantly (the gap is around 0.03).
Essentially, without the nonlinear layer, the model is learning multinomial logistic regression.

# Single Layer Convolutional Neural Network

**5.** Running  q57.sh will run python3 dnn cnn.py with learning rate 0.01, no momentum, no
weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies
over 30 training epochs. It will output CNN lr0.01 m0.0 w0.0 d0.5.json.

**6.** Running q58.sh will run python3 dnn cnn.py --alpha 0.9 with learning rate 0.01, momentum
0.9, no weight decay, and dropout rate 0.5. The output file stores the training and validation
accuracies over 30 training epochs. It will output CNN lr0.01 m0.9 w0.0 d0.5.json.
You will see that this will lead to faster convergence than 5. (i.e., the training/validation
accuracies will be higher than 0.94 after 1 epoch). That is, using momentum will lead to more
stable updates of the parameters.

# Building a deeper architecture: Constructing a two-convolutional-layer CNN

The CNN architecture in dnn_cnn.py has only one convolutional layer. Now, you are going to construct a two-convolutional-layer CNN. The code in dnn_cnn 2.py is similar to that in dnn_cnn.py, except that there are a few parts marked as TODO. You need to fill
in your code so as to construct the CNN.

**7.** Running q510.sh will run python3 dnn cnn 2.py --alpha 0.9 with learning rate 0.01,
momentum 0.9, no weight decay, and dropout rate 0.5. The output file stores the training and
validation accuracies over 30 training epochs. It will output CNN2 lr0.001 m0.9 w0.0 d0.5.json.
You will see that you can achieve slightly higher validation accuracies than those in 6.
