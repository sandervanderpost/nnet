# -*- coding: utf-8 -*-
import numpy as np

# Neural network implementation

class NeuralNet:
    
    def __init__(self, 
                 nr_of_nodes,
                 activation_function,
                 learning_rate,
                 targets):
        self.nodes = nr_of_nodes
        self.acf = act_f(activation_function)
        self.lr = learning_rate
        self.targets = targets
        
    # Initialize starting weights
    def init_weights(self,
                     nodes_input,
                     nodes_hidden,
                     nodes_output):
        self.weights_ih = np.random.rand(nodes_input, nodes_hidden)
        self.weights_ho = np.random.rand(nodes_hidden, nodes_output)
        self.bias_ih = np.random.rand(nodes_hidden)
        self.bias_ho = np.random.rand(nodes_output)
            
    # Feed forward step
    def feed_forward(self,
                     data):
        
#        print ("Initial weights are: ", self.weights_ih, "input -> hidden", "\n")
#        print (self.weights_ho, "hidden -> output", "\n")
        
        weighted_input_hidden_layer = np.dot(np.transpose(self.weights_ih), data) + self.bias_ih
        first_layer = self.acf(weighted_input_hidden_layer)
        
        weighted_input_output_layer = np.dot(np.transpose(self.weights_ho), first_layer) + self.bias_ho
        output = self.acf(weighted_input_output_layer)
        
        print ("Output is: ", output, "\n")
        
        delta_output = np.multiply(compute_errors_prime(output, self.targets), sigmoid_prime(weighted_input_output_layer))
        delta_hidden = np.dot(np.dot(self.weights_ho, delta_output), sigmoid_prime(weighted_input_hidden_layer))
        
        self.weights_ho = self.weights_ho - np.multiply(self.lr, delta_output)
        self.weights_ih = self.weights_ih - np.multiply(self.lr, delta_hidden)
        
#        print ("After update: ", self.weights_ih, "input -> hidden", "\n")
#        print (self.weights_ho, "hidden -> output", "\n")
        
def act_f(activation_function):
    if activation_function == 'tanh':
        return (np.tanh)
    elif activation_function == 'sigmoid':
        return (sigmoid)
    else:
        print("Sorry, the only activation functions currently supported are tanh and sigmoid")
        print("\n")
        
def sigmoid(x):
  return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    return (sigmoid(x)*(1 - sigmoid(x)))
    
def compute_errors(pred, target):
    return (float(1/2)*np.square(pred - target))
    
def compute_errors_prime(pred, target):
    return (pred - target)
    
nr_of_nodes = 10
act_fn = 'tanh'
learning_rate = 0.3
targets = np.random.rand(1)

x = NeuralNet(nr_of_nodes, act_fn, learning_rate, targets)

x.init_weights(10,3,1)

data = np.random.random(10)

i = 0

while i < 100:
    x.feed_forward(data)
    i += 1
    