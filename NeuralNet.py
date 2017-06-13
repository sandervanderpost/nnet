# -*- coding: utf-8 -*-
import numpy as np

# Neural network implementation

class NeuralNet:
    
    def __init__(self, 
                 activation_function,
                 learning_rate,
                 targets,
                 batch_size):
        self.acf = act_f(activation_function)
        self.acf_deriv = act_deriv(activation_function)
        self.lr = learning_rate
        self.targets = targets
        self.batch_size = batch_size
        
    # Initialize starting weights
    def init_weights(self,
                     nodes_input,
                     nodes_hidden,
                     nodes_output):
        self.weights_ih = np.random.rand(nodes_input, nodes_hidden)
        self.weights_ho = np.random.rand(nodes_hidden, nodes_output)
        self.bias_ih = np.random.rand(nodes_hidden)
        self.bias_ho = np.random.rand(nodes_output)
            
    # Feed forward step (with backpropagation)
    def feed_forward(self,
                     data):
        
        idx = np.random.randint(data.shape[0], size=self.batch_size)
        batch_data = data[idx,:]
        targets = self.targets[idx]    

        for batch_index in range(batch_size): 
            weighted_input_hidden_layer = np.dot(self.weights_ih.transpose(), batch_data[batch_index,:]) + self.bias_ih
            first_layer = self.acf(weighted_input_hidden_layer)
        
            weighted_input_output_layer = np.dot(self.weights_ho.transpose(), first_layer) + self.bias_ho
            output = self.acf(weighted_input_output_layer)
            
            delta_output = np.multiply(compute_errors_prime(output, targets[batch_index]), self.acf_deriv(weighted_input_output_layer))
            delta_hidden = np.multiply(np.dot(self.weights_ho, delta_output), self.acf_deriv(weighted_input_hidden_layer))
            
            self.weights_ho -= self.lr*np.multiply(first_layer, delta_output).reshape((-1,1))
            self.weights_ih -= self.lr*np.multiply(batch_data[batch_index,:].reshape((-1,1)), delta_hidden)
            
            self.bias_ho -= self.lr*delta_output
            self.bias_ih -= self.lr*delta_hidden
            
            
    def predict(self, data):
            weighted_input_hidden_layer = np.dot(self.weights_ih.transpose(), data) + self.bias_ih
            first_layer = self.acf(weighted_input_hidden_layer)
        
            weighted_input_output_layer = np.dot(self.weights_ho.transpose(), first_layer) + self.bias_ho
            prediction = self.acf(weighted_input_output_layer)
            
            return prediction
        

        
def act_f(activation_function):
    if activation_function == 'tanh':
        return (np.tanh)
    elif activation_function == 'sigmoid':
        return (sigmoid)
    else:
        raise NameError("Invalid activation function")
        
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
      
def act_deriv(activation_function):
    if activation_function == 'tanh':
        return (1 - np.square(np.tanh))
    elif activation_function == 'sigmoid':
        return (sigmoid_prime)
    else:
        raise NameError("Invalid activation function")

def sigmoid_prime(x):
    return (sigmoid(x)*(1 - sigmoid(x)))
    
def compute_errors_prime(pred, target):
    return (pred - target)
    
act_fn = 'sigmoid'
learning_rate = 0.1
targets = np.array([0, 1, 1, 0])
batch_size = 1

x = NeuralNet(act_fn, learning_rate, targets, batch_size)

x.init_weights(2,2,1)

data = np.array([(1,1), (0,1), (1,0), (0,0)])
i = 0

while i < 100000:
    x.feed_forward(data)
    i += 1
    
print(x.predict(np.array([0,0])))
print(x.predict(np.array([0,1])))   
print(x.predict(np.array([1,0])))
print(x.predict(np.array([1,1]))) 