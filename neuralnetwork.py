
import pandas as pd
import numpy as np


# Class for Neural Netwok

# Input -> 10 x 1
# Weights -> 10 x 3 
# Bias -> 3

class NeuralNetwork:
    def __init__(self, in_size, hidden_size, out_size):
        self.w1 =  np.random.rand(hidden_size, in_size) 
        self.b1 =  np.random.rand(hidden_size, 1) 
        

        self.w2 =  np.random.rand(out_size, hidden_size) 
        self.b2 =  np.random.rand(out_size, 1) 
        
        self.Z1 = None
        self.A1 = None

        self.Z2 = None
        self.A2 = None

        return 
    
    def forward(self, input):
        
        self.Z1 = np.dot(self.w1, input) + self.b1
        self.A1 = self.Sigmoid(self.Z1)
        
        self.Z2 = np.dot(self.w2, self.A1) + self.b2
        self.A2 = self.ReLu(self.Z2)

        return
    
    def backwords(self, Y):
        ...

    def ReLu(self, Z):
        return np.maximum(0,Z)
    
    def Sigmoid(self, Z):
        return 1 / (1 + np.exp( -Z))
    
    def ReLu_derivative(self, Z):
        return (Z > 0).astype(float)

    def Sigmoid_derivative(self, Z):
        sig = self.Sigmoid(Z)
        return sig * (1 - sig)




nn = NeuralNetwork(10,3,1)
input_val = np.random.randn(10, 1) - 0.5

nn.forward(input_val)
print(nn.A2)


