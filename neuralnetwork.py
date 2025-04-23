
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split





# Class for Neural Netwok

# Input -> 10 x 1
# Weights -> 10 x 3 
# Bias -> 3


class NeuralNetwork:
    def __init__(self, in_size=13, hidden_size = 3, out_size = 1, learning_rate=0.001):
        
        self.learning_rate = learning_rate
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
        self.A1 = self.ReLu(self.Z1)
        
        self.Z2 = np.dot(self.w2, self.A1) + self.b2
        self.A2 = self.Sigmoid(self.Z2)

        return self.A2
    
    def backwords(self, X, Y):
        
       dZ2 = self.A2 - Y
       dW2 = np.dot(dZ2, self.A1.T)
       db2 = np.sum(dZ2, axis=1, keepdims=True)
       
       dZ1 = np.dot(self.w2.T, dZ2) * self.ReLu_derivative(self.Z1)
       dW1 = np.dot(dZ1, X.T)
       db1 = np.sum(dZ1, axis=1, keepdims=True)
       
       self.update(dW1, db1, dW2, db2)
       
       return 

    def update(self, dW1, db1, dW2, db2):
        self.w1 = self.w1 - self.learning_rate * (dW1)
        self.b1 = self.b1 - self.learning_rate * (db1)

        self.w2 = self.w2 - self.learning_rate * (dW2)
        self.b2 = self.b2 - self.learning_rate * (db2)
        return
   
    def get_accuracy(self, predictions, Y):
        preds_binary = (predictions > 0.5).astype(int)
        return np.sum(preds_binary == Y) / Y.size

    
    def compute_loss(self, Y):
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(self.A2 + 1e-8) + (1 - Y) * np.log(1 - self.A2 + 1e-8)) / m
        return loss

    def ReLu(self, Z):
        return np.maximum(0,Z)
    
    def Sigmoid(self, Z):
        return 1 / (1 + np.exp( -Z))
    
    def ReLu_derivative(self, Z):
        return (Z > 0)

    def Sigmoid_derivative(self, Z):
        sig = self.Sigmoid(Z)
        return sig * (1 - sig)

    def predict(self, X):
        A2 = self.forward(X)
        return (A2 > 0.5).astype(int)

    def gradient_descent(self, epochs, X, Y):
        ...



# Load and preprocess
df = pd.read_csv("Data.csv")
fire = df.pop("FIRE_START_DAY")
df["Fire"] = fire.astype(int)


for i in df.columns:
    print(df[i].unique())
    






