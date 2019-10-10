import random
import json
import numpy as np
from termcolor import colored

with open('numbers.json') as f:
    data = json.loads(f.read())

class CNN():
   def __init__(self, name, w=10, h=10):
       self.t_p = w*h
       np.random.seed(1)
       self.weights = 2 * np.random.random((self.t_p)) - 1
       self.obj_name = name

   def GetName(self):
       return self.obj_name

   def sigmoid(self, x):
       # applying the sigmoid function
       return 1 / (1 + np.exp(-x))

   def sigmoid_derivative(self,x):
       # computing derivative to the Sigmoid function
       return x * (1 - x)

   def Predict(self, inputs):
      out = np.dot(inputs, self.weights)
      return round(self.sigmoid(out), 3)

   def Perceptron(self,training_inputs, output):
       outputP = self.Predict(training_inputs)
       error = output - outputP
       for i in range(self.t_p):
           self.weights[i] += np.dot(training_inputs[i], error*self.sigmoid_derivative(outputP))
           self.weights[i] = round(self.weights[i], 2)

   def Train(self,all_inputs, all_outputs, iterations):
       for i in range(iterations):
           j = 0
           for inputs in all_inputs:
                self.Perceptron(inputs, all_outputs[j])
                j+=1

Numbers = []
for i in range(8):
    Obj = CNN(str(i), 10, 10)
    Obj.Train(data["supervised"][str(i)]["inputs"], data["supervised"][str(i)]["outputs"], 10)
    Numbers.append(Obj)

def PrintNum(inputs):
    j = 0
    for i in inputs:
        if j==10:
            print('')
            j=0
        if i>0:
           print(colored(i, 'blue'), end=' ')
        else:
           print(' ', end=' ')
        j+=1
    print('')

def PredictNumber(inputs):
    results = []
    for i in range(8):
        results.append(Numbers[i].Predict(inputs))
    maxi = max(results)
    index = 0
    PrintNum(inputs)
    print(results)
    for i in results:
        if i == maxi:
            print("The given number is a",index,"with a probability of",colored(str(round(maxi*100,2))+'%', 'red'))
        index+=1

for d in data["unsupervised"]:
    PredictNumber(d)