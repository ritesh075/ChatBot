from statistics import mode
import numpy as np
from myNeuron import *

num_classes = 7
num_features = 54
num_hidden = 8

l1_weight = np.load("./params/l1-weight.npy")
l1_bias = np.load("./params/l1-bias.npy")

l2_weight = np.load("./params/l2-weight.npy")
l2_bias = np.load("./params/l2-bias.npy")

l3_weight = np.load("./params/l3-weight.npy")
l3_bias = np.load("./params/l3-bias.npy")

model = Sequential()
input = Input(input_shape=num_features)
hl1 = model.add(Dense(num_neurons=num_hidden, previous_layer=input, weight=l1_weight, bias=l1_bias)) # hidden layer1
hl2 = model.add(Dense(num_neurons=num_hidden, previous_layer=hl1, weight=l2_weight, bias=l2_bias)) # hidden layer2
output = model.add(Dense(num_neurons=num_classes, previous_layer=hl2, weight=l3_weight, bias=l3_bias)) # output layer

if __name__ == "__main__":
    model.summary()