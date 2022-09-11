from turtle import right
import numpy

# Supported activation functions by the cnn.py module.
supported_activation_functions = ("sigmoid", "relu", "softmax")

def sigmoid(sop):

    """
    Applies the sigmoid function.

    sop: The input to which the sigmoid function is applied.

    Returns the result of the sigmoid function.
    """

    if type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    return 1.0 / (1 + numpy.exp(-1 * sop))

def relu(sop):

    """
    Applies the rectified linear unit (ReLU) function.

    sop: The input to which the relu function is applied.

    Returns the result of the ReLU function.
    """

    if not (type(sop) in [list, tuple, numpy.ndarray]):
        if sop < 0:
            return 0
        else:
            return sop
    elif type(sop) in [list, tuple]:
        sop = numpy.array(sop)

    result = sop
    result[sop < 0] = 0

    return result

def softmax(layer_outputs):

    """
    Applies the sotmax function.

    sop: The input to which the softmax function is applied.

    Returns the result of the softmax function.
    """
    return numpy.exp(layer_outputs)/(sum(numpy.exp(layer_outputs)))

class Input:

    """
    Implementing the input layer of a CNN.
    The CNN architecture must start with an input layer.
    """

    def __init__(self, input_shape):

        """
        input_shape: Shape of the input sample to the CNN.
        """

        self.input_shape = input_shape # Shape of the input sample.
        self.layer_output_size = input_shape # Shape of the output from the current layer. For an input layer, it is the same as the shape of the input sample.


class ReLU:

    """
    Implementing the ReLU layer.
    """

    def __init__(self, previous_layer):

        """
        previous_layer: Reference to the previous layer.
        """

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")

        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        self.layer_output_size = self.previous_layer.layer_output_size

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def relu_layer(self, layer_input):

        """
        Applies the ReLU function over all elements in input to the ReLU layer.
        
        layer_input: The input to which the ReLU function is applied.

        The relu_layer() method saves its result in the layer_output attribute.
        """

        self.layer_output_size = layer_input.size
        self.layer_output = relu(layer_input)

class Sigmoid:

    """
    Implementing the sigmoid layer.
    """

    def __init__(self, previous_layer):

        """
        previous_layer: Reference to the previous layer.
        """

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer

        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        self.layer_output_size = self.previous_layer.layer_output_size

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def sigmoid_layer(self, layer_input):

        """
        Applies the sigmoid function over all elements in input to the sigmoid layer.
        
        layer_input: The input to which the sigmoid function is applied.

        The sigmoid_layer() method saves its result in the layer_output attribute.
        """

        self.layer_output_size = layer_input.size
        self.layer_output = sigmoid(layer_input)

class Dense:

    """
    Implementing the input dense (fully connected) layer of a NN.
    """

    def __init__(self, num_neurons, previous_layer, activation_function="relu", weight=None, bias=None):

        """
        num_neurons: Number of neurons in the dense layer.
        previous_layer: Reference to the previous layer.
        activation_function: Name of the activation function to be used in the current layer.
        """

        if num_neurons <= 0:
            raise ValueError("Number of neurons cannot be <= 0. Please pass a valid value to the 'num_neurons' parameter.")

        # Number of neurons in the dense layer.
        self.num_neurons = num_neurons

        # Validating the activation function
        if (activation_function == "relu"):
            self.activation = relu
        elif (activation_function == "sigmoid"):
            self.activation = sigmoid
        elif (activation_function == "softmax"):
            self.activation = softmax
        else:
            raise ValueError("The specified activation function '{activation_function}' is not among the supported activation functions {supported_activation_functions}. Please use one of the supported functions.".format(activation_function=activation_function, supported_activation_functions=supported_activation_functions))

        self.activation_function = activation_function

        if previous_layer is None:
            raise TypeError("The previous layer cannot be of Type 'None'. Please pass a valid layer to the 'previous_layer' parameter.")
        # A reference to the layer that preceeds the current layer in the network architecture.
        self.previous_layer = previous_layer
        
        if type(self.previous_layer.layer_output_size) in [list, tuple, numpy.ndarray] and len(self.previous_layer.layer_output_size) > 1:
            raise ValueError("The input to the dense layer must be of type int but {sh} found.".format(sh=type(self.previous_layer.layer_output_size)))
        # Initializing the weights of the layer.
        
        if type(weight) == None:
            self.initial_weights = numpy.random.uniform(low=-0.1,
                                                        high=0.1,
                                                        size=(self.num_neurons, self.previous_layer.layer_output_size)
                                                        )
        else:
            self.initial_weights = weight

        # Initializing the biases of the layer.
        if type(bias) == None:
            self.initial_biases = numpy.random.uniform(low=-0.1,
                                                        high=0.1,
                                                        size=self.num_neurons)
        else:
            self.initial_biases = bias

        # The trained weights of the layer. Only assigned a value after the network is trained (i.e. the train_network() function completes).
        # Just initialized to be equal to the initial weights
        self.trained_weights = weight
        self.trained_biases = bias

        # Size of the input to the layer.
        self.layer_input_size = self.previous_layer.layer_output_size

        # Size of the output from the layer.
        self.layer_output_size = num_neurons

        # The layer_output attribute holds the latest output from the layer.
        self.layer_output = None

    def dense_layer(self, layer_input):

        """
        Calculates the output of the dense layer.
        
        layer_input: The input to the dense layer

        The dense_layer() method saves its result in the layer_output attribute.
        """

        if self.trained_weights is None:
            raise TypeError("The weights of the dense layer cannot be of Type 'None'.")

        sop = numpy.matmul(self.trained_weights, layer_input) + self.trained_biases

        self.layer_output = self.activation(sop)


class Sequential():
    def __init__(self):
        self.network_layers  = []
    
    def add(self,layer):
        self.network_layers.append(layer)
        return layer
    
    def summary(self):

        """
        Prints a summary of the NN architecture.
        """

        print("\n")
        for i in range(15):
            print('*',end='')
        print("Network Architecture", end='')
        for i in range(15):
            print('*', end='')
        print('\n')

        for i in range(45):
            print('-', end='')
        print('\n')

        print(f"Layer type      No. of neurons")

        for i in range(45):
            print('-', end='')
        print('\n')
        print(f"Input_layer0       {self.network_layers[0].layer_input_size}")
        print('\n')
        for i,layer in enumerate(self.network_layers):
            if i>=(len(self.network_layers)-1) and type(layer) is Dense:
                print(f"Output_layer{i+1}", end="")
                print("       ", end="")
                print(layer.num_neurons)
            else:
                print(f"Dense_layer{i+1}", end="")
                print("        ", end="")
                print(layer.num_neurons)
            print("\n")

        for i in range(45):
            print('*', end='')
        print('\n')

    def feed_forward(self, sample):
        
        """
        Feeds a sample in the NN layers.
        
        sample: The samples to be fed to the NN layers.
        
        Returns results of the last layer in the NN.
        """

        last_layer_outputs = sample
        for layer in self.network_layers:
            if type(layer) is Dense:
                layer.dense_layer(layer_input=last_layer_outputs)
                last_layer_outputs = layer.layer_output
            elif type(layer) is ReLU:
                layer.relu_layer(layer_input=last_layer_outputs)
                last_layer_outputs = layer.layer_output
            elif type(layer) is Sigmoid:
                layer.sigmoid_layer(layer_input=last_layer_outputs)
                last_layer_outputs = layer.layer_output
            elif type(layer) is Input:
                pass
            else:
                print("Other")
                raise TypeError("The layer of type {layer_type} is not supported yet.".format(layer_type=type(layer)))
        return self.network_layers[-1].layer_output

