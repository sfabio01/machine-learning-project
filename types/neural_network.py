import numpy as np


class NeuralNetwork:
    def __init__(self, layers: list):
        # at least an input and an output layer
        assert len(layers) >= 2
        self.layers = layers
        # array of input neurons (first layer)
        self.input = np.empty(shape=layers[0])
        # array of hidden layers
        self.hiddens = []
        for i in range(1, len(layers)-1):
            self.hiddens.append(np.empty(shape=layers[i]))
        # array of output neurons (last layer)
        self.output = np.empty(shape=layers[-1])
        # array of weight matrices
        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(np.empty(shape=(layers[i+1], layers[i])))
        # array of biases
        self.biases = []
        for i in range(1, len(layers)):
            self.biases.append(np.empty(shape=layers[i]))

    def init_params_rand(self):
        for i in range(len(self.weights)):
            self.weights[i] = np.random.rand(
                self.weights[i].shape[0], self.weights[i].shape[1])
        for i in range(len(self.biases)):
            self.biases[i] = np.zeros(shape=self.biases[i].shape)

    def __str__(self) -> str:
        text = f"NeuralNetwork object\n"
        text += f"Layers: {self.layers}\n"
        text += "Shape of weight matrices: ["
        for mat in self.weights:
            text += f"{mat.shape},"
        text = text[:-1]
        text += "]\n"
        return text
