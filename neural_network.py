from utils import sigmoid, derivative_sigmoid
import numpy as np

class NeuralNetwork:

    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_output_neurons):

        self.input_hidden_weights = np.random.rand(number_of_input_neurons, number_of_hidden_neurons)
        self.hidden_output_weights = np.random.rand(number_of_hidden_neurons, number_of_output_neurons)
    
    def train(self, X, y, epochs):

        for _ in range(epochs):

            hidden_output = sigmoid(np.dot(X, self.input_hidden_weights))
            output_output = sigmoid(np.dot(hidden_output, self.hidden_output_weights))

            output_error = y - output_output
            self.hidden_output_weights += np.dot(hidden_output.T, output_error*derivative_sigmoid(output_output))

            hidden_error = np.dot(output_error*derivative_sigmoid(output_output), self.hidden_output_weights.T)
            self.input_hidden_weights += np.dot(X.T, hidden_error*derivative_sigmoid(hidden_output))

    def predict(self, X):

        hidden_output = sigmoid(np.dot(X, self.input_hidden_weights))
        output_output = sigmoid(np.dot(hidden_output, self.hidden_output_weights))

        if output_output >= 0.75: return 1

        return 0