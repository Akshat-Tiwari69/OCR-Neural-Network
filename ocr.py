import numpy as np
import json

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    NN_FILE_PATH = 'neural_network.json'

    def __init__(self, num_hidden_nodes, use_file=True):
        self.num_hidden_nodes = num_hidden_nodes
        self._use_file = use_file

        if self._use_file and self._load():
            return

        self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
        self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
        self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
        self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

    def _rand_initialize_weights(self, size_in, size_out):
        return np.random.rand(size_out, size_in) * 0.12 - 0.06

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def train(self, data):
        for sample in data:
            y1 = np.dot(self.theta1, sample['y0']) + self.input_layer_bias
            y1 = self._sigmoid(y1)

            y2 = np.dot(self.theta2, y1) + self.hidden_layer_bias
            y2 = self._sigmoid(y2)

            actual_vals = np.zeros(10)
            actual_vals[sample['label']] = 1

            output_errors = actual_vals - y2
            hidden_errors = np.dot(self.theta2.T, output_errors) * self._sigmoid_prime(y1)

            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors.reshape(-1, 1), sample['y0'].reshape(1, -1))
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors.reshape(-1, 1), y1.reshape(1, -1))
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(self.theta1, test) + self.input_layer_bias
        y1 = self._sigmoid(y1)

        y2 = np.dot(self.theta2, y1) + self.hidden_layer_bias
        y2 = self._sigmoid(y2)

        results = y2.tolist()
        return results.index(max(results))

    def save(self):
        if not self._use_file:
            return

        json_neural_network = {
            "theta1": self.theta1.tolist(),
            "theta2": self.theta2.tolist(),
            "b1": self.input_layer_bias.tolist(),
            "b2": self.hidden_layer_bias.tolist()
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return False

        try:
            with open(OCRNeuralNetwork.NN_FILE_PATH, 'r') as nnFile:
                nn = json.load(nnFile)
            self.theta1 = np.array(nn['theta1'])
            self.theta2 = np.array(nn['theta2'])
            self.input_layer_bias = np.array(nn['b1'])
            self.hidden_layer_bias = np.array(nn['b2'])
            return True
        except FileNotFoundError:
            return False
