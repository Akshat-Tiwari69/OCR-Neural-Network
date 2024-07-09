import numpy as np

class OCRNeuralNetwork:
    def __init__(self, hidden_nodes, data_matrix, data_labels, train_indices, initialize=True):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = data_matrix.shape[1]
        self.output_nodes = 10  # Digits 0-9

        self.weights_input_hidden = np.random.randn(self.input_nodes, self.hidden_nodes)
        self.weights_hidden_output = np.random.randn(self.hidden_nodes, self.output_nodes)
        self.bias_hidden = np.random.randn(self.hidden_nodes)
        self.bias_output = np.random.randn(self.output_nodes)

        if initialize:
            self.train(data_matrix[train_indices], data_labels[train_indices])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, data, labels):
        for _ in range(1000):  # Training iterations
            for x, y in zip(data, labels):
                hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
                hidden_output = self.sigmoid(hidden_input)

                final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                final_output = self.sigmoid(final_input)

                output_error = y - final_output
                output_delta = output_error * self.sigmoid_derivative(final_output)

                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                hidden_delta = hidden_error * self.sigmoid_derivative(hidden_output)

                self.weights_hidden_output += hidden_output.reshape(-1, 1).dot(output_delta.reshape(1, -1))
                self.weights_input_hidden += x.reshape(-1, 1).dot(hidden_delta.reshape(1, -1))
                self.bias_hidden += hidden_delta
                self.bias_output += output_delta

    def predict(self, data):
        hidden_input = np.dot(data, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)

        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(final_input)

        return np.argmax(final_output)

def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    for _ in range(100):
        correct_guess_count = 0
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            if data_labels[i] == prediction:
                correct_guess_count += 1

        avg_sum += (correct_guess_count / float(len(test_indices)))
    return avg_sum / 100

# Example usage
data_matrix = np.random.rand(1000, 400)  # Dummy data
data_labels = np.random.randint(0, 10, 1000)  # Dummy labels
train_indices = np.random.choice(1000, 800, replace=False)
test_indices = np.setdiff1d(np.arange(1000), train_indices)

# Try various number of hidden nodes and see what performs best
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print("{i} Hidden Nodes: {val}".format(i=i, val=performance))
