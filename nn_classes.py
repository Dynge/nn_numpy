import numpy as np
import pandas as pd
import logging


class Layer:
    def __init__(self, input_size, nodes, activation="relu"):
        self.nodes = nodes
        # logging.warning("Initialized with ones. Change to random when working")
        self.weights = np.random.rand(nodes, input_size) * np.sqrt(1 / nodes)
        self.bias = np.random.rand(nodes, 1)
        self.activation = activation

    def relu(self, z):
        return z * (z > 0)

    def d_relu(self, dA, z):
        dZ = dA.copy()
        dZ[z <= 0] = 0
        return dZ

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, dA, z):
        sig = self.sigmoid(z)
        return dA * sig * (1 - sig)

    def forward(self, X):
        logging.debug([self.weights.shape, X.shape, self.bias.shape])
        self.Z = np.matmul(self.weights, X) + self.bias

        if self.activation == "relu":
            activation = self.relu
        elif self.activation == "sigmoid":
            activation = self.sigmoid
        else:
            raise Exception("Non-supported activation function")

        self.output = activation(self.Z)

        return self.output, self.Z

    def backward(self, dA_current, A_prev):
        m = A_prev.shape[1]

        if self.activation == "relu":
            d_activation = self.d_relu
        elif self.activation == "sigmoid":
            d_activation = self.d_sigmoid
        else:
            raise Exception("Non-supported activation function")

        dZ_current = d_activation(dA_current, self.Z)
        logging.debug("dZ {}".format(dZ_current.shape))
        logging.debug("A_prev {}".format(A_prev.T.shape))
        dW_current = np.dot(dZ_current, A_prev.T) / m
        db_current = np.sum(dZ_current, axis=1, keepdims=True) / m
        dA_previous = np.dot(self.weights.T, dZ_current)

        return dA_previous, dW_current, db_current


class DenseNet:
    def __init__(self, input_size, architecture):
        self.input_size = input_size
        self.add_layers(architecture)

    def add_layers(self, architecture):
        self.layers = []
        for layer_dict in architecture:
            if len(self.layers) == 0:
                input_size = self.input_size
            else:
                input_size = self.layers[-1].nodes
            self.layers.append(
                Layer(
                    input_size, layer_dict["nodes"], activation=layer_dict["activation"]
                )
            )

    def predict(self, X):
        memory = {}
        A_current = X
        for i, layer in enumerate(self.layers):
            logging.debug("layer number: {}".format(i + 1))
            A_previous = A_current

            A_current, Z_current = layer.forward(A_previous)
            memory["A" + str(i)] = A_previous

        return A_current, memory

    def loss(self, Y_hat, Y):
        return np.sum((Y - Y_hat) ** 2)

    def d_loss(self, Y_hat, Y):
        return np.sum(2 * (Y - Y_hat))

    def train(self, X, Y, epochs=10, learning_rate=0.1):
        grads_values = {}
        for epoch in range(epochs):
            logging.debug("Epoch number: {}".format(epoch + 1))

            model_output, memory = self.predict(X)
            m = Y.shape[1]
            Y = Y.reshape(model_output.shape)

            dA_previous = self.d_loss(model_output, Y)

            for i_prev, layer in reversed(list(enumerate(self.layers))):
                i_current = i_prev + 1
                dA_current = dA_previous

                A_previous = memory["A" + str(i_prev)]
                dA_previous, dW_curr, db_curr = layer.backward(dA_current, A_previous)

                grads_values["dW" + str(i_current)] = dW_curr
                grads_values["db" + str(i_current)] = db_curr
            for layer_idx, layer in enumerate(self.layers):
                logging.debug(
                    "Layer {}: {}".format(
                        layer_idx + 1,
                        learning_rate * grads_values["dW" + str(layer_idx + 1)],
                    )
                )
                layer.weights += learning_rate * grads_values["dW" + str(layer_idx + 1)]
                layer.bias += learning_rate * grads_values["db" + str(layer_idx + 1)]

        return grads_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X = np.array([x / 100 for x in range(100)]).reshape(-1, 100)
    Y = X / 2

    nn_architecture = [
        {"nodes": 3, "activation": "relu"},
        {"nodes": 6, "activation": "relu"},
        {"nodes": 3, "activation": "relu"},
        {"nodes": 1, "activation": "sigmoid"},
    ]
    net = DenseNet(1, nn_architecture)
    y_hat, _ = net.predict(X)
    logging.info(net.loss(Y=Y, Y_hat=y_hat))
    logging.info(y_hat)
    net.train(X, Y, epochs=5)
    y_hat, _ = net.predict(X)
    logging.info(net.loss(Y=Y, Y_hat=y_hat))
    logging.info(y_hat)
