import nn_classes as nn
import numpy as np


def test_layer_weights_nodes():
    layer = nn.Layer(2, 2)
    assert layer.nodes == 2


def test_layer_weights_weight_shape():
    layer = nn.Layer(2, 2)
    assert layer.weights.shape == (2, 2)


def test_layer_weights_bias_shape():
    layer = nn.Layer(2, 2)
    assert layer.bias.shape == (1, 2)


def test_layer_Z_function():
    layer = nn.Layer(2, 2)
    layer.weights = np.array([[0, 1], [-1, 0]])
    layer.bias = np.array([[1, 1]])
    X = np.array([[1, 1]])
    Z = layer.Z_function(X)
    assert (Z == np.array([[0, 2]])).all()


def test_layer_activation_relu():
    layer = nn.Layer(2, 2, activation="relu")
    layer.weights = np.array([[0, 1], [-2, 0]])
    layer.bias = np.array([[1, 1]])
    X = np.array([[1, 1]])
    A = layer.A_function(X)
    assert (A == np.array([[0, 2]])).all()


def test_layer_relu():
    layer = nn.Layer(2, 2, activation="relu")
    z = np.array([[0, 1], [-2, 0]])
    relu_z = layer.relu(z)
    assert (relu_z == np.array([[0, 1], [0, 0]])).all()


def test_layer_d_relu():
    layer = nn.Layer(2, 2, activation="relu")
    z = np.array([[0, 20], [-2, 0]])
    relu_z = layer.d_relu(z)
    assert (relu_z == np.array([[0, 1], [0, 0]])).all()
