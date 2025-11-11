# Using lecture notes

# A python implementation of a simple feedforward neural network
# with backpropagation for training, applied to the MNIST dataset.
# The code is based on Michael Nielsen's "Neural Networks and Deep Learning" book
# (see http://neuralnetworksanddeeplearning.com/).

import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
import random


def sigmoid(z):
    """The sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)


def feedforward(a_1, weights, biases):
    """
    Compute the network output for a given input.

    a_1: input vector
    weights: list of weight matrices for each layer
    biases: list of bias vectors for each layer
    Returns: output vector
    """
    a = a_1.copy()
    for w, b, in zip(weights, biases):
        a = sigmoid(w @ a + b)
    return a


def train_network(training_data, epochs, minibatch_size, eta, weights, biases, seed=42, log_every=1):
    """
    Train the neural network using stochastic gradient descent.

    training_data: list of (input, target) tuples
    epochs: number of epochs
    minibatch_size: size of each minibatch
    eta: learning rate
    weights: list of weight matrices for each layer
    biases: list of bias vectors for each layer
    seed: random seed for shuffling data
    log_every: frequency of logging progress (in epochs)
    Returns: updated weights and biases
    """
    rng = random.Random(seed)
    n = len(training_data)

    for epoch in range(epochs):
        # Shuffle training data each epoch
        training_data = list(training_data)
        rng.shuffle(training_data)

        # Update parameters for each minibatch
        for start in range(0, n, minibatch_size):
            # Get minibatch
            batch = training_data[start:start + minibatch_size]

            # Initialize gradients
            grad_b = [np.zeros_like(b) for b in biases]
            grad_w = [np.zeros_like(w) for w in weights]
            for x, y in batch:
                # Compute gradient for this example
                dw, db = backprop(x, y, weights, biases)

                # Accumulate gradients
                grad_b = [gb + db_i for gb, db_i in zip(grad_b, db)]
                grad_w = [gw + dw_i for gw, dw_i in zip(grad_w, dw)]

            # Update weights and biases (gradient descent step)
            scale = eta / len(batch)
            weights = [w - scale * gw for w, gw in zip(weights, grad_w)]
            biases = [b - scale * gb for b, gb in zip(biases, grad_b)]

        # Periodic logging
        is_last = (epoch == epochs - 1)
        if log_every and (epoch % log_every == 0 or is_last):
            mse = np.mean([
                np.square(feedforward(x, weights, biases) - y)
                for x, y in training_data
            ])
            print(f"Epoch {epoch + 1}/{epochs}: MSE = {mse:.6f}")
    return weights, biases


def backprop(x, y, weights, biases):
    """
    Backpropagation algorithm to compute gradients of the cost function
    with respect to weights and biases for a single training example (x, y).

    x: input vector
    y: target output vector
    weights: list of weight matrices for each layer
    biases: list of bias vectors for each layer
    Returns: (grad_w, grad_b) where grad_w and grad_b are lists of gradients
    with respect to the weights and biases.
    """
    L = len(weights)

    # Feedforward pass to compute activations and pre-activations
    activations = [x]
    preacts = []
    a = x
    for w, b in zip(weights, biases):
        z = w @ a + b
        preacts.append(z)
        a = sigmoid(z)
        activations.append(a)

    # Backward pass to compute gradients
    grad_w = [np.zeros_like(w) for w in weights]
    grad_b = [np.zeros_like(b) for b in biases]

    delta = (activations[-1] - y) * sigmoid_prime(preacts[-1])
    grad_b[-1] = delta
    grad_w[-1] = np.outer(delta, activations[-2])

    for l in range(L - 2, -1, -1):
        delta = (weights[l + 1].T @ delta) * sigmoid_prime(preacts[l])
        grad_b[l] = delta
        grad_w[l] = np.outer(delta, activations[l])

    return grad_w, grad_b


if __name__ == "__main__":
    def create_y_from_int_label(int_label):
        """Helper function to convert integer label to encoded vector
        e.g. 3 -> [0,0,0,1,0,0,0,0,0,0]"""
        y = np.zeros(10)
        y[int_label] = 1.0
        return y

    def show_mnist_image(image_data):
        """
        Visualise an MNIST image.
        """
        image = (1.0 - image_data).reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.show()

    # Load MNIST dataset
    print("Loading MNIST...")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Flatten and normalize
    train_x = train_x.reshape(-1, 784) / 255.0
    test_x = test_x.reshape(-1, 784) / 255.0

    # Prepare data
    training_data = [(train_x[i], create_y_from_int_label(train_y[i]))
                     for i in range(len(train_x))]
    test_data = list(zip(test_x, test_y))

    # Define network architecture
    sizes = [784, 16, 16, 10]

    # Initialize weights and biases randomly
    weights = [np.random.randn(n_1, n_2) / np.sqrt(n_2)
               for n_1, n_2 in zip(sizes[1:], sizes[:-1])]
    biases = [np.random.randn(n) for n in sizes[1:]]

    # Train network
    print("Training network...")
    weights, biases = train_network(training_data,
                                    epochs=10,
                                    minibatch_size=50,
                                    eta=3.0,
                                    weights=weights,
                                    biases=biases)

    # Evaluate on test data
    print("\nEvaluating on test data...")
    num_correct = 0
    for x, y_true in test_data:
        output = feedforward(x, weights, biases)
        predicted_label = np.argmax(output)
        if predicted_label == y_true:
            num_correct += 1

    accuracy = num_correct / len(test_data)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # Show an example test image and prediction
    image_index = 0
    image_data = test_x[image_index]
    true_label = test_y[image_index]
    output = feedforward(image_data, weights, biases)
    predicted_label = np.argmax(output)
    print(
        f"\nExample test image - True label: {true_label}, Predicted label: {predicted_label}")
    show_mnist_image(image_data)
