import numpy as np

def reshape_data(data):
    # Reshape all 60.000 images - 28x28 -> 784x1
    # resulting in new_data = 784x60.000
    # Normalize data
    new_data = data.reshape(data.shape[0], -1).T / 255
    return new_data

def reshape_labels(labels):
    # Reshape labels from integer (0, 9) to one-hot vectors
    new_labels = np.zeros((10, labels.shape[0]))
    for i in range(labels.shape[0]):
        new_labels[labels[i], i] = 1
    return new_labels
