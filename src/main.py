import idx2numpy
from src.NeuralNetwork import NeuralNetwork
from src.data_formatting import *

if __name__ == "__main__":
    # Import training datqa
    training_data = idx2numpy.convert_from_file('data/train-images.idx3-ubyte')
    training_labels = idx2numpy.convert_from_file('data/train-labels.idx1-ubyte')
    
    # Import test data
    test_data = idx2numpy.convert_from_file('data/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('data/t10k-labels.idx1-ubyte')
    

    # Format data
    training_data = reshape_data(training_data)
    training_y = reshape_labels(training_labels)
    test_data = reshape_data(test_data)
   
    # Init NN and train with training data/labels
    nn = NeuralNetwork()
    nn.fit(training_data, training_y, 100)
    
    # Test accuracy
    accuracy = nn.compute_prediction_accuracy(test_data, test_labels)
    print(f"Accuracy: {accuracy:.2f}%")
