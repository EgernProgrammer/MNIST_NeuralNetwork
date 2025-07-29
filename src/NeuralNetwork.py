import numpy as np

        
class NeuralNetwork:
    def __init__(self):
        self.net_arch = [784, 100, 10]
        self.learning_rate = 0.01
        self.dropout_rate = 0.4
        self.weights = []
        
        # He initialization for weights
        for i in range(len(self.net_arch) - 1):
            w = np.random.randn(self.net_arch[i + 1], self.net_arch[i] + 1) * np.sqrt(2/self.net_arch[i]) # One extra column for bias
            self.weights.append(w)
    
    def update_learning_rate(self, epoch):
        if epoch < 5:
            pass
        else:
            self.learning_rate = 0.01 * (0.95 ** epoch)
            
    def ReLU(self, x):
        return x * (x > 0)
    
    def ReLUderiv(self, x):
        return 1. * (x > 0)
    
    def softmax(self, z):
        shiftz = z - np.max(z, axis=0, keepdims=True) # Shift z for handling overflow
        exp = np.exp(shiftz)
        return exp / np.sum(exp, axis=0, keepdims=True)
    
    def forward_prop(self, input_data, training=True):
        # Add bias to input_data - 784xbatch_size -> 785xbatch_size
        bias = np.ones((1, input_data.shape[1]))
        data = np.concatenate((bias, input_data), axis=0)
       
    ##### Hidden layer #####
        z0 = np.empty((self.net_arch[1], data.shape[1])) # z0 size = 100xbatch_size
        a0 = np.empty((self.net_arch[1], data.shape[1])) # a0 size = 100xbatch_size
        
        # Compute the weighted sum and apply ReLU activation function
        for i in range(data.shape[1]):
            z0[:, i] = self.weights[0] @ data[:, i]
            a0[:, i] = self.ReLU(z0[:, i])
         
        # Only droput in training
        if training:   
            # *a0.shape unpacks shape -> np.random.rand(*a0.shape) creates
            # a 100xbatch_size array of random numbers between (0, 1)
            # if the numbers is less than self.dropout_rate - set input to 0, otherwise set it to 1
            mask = np.random.rand(*a0.shape) > self.dropout_rate
            a0 *= mask
        
        # Add bias to a0
        bias = np.ones((1, a0.shape[1]))
        a0 = np.concatenate((bias, a0))
            
    
    ##### Output layer #####
        # Init empty arrays
        z1 = np.empty((self.net_arch[2], data.shape[1])) # z1 size = 10xbatch_size
        a1 = np.empty((self.net_arch[2], data.shape[1])) # a1 size = 10xbatch_size
        
        # Compute the weighted sum and softmax activation function
        for i in range(data.shape[1]):      
              z1[:, i] = self.weights[1] @ a0[:, i]
              a1[:, i] = self.softmax(z1[:, i])
              
        return z0, a0, z1, a1
    
    def back_prop(self, z0, a0, z1, a1, x, y):
        # x (784xbatch_size)
        # z0 (100xbatch_size)
        # a0 (101xbatch_size)
        # w0 (100x785)
        # w1 (10x101)
        
        # Compute delta1 (10xbatch_size)
        delta1 = a1 - y
        # Compute delta0 (100xbatch_size)
        delta0 = (self.weights[1][:, 1:].T @ delta1) * self.ReLUderiv(z0)
        
        # Update weights (Exlcuding 1st row - biases)
        self.weights[0][:, 1:] -= self.learning_rate * (delta0 @ x.T)
        self.weights[1][:, 1:] -= self.learning_rate * (delta1 @ a0[1:, :].T)
        
        # Compute bias grads (the sum of the grad)
        bias_grad_0 = np.sum(delta0, axis=1) # 100x1
        bias_grad_1 = np.sum(delta1, axis=1) # 10x1
        
        # Update biases   
        self.weights[0][:, 0] -= self.learning_rate * bias_grad_0
        self.weights[1][:, 0] -= self.learning_rate * bias_grad_1
        
    def fit(self, data, labels, epochs, batch_size=32):
        for k in range(epochs):
            if k % 10 == 0:
                print(f"{(k/epochs)*100:.2f}%")
                
            # Shuffle the indices randomly
            perm = np.random.permutation(data.shape[1])
            
            # Update the batch for the random shuffle
            shuffled_data = data[:, perm]
            shuffled_labels = labels[:, perm]
            
            # Update learning_rate
            self.update_learning_rate(k)
            
            # Batch
            for i in range(0, data.shape[1], batch_size):
                # Load batch
                batch_data = shuffled_data[:, i:i+batch_size]
                batch_labels = shuffled_labels[:, i:i+batch_size]
                
                # Forward and back prop
                z0, a0, z1, a1 = self.forward_prop(batch_data)
                self.back_prop(z0, a0, z1, a1, batch_data, batch_labels)
                
            
    def compute_prediction_accuracy(self, data, labels):
        # Compute output a1
        _, _, _, a1 = self.forward_prop(data, False)
        
        # Convert vector outputs to digit, e.g. [0, 1, ..., 0] = 1
        predicted_classes = np.argmax(a1, axis=0)
        
        # Init true classes
        true_classes = labels
        
        # Compute number of correct predictions
        correct = np.sum(predicted_classes == true_classes)
        
        # Calculate percentage of correct predicitons
        accuracy = (correct / labels.shape[0]) * 100
        return accuracy
                
            
    
