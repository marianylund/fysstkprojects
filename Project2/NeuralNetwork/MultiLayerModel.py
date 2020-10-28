import numpy as np
import typing

class MultiLayerModel():
    def __init__(self,  neurons_per_layer: typing.List[int], input_nodes:int, l2_reg_lambda: float = 1.0, linear = True):
        self.I = input_nodes
        self.neurons_per_layer = neurons_per_layer
        self.num_of_layers = len(neurons_per_layer)
        self.linear = linear # false if classification
        self.l2_reg_lambda = l2_reg_lambda

        # Initialise the weights to randomly sampled
        self.ws = []
        
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            #print("Initializing weight to shape:", w_shape)
            w = self.init_weights(w_shape)
            self.ws.append(w)
            prev = size
        
        self.grads = [None for i in range(len(self.ws))]

        self.zs = [None for i in range(len(self.ws))]
        self.activations = [None for i in range(len(self.ws))]

    def init_weights(self, w_shape):
        improved = np.random.uniform(-1, 1, w_shape)
        # if self.use_improved_weight_init:
        #     improved *= np.sqrt(1/w_shape[0])
        return improved

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
            return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def cost_derivative(self, targets, outputs):
        return -(targets - outputs)

    def soft_max(self, x):
        z_exp = np.exp(x)
        part = np.sum(z_exp, axis = 1, keepdims = True)
        soft_max_var = z_exp/part
        return soft_max_var

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.activations[0] = X

        # For all layers except the last one
        for layer in range(self.num_of_layers - 1):
            # print("Layer: " + str(layer))
            self.zs[layer] = np.dot(self.activations[layer], self.ws[layer])
            self.activations[layer + 1] = self.sigmoid(self.zs[layer])
        
        # here use soft-max for the last layer if classification, but just identity if linear:
        last_layer = self.num_of_layers - 1
        if self.linear:
            return self.activations[last_layer]
        
        self.zs[last_layer] = np.dot(self.activations[last_layer], self.ws[last_layer])
        y_k = self.soft_max(self.zs[last_layer])

        return y_k
