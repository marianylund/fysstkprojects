# Inspired from: https://github.com/hukkelas/TDT4265-StarterCode
import numpy as np
import typing
from yacs.config import CfgNode as CN

class MultiLayerModel():
    def __init__(self, cfg:CN, input_nodes:int):
        # neurons_per_layer: typing.List[int], input_nodes:int, l2_reg_lambda: float = 1.0, linear = True
        self.I = input_nodes
        self.neurons_per_layer = cfg.MODEL.SHAPE
        self.activation_functions = cfg.MODEL.ACTIVATION_FUNCTIONS
        self.cost_function = cfg.MODEL.COST_FUNCTION
        self.shape = [input_nodes] + self.neurons_per_layer
        self.num_of_layers = len(cfg.MODEL.SHAPE)
        self.l2_reg_lambda = cfg.OPTIM.L2_REG_LAMBDA
        self.leaky_slope = cfg.MODEL.LEAKY_SLOPE
        self.weight_init = cfg.MODEL.WEIGHT_INIT

        self.check_config()

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

    def check_config(self):
        assert len(self.neurons_per_layer) == len(self.activation_functions), " the number of layers and activations is not equal: " + str(len(self.neurons_per_layer)) + " " + str(len(self.activation_functions))

    def init_weights(self, w_shape):
        """{'random', 'he', 'xavier', 'zeros'}, default: 'random'"""
        #Xavier is the recommended weight initialization method for sigmoid and tanh activation function
        if self.weight_init == "zeros":
            return np.zeros(w_shape)

        weights = np.random.uniform(-1, 1, w_shape)
        if self.weight_init == "random":
            return weights
        elif self.weight_init == "he":
            weights *= np.sqrt(2/w_shape[0])
        elif self.weight_init == "xavier":
            weights *= np.sqrt(1/w_shape[0])
        else:
            print("Did not find weight init type: ", self.weight_init)
        return weights

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
            return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def cost_derivative(self, activation, output_error, N):
        if self.cost_function == "mse":
            return 2 * activation.T @ output_error
        elif self.cost_function == "ce":
            return np.dot(activation.T, output_error) / N
        else:
            raise ValueError(self.cost_function, " not found in cost_derivative")

    def soft_max(self, x):
        z_exp = np.exp(x)
        part = np.sum(z_exp, axis = 1, keepdims = True)
        soft_max_var = z_exp/part
        return soft_max_var

    def cross_entropy_loss(self, targets: np.ndarray, outputs: np.ndarray):
        assert targets.shape == outputs.shape,\
            f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

        # C(w) = −1/N * N∑n=1 K∑k=1 targets * ln(outputs)
        # We need to sum over klasses and batches, but divide just by batches
        ce = - np.sum(targets * (np.log(outputs))) / targets.shape[0]
        return ce

    # https://towardsdatascience.com/implementing-different-activation-functions-and-weight-initialization-methods-using-python-c78643b9f20f
    def forward_activation(self, z, func:str = "identity") -> np.ndarray:
        """{'identity', 'sigmoid', 'tanh', 'relu', 'softmax', 'leaky_relu'}, default='identity'"""
        if func == "identity":
            return z
        if func == "sigmoid":
            return self.sigmoid(z)
        elif func == "tanh":
            return np.tanh(z)
        elif func == "relu":
            return np.maximum(0, z)
        elif func == "leaky_relu":
            return np.maximum(self.leaky_slope * z, z)
        elif func == "softmax":
            return self.soft_max(z)
        else:
            raise ValueError(func, " not found in activation functions")
      
    def grad_activation(self, z, func:str = "identity") -> np.ndarray:
        """{'identity', 'sigmoid', 'tanh', 'relu', 'leaky_relu'}, default='identity'"""
        if func == "identity":
            return z
        elif func == "sigmoid":
            return self.sigmoid_prime(z)
        elif func == "tanh":
            return (1 - np.square(z))
        elif func == "relu":
            return 1.0 * (z > 0)
        elif func == "leaky_relu":
            d=np.zeros_like(z)
            d[z <= 0] = self.leaky_slope
            d[z > 0] = 1
            return d
        elif func == "softmax":
            raise ValueError("Softmax should not be used in middle layers")
        else:
            raise ValueError(func, " not found in activation functions")

    def forward(self, X_data: np.ndarray) -> np.ndarray:
        self.activations[0] = X_data
        # For all layers except the last one
        for layer in range(self.num_of_layers - 1):
            self.zs[layer] = np.dot(self.activations[layer], self.ws[layer])
            self.activations[layer + 1] = self.forward_activation(self.zs[layer], self.activation_functions[layer]) #self.sigmoid(self.zs[layer])
        
        last_layer = self.num_of_layers - 1
        self.zs[last_layer] = np.dot(self.activations[last_layer], self.ws[last_layer])

        # here use soft-max for the last layer if classification, but just identity if linear:
        return self.forward_activation(self.zs[last_layer], self.activation_functions[last_layer])

    def backward(self, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N = targets.shape[0]
        output_error = outputs - targets
        self.grads[-1] = self.cost_derivative(self.activations[-1], output_error, N)
        #self.grads[-1] = np.dot(self.activations[-1].T, output_error) / N
        for l in range(2, self.num_of_layers + 1): # OBS no +1 in the book

            # with ndarrays for hadamart multiplication just use *
            delta_cost = np.dot(output_error, self.ws[-l+1].T)
            # Compute error
            # δ = ∇C ⊙ σ′(z).
            activation = self.grad_activation(self.zs[-l], self.activation_functions[-l]) # self.sigmoid_prime(self.zs[-l])
            output_error = activation * delta_cost
            # backpropogate the error
            # δ = ((w of next layer )^T * δ of next layer) ⊙ σ′(z)
            average_grad = self.cost_derivative(self.activations[-l], output_error, N) # OBS activations[-l-1] in book and no /N
            # ηm∑xδx,l(ax,l−1)T Average gradient?
            self.grads[-l] = average_grad

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
