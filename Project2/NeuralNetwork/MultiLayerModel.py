# Inspired from: https://github.com/hukkelas/TDT4265-StarterCode
import numpy as np
import typing
from yacs.config import CfgNode as CN

class MultiLayerModel():
    def __init__(self, cfg:CN, input_nodes:int):
        # neurons_per_layer: typing.List[int], input_nodes:int, l2_reg_lambda: float = 1.0, linear = True
        self.I = input_nodes
        self.linear = cfg.MODEL.LINEAR # false if classification
        self.neurons_per_layer = cfg.MODEL.SHAPE
        if self.linear:
            self.neurons_per_layer[-1] = input_nodes # if it is linear, input will be equal the output
        self.shape = [input_nodes] + self.neurons_per_layer
        print("Shape: ", self.shape)
        self.num_of_layers = len(cfg.MODEL.SHAPE)
        self.l2_reg_lambda = cfg.OPTIM.L2_REG_LAMBDA

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
        print("Number of layers: " + str(self.num_of_layers))
        # For all layers except the last one
        for layer in range(self.num_of_layers - 1):
            # print("Layer: " + str(layer))
            self.zs[layer] = np.dot(self.activations[layer], self.ws[layer])
            self.activations[layer + 1] = self.sigmoid(self.zs[layer])
            print("Layer: ", str(layer), " zs: ", self.zs[layer].shape, " activation: ", self.activations[layer + 1].shape)
        
        # here use soft-max for the last layer if classification, but just identity if linear:
        last_layer = self.num_of_layers - 1
        if self.linear:
            self.zs[last_layer] = np.dot(self.activations[last_layer], self.ws[last_layer])
            self.activations[last_layer] = self.sigmoid(self.zs[last_layer])
            y_pred = self.activations[last_layer]
            print("y_pred: ", y_pred.shape)
           # y_pred.shape = (y_pred.shape[0], 1)
            return y_pred
        else:
            self.zs[last_layer] = np.dot(self.activations[last_layer], self.ws[last_layer])
            y_pred = self.soft_max(self.zs[last_layer])
            return y_pred

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"

        N = targets.shape[0]

        output_error = self.cost_derivative(targets, outputs)
        print("Backwards activations: ", self.activations)
        self.grads[-1] = np.dot(self.activations[-1].T, output_error) / N
        for l in range(2, self.num_of_layers + 1): # OBS no +1 in the book
            print("Backward layer: ", str(l), "activations: ", self.activations[-l])

            # with ndarrays for hadamart multiplication just use *
            delta_cost = np.dot(output_error, self.ws[-l+1].T)
            # Compute error
            # δ = ∇C ⊙ σ′(z).
            output_error = self.sigmoid_prime(self.zs[-l]) * delta_cost
            # backpropogate the error
            # δ = ((w of next layer )^T * δ of next layer) ⊙ σ′(z)
            average_grad = np.dot(self.activations[-l].T, output_error) / N # OBS activations[-l-1] in book and no /N
            # ηm∑xδx,l(ax,l−1)T Average gradient?
            self.grads[-l] = average_grad

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
