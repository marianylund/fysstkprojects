import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Needed for types:
from typing import Callable
from yacs.config import CfgNode as CN

def is_increasing(a):
    return np.all(np.diff(a) > 0)

def learning_schedule(t:float, t0:float, t1:float) -> float: 
    return t0/(t+t1)

class SimpleModel():
    def __init__(self, input_nodes:int, l2_reg_lambda: float = 1.0):
        self.I = input_nodes
        self.w = np.zeros((self.I, 1))
        #self.w = np.random.randn(self.I, 1) # TODO: change to uniform distribution
        self.grad = None

        self.l2_reg_lambda = l2_reg_lambda
    
    def forward(self, X_batch, y_batch):
        self.grad = 2 * X_batch.T @ ((X_batch @ self.w) - y_batch)
        return self

    def zero_grad(self) -> None:
        self.grad = None
    
def train_sgd(cfg:CN, X:np.array, y:np.array,
            scheduler: Callable[[float, float, float], float] = None):

    # Move this to data loader or something
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    scaler = StandardScaler()
    scaler.fit(X_train); X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)
    y_test.shape = (y_test.shape[0], 1)
    y_train.shape = (y_train.shape[0], 1)

    model = SimpleModel(X.shape[1])
    num_batches_per_epoch = X_train.shape[0] // cfg.BATCH_SIZE
    learning_rate = cfg.LR # do something with the scheduler

    train_mse = {}
    #train_r2 = {}

    # early_stopping_value = 0
    # early_stopping_step = 0

    global_step = 0
    
    for epoch in range(cfg.NUM_EPOCHS):
        print("Epohs: ", epoch)

        for step in range(num_batches_per_epoch):
            print("Step gone: ", step)

            # Select the mini-batch
            start = step * cfg.BATCH_SIZE
            end = start + cfg.BATCH_SIZE
            X_batch, y_batch = X_train[start:end], y_train[start:end]

            # Compute gradient:
            model.forward(X_batch, y_batch)
            model.w = model.w -  learning_rate * model.grad
            y_pred = X_batch @ model.w
            _train_mse = mean_squared_error(y_batch, y_pred)
            train_mse[global_step] = _train_mse

            global_step += 1

    return model, train_mse
    
    