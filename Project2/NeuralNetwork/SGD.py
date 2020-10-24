import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

from RegLib.load_save_data import *
# Needed for types:
from typing import Callable
from yacs.config import CfgNode as CN

class SGD():

    def train_and_test(self, cfg:CN, X, y, checkpoints_path:Path = None, perm_index = [-1]):
        self.model = SimpleModel(X.shape[1], l2_reg_lambda=cfg.OPTIM.L2_REG_LAMBDA)
        self.split_and_scale_train_test(X, y, perm_index, cfg.TEST_SIZE)
        if(checkpoints_path == None):
            checkpoints_path = Path.cwd()
        self.train_sgd(cfg, self.model, self.X_train, self.X_test, self.y_train, self.y_test, checkpoints_path)
        return self
    
    def train_sgd(self, cfg:CN, model, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray,
                    checkpoints_path:Path):
        batch_size = cfg.OPTIM.BATCH_SIZE
        num_batches_per_epoch = X_train.shape[0] // batch_size
        learning_rate = cfg.OPTIM.LR # do something with the scheduler
        #if cfg.SCHEDULER:
            # do something with the scheduler
        train_mse = {}
        learning_rate_all = {}
        train_r2 = {}
        lr_step = {}
        
        # early_stopping_value = 0
        # early_stopping_step = 0

        global_step = 0
        
        for epoch in range(cfg.OPTIM.NUM_EPOCHS):
            print("Epohs: ", epoch)

            for step in range(num_batches_per_epoch):
                print("Step gone: ", step)

                # Select the mini-batch
                start = step * batch_size
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # Compute gradient:
                model.forward(X_batch, y_batch)
                _lr_step = learning_rate * model.grad
                lr_step[global_step] = _lr_step.tolist()
                model.w = model.w - _lr_step
                
                y_pred = X_batch @ model.w
                train_mse[global_step] = SGD.MSE(y_batch, y_pred)
                train_r2[global_step] = SGD.R2(y_batch, y_pred)
                learning_rate_all[global_step] = learning_rate

                if( global_step % cfg.MODEL_SAVE_STEP == 0): # Time to save the model
                    state_dict = {
                        "Weights": model.w.tolist(),
                        "Train_mse": train_mse,
                        "Train_r2": train_r2,
                        "Learning_rate": learning_rate_all,
                        "Learning_step": lr_step
                    }
                    save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=False, max_keep=5)

                global_step += 1
        print("Finished.")
        return model, train_mse
    
    def split_and_scale_train_test(self, X, y, perm_index = [-1], test_size  = 0.2):
        assert X.shape[0] == y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(X.shape[0]) + " != " + str(y.shape[0]))
        if(len(perm_index) > 1):
            X = X[perm_index]
            y = y[perm_index]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        self.X_train, self.X_test = SGD.scale_standard(self.X_train, self.X_test)
        # Force the correct shape:
        self.y_test.shape = (self.y_test.shape[0], 1)
        self.y_train.shape = (self.y_train.shape[0], 1)
        return self

    @staticmethod  
    def is_increasing(a):
        return np.all(np.diff(a) > 0)

    @staticmethod
    def learning_schedule(t:float, t0:float, t1:float) -> float: 
        return t0/(t+t1)

    @staticmethod
    def R2(y_data, y_pred):
        return 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_pred):
        return np.mean((y_data - y_pred)**2)

    @staticmethod
    def scale_standard(train_data, test_data):
        data_mean = np.mean(train_data[:,1:], axis = 0)
        data_std = np.std(train_data[:,1:], axis = 0)
        train_data_scaled = train_data
        test_data_scaled = test_data
        train_data_scaled[:,1:] = np.divide((train_data[:,1:] - data_mean), data_std)
        test_data_scaled[:,1:] = np.divide((test_data[:,1:] - data_mean), data_std)
        
        return train_data_scaled, test_data_scaled

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