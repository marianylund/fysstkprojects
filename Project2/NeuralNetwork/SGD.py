import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from math import inf
from time import time

from RegLib.load_save_data import save_checkpoint
from RegLib.HelperFunctions import progressBar

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
        learning_rate = cfg.OPTIM.LR
        decay = cfg.OPTIM.LR_DECAY

        train_mse = {}
        learning_rate_all = {}
        train_r2 = {}
        
        best_mse = inf # 0 is best

        global_step = 0
        total_steps = cfg.OPTIM.NUM_EPOCHS * num_batches_per_epoch

        start_time = time()
        
        for epoch in range(cfg.OPTIM.NUM_EPOCHS):

            _lr = SGD.learning_schedule(learning_rate, decay, epoch)

            for step in range(num_batches_per_epoch):
                progressBar(global_step, total_steps)

                # Select the mini-batch
                start = step * batch_size
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # Compute gradient:
                model.forward(X_batch, y_batch)
                _lr_step = _lr * model.grad
                model.w = model.w - _lr_step
                
                y_pred = X_batch @ model.w
                _mse = SGD.MSE(y_batch, y_pred)
                train_mse[global_step] = _mse

                train_r2[global_step] = SGD.R2(y_batch, y_pred)
                learning_rate_all[global_step] = _lr

                if _mse < best_mse: # Save best model
                    best_mse = _mse
                    test_pred = X_test @ model.w
                    self.best_test_mse = SGD.MSE(y_test, test_pred)
                    state_dict = {
                        "Step": global_step,
                        "Weights": model.w.tolist(),
                        "Test_mse": self.best_test_mse,
                        "Test_r2": SGD.R2(y_test, test_pred),
                        "Train_mse": train_mse,
                        "Train_r2": train_r2,
                        "Learning_rate": learning_rate_all,
                    }
                    save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=True, max_keep=1)
                if( global_step % cfg.MODEL_SAVE_STEP == 0): # Time to save the model
                    state_dict = {
                        "Weights": model.w.tolist(),
                        "Train_mse": train_mse,
                        "Train_r2": train_r2,
                        "Learning_rate": learning_rate_all,
                    }
                    save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=False, max_keep=1)
                if(cfg.OPTIM.EARLY_STOP_LR_STEP != -1 and abs(np.mean(_lr_step)) <= cfg.OPTIM.EARLY_STOP_LR_STEP):
                    self.process_time = time() - start_time
                    print(global_step, " step. Finished early: ", np.mean(_lr_step))
                    return self
                global_step += 1
        self.process_time = time() - start_time
        print("Finished.")
        return self
    
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
    def learning_schedule(learning_rate:float, decay:float, epoch:float) -> float: 
        return learning_rate * 1/(1 + decay * epoch)

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