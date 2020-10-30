# Lots of copy paste from SGD, maybe merge them somehow?
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from math import inf
from time import time

from RegLib.load_save_data import save_checkpoint
from RegLib.HelperFunctions import progressBar
from NeuralNetwork.MultiLayerModel import MultiLayerModel
from NeuralNetwork.SGD import SGD
from NeuralNetwork.dataloader import DataLoader

# Needed for types:
from typing import Callable
from yacs.config import CfgNode as CN

class Trainer():
    def train_and_test(self, cfg:CN, data_loader:DataLoader, checkpoints_path:Path = None):
        print("data_loader.y_train.shape: ", data_loader.y_train.shape)
        self.model = MultiLayerModel(cfg, data_loader.X_train.shape[1], data_loader.y_train.shape[1]) # here it was X.shape[1]
        if(checkpoints_path == None):
            checkpoints_path = Path.cwd()
        self.train(cfg, self.model, data_loader.X_train, data_loader.X_test, data_loader.y_train, data_loader.y_test, checkpoints_path)
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
    
    def train(self, cfg:CN, model, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray,
                    checkpoints_path:Path):

        batch_size = cfg.OPTIM.BATCH_SIZE
        num_batches_per_epoch = X_train.shape[0] // batch_size
        learning_rate = cfg.OPTIM.LR
        decay = cfg.OPTIM.LR_DECAY
        use_shuffle = cfg.SHUFFLE
        use_momentum = cfg.OPTIM.USE_MOMENTUM
        if use_momentum:
            velocity = [0 for i in range(len(model.ws))]
            momentum_gamma = cfg.OPTIM.MOMENTUM

        train_eval = {}
        learning_rate_all = {}
        train_r2 = {}
        use_accuracy = cfg.MODEL.ACTIVATION_FUNCTIONS[-1] == "softmax"
        best_eval = inf * (-1 * use_accuracy) # 1 is best for accuracy, 0 for MSE

        global_step = 0
        total_steps = cfg.OPTIM.NUM_EPOCHS * num_batches_per_epoch

        start_time = time()
        
        for epoch in range(cfg.OPTIM.NUM_EPOCHS):

            _lr = SGD.learning_schedule(learning_rate, decay, epoch)

            if use_shuffle:
                s = np.arange(X_train.shape[0])
                np.random.shuffle(s)
                X_train = X_train[s]
                y_train = y_train[s]

            for step in range(num_batches_per_epoch):
                progressBar(global_step, total_steps)

                # Select the mini-batch
                start = step * batch_size
                end = start + batch_size
                X_batch, y_batch = X_train[start:end], y_train[start:end]

                # Compute gradient:
                y_pred = model.forward(X_batch)
                model.backward(y_pred, y_batch)

                # Update the weights
                _lr_step = np.multiply(model.grads, _lr)
                if(use_momentum):
                    velocity = np.multiply(velocity, momentum_gamma) - _lr_step
                    model.ws = model.ws + velocity
                else:
                    model.ws = model.ws - _lr_step
                
                # Compute the cost
                _eval = model.get_evaluation(y_batch, y_pred)
                train_eval[global_step] = _eval

                train_r2[global_step] = SGD.R2(y_batch, y_pred)
                learning_rate_all[global_step] = _lr

                if (not use_accuracy and _eval < best_eval) or (use_accuracy and _eval > best_eval): # Save best model
                    best_eval = _eval
                    test_pred = model.forward(X_test)
                    self.best_test_eval = model.get_evaluation(y_test, test_pred)
                    state_dict = {
                        "Step": global_step,
                        "Test_eval": self.best_test_eval,
                        "Test_r2": SGD.R2(y_test, test_pred),
                        "Train_eval": train_eval,
                        "Train_r2": train_r2,
                        "Weights": model.ws.tolist(),
                        "Learning_rate": learning_rate_all,
                    }
                    save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=True, max_keep=1)

                if( global_step % cfg.MODEL_SAVE_STEP == 0): # Time to save the model
                    state_dict = {
                        "Train_eval": train_eval,
                        "Train_r2": train_r2,
                        "Weights": model.ws.tolist(),
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
