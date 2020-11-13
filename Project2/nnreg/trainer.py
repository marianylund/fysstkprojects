import numpy as np
from pathlib import Path
from math import inf, isnan
from time import time

from RegLib.load_save_data import save_checkpoint
from RegLib.HelperFunctions import progressBar
from nnreg.model import Model
from nnreg.dataloader import DataLoader

# Needed for types:
from typing import Callable
from yacs.config import CfgNode as CN

class Trainer():
    def train_and_save(self, cfg:CN, data_loader:DataLoader, checkpoints_path:Path = None):
        self.model = Model(cfg, data_loader.X_train.shape[1], data_loader.y_train.shape[1])
        if(checkpoints_path == None):
            checkpoints_path = Path.cwd()
        self.train(cfg, self.model, data_loader.X_train, data_loader.X_test, data_loader.X_val, data_loader.y_train, data_loader.y_test, data_loader.y_val, checkpoints_path)
        return self
    
    def train(self, cfg:CN, model, X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, y_val:np.ndarray,
                    checkpoints_path:Path):

        batch_size = cfg.OPTIM.BATCH_SIZE
        num_batches_per_epoch = X_train.shape[0] // batch_size
        learning_rate = cfg.OPTIM.LR
        decay = cfg.OPTIM.LR_DECAY
        use_shuffle = cfg.SHUFFLE
        eval_step = cfg.EVAL_STEP if cfg.EVAL_STEP > 1 else 1
        log_step = cfg.LOG_STEP * eval_step
        use_momentum = cfg.OPTIM.USE_MOMENTUM
        if use_momentum:
            velocity = [0 for i in range(len(model.ws))]
            momentum_gamma = cfg.OPTIM.MOMENTUM

        train_eval = {}
        val_eval = {}
        learning_rate_all = {}
        train_r2 = {}
        use_accuracy = cfg.MODEL.EVAL_FUNC == "acc"
        best_eval = inf * (-1 if use_accuracy else 1) # 1 is best for accuracy, 0 for MSE
        best_eval_step = -1

        global_step = 0
        total_steps = cfg.OPTIM.NUM_EPOCHS * num_batches_per_epoch

        self.start_time = time()
        
        for epoch in range(cfg.OPTIM.NUM_EPOCHS):

            _lr = Trainer.learning_schedule(learning_rate, decay, epoch)

            if use_shuffle:
                s = np.arange(X_train.shape[0])
                np.random.shuffle(s)
                X_train = X_train[s]
                y_train = y_train[s]

            for step in range(num_batches_per_epoch):

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
                
                if( global_step % eval_step == 0): # Time to evaluate the model
                    # Compute the cost
                    _eval_train = model.get_evaluation(y_batch, y_pred)
                    train_eval[global_step] = _eval_train

                    train_r2[global_step] = Trainer.R2(y_batch, y_pred)
                    learning_rate_all[global_step] = _lr

                    y_pred_val = model.forward(X_val)

                    _eval = model.get_evaluation(y_val, y_pred_val)
                    val_eval[global_step] = _eval

                    if (not use_accuracy and _eval < best_eval) or (use_accuracy and _eval > best_eval): # Save best model
                        best_eval_step = global_step
                        best_eval = _eval
                        test_pred = model.forward(X_test)
                        self.best_test_eval = model.get_evaluation(y_test, test_pred)
                        state_dict = {
                            "Step": global_step,
                            "Test_eval": self.best_test_eval,
                            "Test_r2": Trainer.R2(y_test, test_pred),
                            "Proccess_time": self.get_time(),
                            "Train_eval": train_eval,
                            "Train_r2": train_r2,
                            "Val_eval": val_eval,
                            "Weights": model.ws.tolist(),
                            "Learning_rate": learning_rate_all,
                        }
                        save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=True, max_keep=1)

                if( global_step % cfg.MODEL_SAVE_STEP == 0): # Time to save the model
                    state_dict = {
                        "Proccess_time": self.get_time(),
                        "Weights": model.ws.tolist(),
                        "Learning_rate": learning_rate_all,
                        "Train_eval": train_eval,
                        "Val_eval": val_eval,
                    }
                    save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=False, max_keep=1)
                
                if( global_step % log_step == 0): # Time to log
                    msg = f"Step: {global_step} train_eval: {train_eval[global_step]}, last best: {abs(global_step - best_eval_step)}"
                    progressBar(global_step, total_steps, msg)

                    if(train_eval[global_step] == inf or isnan(train_eval[global_step])):
                        m, s = self.get_time()
                        print(f"Network failing, train eval is {train_eval[global_step]}, lr: {learning_rate}, batch_size {batch_size}, name: {cfg.OUTPUT_DIR}\n")

                        return self
                
                if(cfg.OPTIM.EARLY_STOP_LR_STEP != -1 and abs(global_step - best_eval_step) >= cfg.OPTIM.EARLY_STOP_LR_STEP):
                    m, s = self.get_time()
                    print(f"\n{global_step} step. Finished early: {m:.0f}:{s:.0f}")
                    return self
                
               

                global_step += 1

         # Compute the cost
        _eval_train = model.get_evaluation(y_batch, y_pred)
        train_eval[global_step] = _eval_train

        train_r2[global_step] = Trainer.R2(y_batch, y_pred)
        learning_rate_all[global_step] = _lr

        y_pred_val = model.forward(X_val)

        _eval = model.get_evaluation(y_val, y_pred_val)
        val_eval[global_step] = _eval

        if (not use_accuracy and _eval < best_eval) or (use_accuracy and _eval > best_eval): # Save best model
            best_eval = _eval
            test_pred = model.forward(X_test)
            self.best_test_eval = model.get_evaluation(y_test, test_pred)
            state_dict = {
                "Step": global_step,
                "Test_eval": self.best_test_eval,
                "Test_r2": Trainer.R2(y_test, test_pred),
                "Proccess_time": self.get_time(),
                "Train_eval": train_eval,
                "Train_r2": train_r2,
                "Val_eval": val_eval,
                "Weights": model.ws.tolist(),
                "Learning_rate": learning_rate_all,
            }
            save_checkpoint(state_dict, checkpoints_path.joinpath(str(global_step)+".json"), is_best=True, max_keep=1)

        m, s = self.get_time()
        print(f"{global_step} step. Finished. Time: {m:.0f}:{s:.0f}")
        return self
    
    def get_time(self):
        m, s = divmod(time() - self.start_time, 60)
        return m, s

    @staticmethod
    def learning_schedule(learning_rate:float, decay:float, epoch:float) -> float: 
        return learning_rate * 1/(1 + decay * epoch)

    @staticmethod
    def R2(y_data, y_pred):
        return 1 - np.sum((y_data - y_pred)**2) / np.sum((y_data - np.mean(y_data))**2)

    @staticmethod
    def MSE(y_data, y_pred):
        return np.mean((y_data - y_pred)**2)
