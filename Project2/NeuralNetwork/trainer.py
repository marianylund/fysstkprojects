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

def train(cfg:CN, X:np.array, y:np.array, model,
            scheduler: Callable[[float, float, float], float]):
    
    # Move this to data loader or something
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    scaler = StandardScaler()
    scaler.fit(X_train); X_train = scaler.transform(X_train); X_test = scaler.transform(X_test)
    
    num_batches_per_epoch = X_train.shape[0] // cfg.BATCH_SIZE
    num_steps_per_val = num_batches_per_epoch // 5
    learning_rate = cfg.LR # do something with the scheduler

    train_mse = {}
    train_r2 = {}

    early_stopping_value = 0
    early_stopping_step = 0

    global_step = 0
    
    for epoch in range(cfg.NUM_EPOCHS):
        for step in range(num_batches_per_epoch):

            # Select the mini-batch
            start = step * cfg.BATCH_SIZE
            end = start + cfg.BATCH_SIZE
            X_batch, Y_batch = X_train[start:end], y_train[start:end]

            # forward
            y_pred = model.forward(X_batch)
            # backwards, compute gradient
            model.backward(X_batch, y_pred, Y_batch)

            # update weights minus or plus?
            model.w -= model.grad * learning_rate

            _train_mse = mean_squared_error(Y_batch, y_pred)
            train_mse[global_step] = _train_mse

            global_step += 1
    
    return model, train_mse