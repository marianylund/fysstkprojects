import numpy as np
import mnist
from sklearn.model_selection import train_test_split

from RegLib.HelperFunctions import create_frankie_data, create_X, plot_values_with_info,plot_values_with_two_y_axis

from yacs.config import CfgNode as CN


class DataLoader():
    """
    Contains X_train, X_test, y_train, y_test for the dataset given in the config. 
    Can also have X_val and y_val for other sets than franke, size depends on the the config
    """

    def __init__(self, cfg: CN, perm_index = [-1]):
        self.data_name = cfg.DATA.NAME

        if self.data_name == "franke":
            self.load_franke_data(cfg, perm_index)
        elif self.data_name == "mnist":
            self.load_mnist_data(cfg)
        else:
            raise ValueError(self.data_name, " is not found in DataLoader init")

    def load_franke_data(self, cfg: CN, perm_index):
        x, y, z = create_frankie_data(cfg.SEED, cfg.DATA.FRANKIE.N, cfg.DATA.FRANKIE.NOISE)
        X = create_X(x, y, cfg.DATA.FRANKIE.P)
        
        self.split_and_scale_train_test(X, z, perm_index, test_size = cfg.TEST_SIZE)
        return self

    def load_mnist_data(self, cfg: CN):
        val_percent = cfg.DATA.MNIST.VAL_PERCENT
        binary_classes = cfg.DATA.MNIST.BINARY
        num_of_classes = len(binary_classes)
        if(num_of_classes != 0):
            assert num_of_classes == 2, "Cannot have " + str(num_of_classes) + " classes"
            X_train, self.y_train, X_val, self.y_val, X_test, self.y_test = load_binary_dataset(binary_classes[0], binary_classes[1], val_percent)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_full_mnist(val_percent)
            # One hot encode the results
            self.y_train = self.one_hot_encode(y_train, 10)
            self.y_val = self.one_hot_encode(y_val, 10)
            self.y_test = self.one_hot_encode(y_test, 10)

        # Pre-process the batch
        X_mean, X_std = (np.mean(X_train), np.std(X_train))
        self.X_train = self.pre_process_images(X_train, X_mean, X_std)
        self.X_val = self.pre_process_images(X_val, X_mean, X_std)
        self.X_test = self.pre_process_images(X_test, X_mean, X_std)

    
    def one_hot_encode(self, Y: np.ndarray, num_classes: int):
        new_Y = np.zeros((Y.shape[0], num_classes))
        for i in range(len(Y)):
            new_Y[i][Y[i]] = 1
        return new_Y

    def pre_process_images(self, X: np.ndarray, X_mean: float, X_std: float):
        assert X.shape[1] == 784,\
            f"X.shape[1]: {X.shape[1]}, should be 784"
        X = (X - X_mean) / X_std
        X = np.c_[X,  np.ones(X.shape[0])] # Apply bias trick
        return X
    
    def split_and_scale_train_test(self, X, y, perm_index = [-1], test_size  = 0.2):
        assert X.shape[0] == y.shape[0], ("X.shape[0] and y.shape[0] needs to be the same length, but: " + str(X.shape[0]) + " != " + str(y.shape[0]))
        if(len(perm_index) > 1):
            X = X[perm_index]
            y = y[perm_index]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        self.X_train, self.X_test = self.scale_standard(self.X_train, self.X_test)
        # Force the correct shape:
        self.y_test.shape = (self.y_test.shape[0], 1)
        self.y_train.shape = (self.y_train.shape[0], 1)
        return self
    
    def scale_standard(self, train_data, test_data):
        data_mean = np.mean(train_data[:,1:], axis = 0)
        data_std = np.std(train_data[:,1:], axis = 0)
        train_data_scaled = train_data
        test_data_scaled = test_data
        train_data_scaled[:,1:] = np.divide((train_data[:,1:] - data_mean), data_std)
        test_data_scaled[:,1:] = np.divide((test_data[:,1:] - data_mean), data_std)
        
        return train_data_scaled, test_data_scaled


# https://github.com/hukkelas/TDT4265-StarterCode/tree/master/assignment1
def binary_prune_dataset(class1: int, class2: int,
                        X: np.ndarray, Y: np.ndarray):
    """
    Splits the dataset into the class 1 and class2. All other classes are removed.
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
        Y: labels of shape [batch size]
    """

    mask1 = (Y == class1)
    mask2 = (Y == class2)
    mask_total = np.bitwise_or(mask1, mask2)
    Y_binary = Y.copy()
    Y_binary[mask1] = 1
    Y_binary[mask2] = 0
    return X[mask_total], Y_binary[mask_total]

def train_val_split(X: np.ndarray, Y: np.ndarray, val_percentage: float):
    """
    Randomly splits the training dataset into a training and validation set.
    """
    idx = np.arange(0, X.shape[0])
    np.random.shuffle(idx)
    train_size = int(X.shape[0] * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]

    return X_train, Y_train, X_val, Y_val


def load_binary_dataset(class1: int, class2: int, val_percentage: float):
    """
    Loads, prunes and splits the dataset into train, validation and test.
    """
    train_size = 20000
    test_size = 2000
    X_train, Y_train, X_test, Y_test = mnist.load()

    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_test, Y_test = X_test[-test_size:], Y_test[-test_size:]
    X_train, Y_train = binary_prune_dataset(
        class1, class2, X_train, Y_train
    )
    X_test, Y_test = binary_prune_dataset(
        class1, class2, X_test, Y_test
    )
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    X_train, Y_train, X_val, Y_val = train_val_split(
        X_train, Y_train, val_percentage
    )
    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")
    print(f"Test shape: X: {X_test.shape}, Y: {Y_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def load_full_mnist(val_percentage: float):
    """
    Loads and splits the dataset into train, validation and test.
    """
    train_size = 20000
    test_size = 2000
    X_train, Y_train, X_test, Y_test = mnist.load()
    
    # First 20000 images from train set
    X_train, Y_train = X_train[:train_size], Y_train[:train_size]
    # Last 2000 images from test set
    X_test, Y_test = X_test[-test_size:], Y_test[-test_size:]
    # Reshape to (N, 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    X_train, Y_train, X_val, Y_val = train_val_split(
        X_train, Y_train, val_percentage
    )
    print(f"Train shape: X: {X_train.shape}, Y: {Y_train.shape}")
    print(f"Validation shape: X: {X_val.shape}, Y: {Y_val.shape}")
    print(f"Test shape: X: {X_test.shape}, Y: {Y_test.shape}")

    return X_train, Y_train, X_val, Y_val, X_test, Y_test