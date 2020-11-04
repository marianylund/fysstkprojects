import numpy as np

from nnreg.model import Model

error_tolerance = 1e-10

def test_accuracy():
    y_data = np.asarray([[0, 0, 1, 0]])
    y_pred = np.asarray([[1, 0, 0, 0]])
    acc = Model.calculate_accuracy(y_data, y_pred)
    assert acc == 0, acc

    acc = Model.calculate_accuracy(y_data, y_data)
    assert acc == 1, acc

    y_data = np.asarray([[0, 1]])
    y_pred = np.asarray([[1, 0]])
    acc = Model.calculate_accuracy(y_data, y_pred)
    assert acc == 0, acc

    acc = Model.calculate_accuracy(y_data, y_data)
    assert acc == 1, acc

if __name__ == "__main__":
    print("Start tests for project 2")
    test_accuracy()

    print("All tests have passed")
