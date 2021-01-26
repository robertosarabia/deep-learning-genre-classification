import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

if __name__ == "__main__":

    # load data
    inputs, targets = load_data(DATASET_PATH)

    # split into train and test sets
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.3)

    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        # first hidden layer
        keras.layers.Dense(512, activation="relu"),
        # second hidden layer
        keras.layers.Dense(256, activation="relu"),
        #third hidden layer
        keras.layers.Dense(64, activation="relu"),
        # output layer
        keras.layers.Dense(10, activation="softmax")
    ])

    # compile network

    # train network
