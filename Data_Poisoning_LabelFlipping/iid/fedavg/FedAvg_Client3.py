

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

import flwr as fl
import os

FILE_PATH = "./Data/iid_part_three.csv" 

def load_and_preprocess_data(file_path):
    dataset = pd.read_csv(file_path)

    label_encoder = LabelEncoder()
    dataset['activity'] = label_encoder.fit_transform(dataset['activity'])

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled = x_scaled.reshape(x_scaled.shape[0], 1, x_scaled.shape[1])

    y_onehot = to_categorical(y)

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, test_size=0.2, train_size=0.8, random_state=1)
    return x_train, x_test, y_train, y_test


def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class LSTMClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=64, verbose=1)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        y_pred_prob = self.model.predict(self.x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        kappa = cohen_kappa_score(y_true, y_pred)
        roc = roc_auc_score(self.y_test, y_pred_prob, multi_class='ovr')

        print(f"Accuracy: {accuracy:.2f}, F1-score: {f1:.2f}, Precision: {precision:.2f}, Kappa: {kappa:.2f}, ROC AUC: {roc:.2f}")

        return loss, len(self.x_test), {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "kappa": float(kappa),
        "roc_auc": float(roc)
         }
    

from flwr.common import Context
from flwr.client import ClientApp

def client_fn3(context: Context):
    x_train, x_test, y_train, y_test = load_and_preprocess_data(FILE_PATH)
    model = build_lstm_model((1, x_train.shape[2]), y_train.shape[1])
    client = LSTMClient(model, x_train, x_test, y_train, y_test)
    return client


if __name__ == "__main__":
    app = ClientApp(client_fn=client_fn3)
