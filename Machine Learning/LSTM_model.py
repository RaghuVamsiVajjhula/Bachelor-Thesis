import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, cohen_kappa_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

def plot_accuracy_history(history, epochs,,save_path="server_plots/accuracy_point.txt"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), history.history['accuracy'], marker='o', color='blue', linewidth=2, label='Training Accuracy')
    plt.plot(range(1, epochs + 1), history.history['val_accuracy'], marker='o', color='green', linewidth=2, label='Validation Accuracy')
    
    plt.title('Model Accuracy Over Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Epoch', fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    os.makedirs('model_plots', exist_ok=True)
    plt.savefig('model_plots/accuracy_history.png')
    plt.show()
    with open(save_path, "w") as f:
        for acc in accuracy_history:
            f.write(f"{acc}\n")

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
    
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, test_size=0.2, 
                                                        train_size=0.8, random_state=1)
    
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

def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    
    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Kappa score: {kappa:.4f}")
    print(f"ROC AUC score: {roc:.4f}")
    
    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision": float(precision),
        "kappa": float(kappa),
        "roc_auc": float(roc)
    }

def main():
    file_path = "Six_Labels_CombinedDataset.csv"  
    
    x_train, x_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    input_shape = (1, x_train.shape[2])
    num_classes = y_train.shape[1]
    model = build_lstm_model(input_shape, num_classes)
    
    epochs = 50
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    print("\nTraining completed. Generating accuracy graph...")
    # plot_accuracy_history(history, epochs)
    filename="server_plots/accuracy_point.txt"
    plot_server_accuracy(accuracy_history,len(accuracy_history),save_path=filename)

    metrics = evaluate_model(model, x_test, y_test)
    
    model.save("LSTM_ACTIVITY_MODEL.h5")
    print("Model saved successfully")
    
    return model, metrics

if __name__ == "__main__":
    main()
