import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import resample
import keras
from sklearn.model_selection import train_test_split
from keras import backend as K  
from imblearn.over_sampling import SMOTE
import keras
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

K.clear_session()

# ラベルのマッピング
label_mapping = {0: 0, 2: 1, 3: 2, 4: 3}
reverse_label_mapping = {v: k for k, v in label_mapping.items()}
seq_length = 60
# LSTMモデルの構築
model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 4)))  # 入力の形状を更新
model.add(Dense(4, activation='softmax'))


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def preprocess_data(file_path, window_size=50):
    data = pd.read_csv(file_path)
    temperatures = data['体表温度'].values
    hr = data['脈周期[ms]'].fillna(600)
    labels = data['ラベル'].values

    # 体表温度を移動平均で平滑化
    smoothed_temperatures = moving_average(temperatures, window_size)
    smoothed_hr = moving_average(hr,window_size*2)
    # 端のデータが失われるため、元の長さに戻す
    diff_length = len(temperatures) - len(smoothed_temperatures)
    padded_temperatures = np.pad(smoothed_temperatures, (diff_length//2, diff_length - diff_length//2), mode='edge')
    diff_length2 = len(hr) - len(smoothed_hr)
    padded_hr = np.pad(smoothed_hr, (diff_length2//2, diff_length2 - diff_length2//2), mode='edge')

    # 平滑化された体表温度を正規化
    min_temp = padded_temperatures.min()
    max_temp = padded_temperatures.max()
    normalized_temperatures = (padded_temperatures - min_temp) / (max_temp - min_temp)
    min_hr = padded_hr.min()
    max_hr = padded_hr.max()
    normalized_hr = (padded_hr - min_hr) / (max_hr - min_hr)

    # 正規化された体表温度の微分を計算
    diff_temp = np.diff(normalized_temperatures)
    diff_temp = np.insert(diff_temp, 0, 0)
    diff2_temp = np.diff(diff_temp)
    diff2_temp = np.insert(diff2_temp, 0, 0)

    # 正規化された体表温度とその微分を組み合わせる
    combined_features = np.vstack((normalized_temperatures, diff_temp, diff2_temp, normalized_hr)).T

    # シーケンスの作成
    X, y = [], []
    for i in range(len(combined_features) - seq_length):
        X.append(combined_features[i:i+seq_length])
        y.append(labels[i+seq_length])
    X = np.array(X)
    y = np.array(y)

    return X, y

def oversample_data(X, y):
    # クラスごとのデータのインデックスを取得
    class_indices = [np.where(y == i)[0] for i in np.unique(y)]
    max_size = np.max([len(indices) for indices in class_indices])
    X_oversampled, y_oversampled = [], []
    for indices in class_indices:
        oversampled_indices = resample(indices, replace=True, n_samples=max_size)
        X_oversampled.append(X[oversampled_indices])
        y_oversampled.append(y[oversampled_indices])
    X_oversampled = np.concatenate(X_oversampled)
    y_oversampled = np.concatenate(y_oversampled)
    return X_oversampled, y_oversampled

def smote_data(X, y):
    # Reshape from [samples, timesteps, features] to [samples, timesteps*features]
    num_samples, num_timesteps, num_features = X.shape
    X_reshaped = X.reshape((num_samples, num_timesteps * num_features))
    
    smote = SMOTE()
    X_smote_reshaped, y_smote = smote.fit_resample(X_reshaped, y)
    
    # Reshape back to [samples, timesteps, features]
    X_smote = X_smote_reshaped.reshape((-1, num_timesteps, num_features))
    
    return X_smote, y_smote

def undersample_data(X, y):
    # クラスごとのデータのインデックスを取得
    class_indices = [np.where(y == i)[0] for i in np.unique(y)]
    min_size = np.min([len(indices) for indices in class_indices])
    X_undersampled, y_undersampled = [], []
    for indices in class_indices:
        undersampled_indices = resample(indices, replace=False, n_samples=min_size)
        X_undersampled.append(X[undersampled_indices])
        y_undersampled.append(y[undersampled_indices])
    X_undersampled = np.concatenate(X_undersampled)
    y_undersampled = np.concatenate(y_undersampled)
    return X_undersampled, y_undersampled

def custom_loss(y_true, y_pred):
    # ラベル2と3のMSEを計算
    mse_label2 = keras.losses.mean_squared_error(y_true[:, 2], y_pred[:, 2])
    mse_label3 = keras.losses.mean_squared_error(y_true[:, 3], y_pred[:, 3])
    return keras.losses.categorical_crossentropy(y_true, y_pred) + 0.25 * (mse_label2 + mse_label3)


def to_categorical_custom(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

accumulate_val_accuracy = []
accumulte_accuracy = []
def train_lstm_model(train_path,isGraph=True,epoch=1):
    if isGraph:
        graph(train_path)
    global accumulate_val_accuracy
    global accumulte_accuracy
    # graph(train_path)  # Assuming this is some visualization, commented out for brevity
    X, y = preprocess_data(train_path)
    y_mapped = np.array([label_mapping[label] for label in y])

    nan_positions = np.where(np.isnan(X))
    unique_positions, counts = np.unique(nan_positions[1], return_counts=True)
    print(f"Features with NaNs: {unique_positions}")
    print(f"Count of NaNs for each feature: {counts}")


    X_train, X_test, y_train_mapped, y_test_mapped = train_test_split(X, y_mapped, test_size=0.2, random_state=42)



    # X_train_smote, y_train_smote = smote_data(X_train, y_train_mapped)
    X_train_smote, y_train_smote = oversample_data(X_train, y_train_mapped)
    # X_train_smote, y_train_smote = undersample_data(X_train, y_train_mapped)

    num_classes = len(np.unique(y_train_smote))
    y_train_cat = to_categorical_custom(y_train_mapped, num_classes=num_classes)
    y_test_cat = to_categorical_custom(y_test_mapped, num_classes=num_classes)

    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
    class_weight_dict = {i: weights[i] for i in range(len(weights))}

    # Assuming model is already defined elsewhere
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping mechanism
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train_smote, to_categorical_custom(y_train_smote, num_classes=num_classes),
        epochs=epoch, 
        batch_size=32, 
        validation_data=(X_test, y_test_cat),
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    # Plot training & validation accuracy values
    # two row one line to one row two line
    # Assuming history.history['accuracy'] and history.history['val_accuracy'] are lists of floats
    accumulte_accuracy.extend(history.history['accuracy'])
    accumulate_val_accuracy.extend(history.history['val_accuracy'])

    if isGraph:
        plt.plot(accumulte_accuracy)
        plt.plot(accumulate_val_accuracy)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

import os
def get_files(folder_path):
    files = os.listdir(folder_path)
    csv_files = []
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(folder_path+"/"+file)
    return csv_files

