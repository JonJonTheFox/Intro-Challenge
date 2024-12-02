import pandas as pd
import seaborn as sns
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from scipy.stats import skew
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
import numpy as np
import json

# Load your datasets
aorta_test_df = pd.read_csv('aortaP_test_data.csv')  # Replace with actual test file path
brachial_test_df = pd.read_csv('brachP_test_data.csv')  # Replace with actual test file path
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')

def handle_nans_knn(df, dataset_name, n_neighbors=5):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    print(f"{dataset_name} - Total NaNs after KNN imputation: {df_imputed.isna().sum().sum()}")
    return df_imputed

def add_log_transform(df):
    # Logarithmic transformation of pressure values (to accentuate small differences)
    df['log_pressure'] = np.log1p(df['pressure'])
    return df

def add_polynomial_features(df, degree=2):
    poly = PolynomialFeatures(degree=degree)
    df_poly = poly.fit_transform(df)
    return pd.DataFrame(df_poly, columns=poly.get_feature_names_out(df.columns))

def add_rolling_window_features(df, window_size=3):
    df['moving_avg'] = df.rolling(window=window_size, axis=1).mean().mean(axis=1)
    return df

def add_pressure_ratio(df_aorta, df_brachial):
    df_aorta['pressure_ratio'] = df_aorta['max_pressure'] / df_aorta['min_pressure']
    df_brachial['pressure_ratio'] = df_brachial['max_pressure'] / df_brachial['min_pressure']
    return df_aorta, df_brachial

def add_pressure_diff(df_aorta, df_brachial):
    df_aorta['pressure_diff'] = df_aorta.diff(axis=1).mean(axis=1)
    df_brachial['pressure_diff'] = df_brachial.diff(axis=1).mean(axis=1)
    return df_aorta, df_brachial

def add_pressure_skew(df_aorta, df_brachial):
    df_aorta['pressure_skew'] = df_aorta.apply(lambda row: skew(row), axis=1)
    df_brachial['pressure_skew'] = df_brachial.apply(lambda row: skew(row), axis=1)
    return df_aorta, df_brachial

def add_pressure_asymmetry(df_aorta, df_brachial):
    df_aorta['pressure_asymmetry'] = (df_aorta['max_pressure'] - df_aorta['min_pressure']) / df_aorta['median_pressure']
    df_brachial['pressure_asymmetry'] = (df_brachial['max_pressure'] - df_brachial['min_pressure']) / df_brachial['median_pressure']
    return df_aorta, df_brachial

aorta_df = handle_nans_knn(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans_knn(brachial_df, 'Brachial Dataset')

aorta_df = add_log_transform(aorta_df)
brachial_df = add_log_transform(brachial_df)
aorta_df = add_polynomial_features(aorta_df)
brachial_df = add_polynomial_features(brachial_df)

aorta_df, brachial_df = add_rolling_window_features(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_ratio(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_diff(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_skew(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_asymmetry(aorta_df, brachial_df)

aorta_test_df = handle_nans_knn(aorta_test_df, 'Aorta Test Dataset')
brachial_test_df = handle_nans_knn(brachial_test_df, 'Brachial Test Dataset')

aorta_test_df = add_log_transform(aorta_test_df)
brachial_test_df = add_log_transform(brachial_test_df)
aorta_test_df = add_polynomial_features(aorta_test_df)
brachial_test_df = add_polynomial_features(brachial_test_df)

aorta_test_df, brachial_test_df = add_rolling_window_features(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_ratio(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_diff(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_skew(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_asymmetry(aorta_test_df, brachial_test_df)

scaler = RobustScaler()
aorta_normalized = scaler.fit_transform(aorta_df)
brachial_normalized = scaler.fit_transform(brachial_df)

assert aorta_normalized.shape[0] == len(aorta_targets), "Mismatch between Aorta features and targets length."
assert brachial_normalized.shape[0] == len(brachial_targets), "Mismatch between Brachial features and targets length."

X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, aorta_targets, test_size=0.2, random_state=42)

X_train_aorta_reshaped = X_train_aorta[..., np.newaxis]
X_train_brachial_reshaped = X_train_brachial[..., np.newaxis]
X_val_aorta_reshaped = X_val_aorta[..., np.newaxis]
X_val_brachial_reshaped = X_val_brachial[..., np.newaxis]

aorta_input = Input(shape=(X_train_aorta_reshaped.shape[1], 1), name='aorta_input')
brachial_input = Input(shape=(X_train_brachial_reshaped.shape[1], 1), name='brachial_input')

aorta_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(aorta_input)
brachial_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(brachial_input)

combined = Concatenate()([aorta_lstm, brachial_lstm])

combined = Dropout(0.3)(combined)

output = Dense(6, activation='softmax')(combined)

model = Model(inputs=[aorta_input, brachial_input], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    [X_train_aorta_reshaped, X_train_brachial_reshaped],
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val),
    callbacks=[early_stopping]
)

val_loss, val_accuracy = model.evaluate([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val)
print(f'Validation Accuracy: {val_accuracy:.4f}')

val_predictions = model.predict([X_val_aorta_reshaped, X_val_brachial_reshaped])

predicted_classes = np.argmax(val_predictions, axis=1)

cm = confusion_matrix(y_val, predicted_classes)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

model.save("pressure_model.h5")
