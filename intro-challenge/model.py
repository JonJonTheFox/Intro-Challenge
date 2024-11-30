import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import StandardScaler
import numpy as np

aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')

def handle_nans(df, dataset_name):
    nan_count = df.isna().sum().sum()
    print(f"{dataset_name} - Total NaNs: {nan_count}")

    df_filled = df.apply(lambda row: row.fillna(row.median()), axis=1)
    return df_filled

aorta_df = handle_nans(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans(brachial_df, 'Brachial Dataset')
print(f"aorta_df shape: {aorta_df.shape}")
print(f"brachial_df shape: {brachial_df.shape}")

if 'target' in aorta_df.columns:
    targets = aorta_df['target']
    aorta_df = aorta_df.drop(columns=['target'])

scaler = StandardScaler()
aorta_normalized = scaler.fit_transform(aorta_df)
brachial_normalized = scaler.fit_transform(brachial_df)

X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, targets, test_size=0.4, random_state=42)

print(f"X_train_aorta shape before reshaping: {X_train_aorta.shape}")
print(f"X_train_brachial shape before reshaping: {X_train_brachial.shape}")


X_train_aorta_reshaped = X_train_aorta[..., np.newaxis]
X_train_brachial_reshaped = X_train_brachial[..., np.newaxis]
X_val_aorta_reshaped = X_val_aorta[..., np.newaxis]
X_val_brachial_reshaped = X_val_brachial[..., np.newaxis]

print(f"X_train_aorta shape after reshaping: {X_train_aorta_reshaped.shape}")
print(f"X_train_brachial shape after reshaping: {X_train_brachial_reshaped.shape}")

aorta_input = Input(shape=(X_train_aorta_reshaped.shape[1], 1), name='aorta_input')
brachial_input = Input(shape=(X_train_brachial_reshaped.shape[1], 1), name='brachial_input')

aorta_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(aorta_input)
brachial_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(brachial_input)

combined = Concatenate()([aorta_lstm, brachial_lstm])

combined = Dropout(0.3)(combined)

output = Dense(6, activation='softmax')(combined)

model = Model(inputs=[aorta_input, brachial_input], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit([X_train_aorta_reshaped, X_train_brachial_reshaped], y_train, epochs=20, batch_size=32, validation_data=([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val))

val_loss, val_accuracy = model.evaluate([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val)
print(f'Validation Accuracy: {val_accuracy:.4f}')
