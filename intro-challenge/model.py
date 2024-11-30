import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

aorta_test_df = pd.read_csv('aortaP_test_data.csv')  # Replace with actual test file path
brachial_test_df = pd.read_csv('brachP_test_data.csv')  # Replace with actual test file path


def handle_nans_knn(df, dataset_name, n_neighbors=5):
    # Create a KNN imputer instance with specified number of neighbors
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    # Perform the imputation
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

    # Count the number of NaNs after imputation to verify
    nan_count_after = df_imputed.isna().sum().sum()
    print(f"{dataset_name} - Total NaNs after KNN imputation: {nan_count_after}")

    return df_imputed


# Load your datasets
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')

# Handle NaNs using derivative-based approximation
aorta_df = handle_nans_knn(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans_knn(brachial_df, 'Brachial Dataset')

# Print the updated dataframes to check the changes
print(aorta_df.head())
print(brachial_df.head())

# Check if 'target' exists and split it out
if 'target' in aorta_df.columns:
    aorta_targets = aorta_df['target']
    aorta_df = aorta_df.drop(columns=['target'])

if 'target' in brachial_df.columns:
    brachial_targets = brachial_df['target']
    brachial_df = brachial_df.drop(columns=['target'])

# Ensure targets are the same length and represent the same people
assert len(aorta_targets) == len(brachial_targets), "The number of targets in both datasets must be the same."

# Normalize the datasets (fit scaler on training data only)
scaler = StandardScaler()
aorta_normalized = scaler.fit_transform(aorta_df)  # Train the scaler on the training data (without target column)
brachial_normalized = scaler.fit_transform(brachial_df)  # Train the scaler on the training data (without target column)

# Verify that the lengths of the normalized arrays match the target array length
print(f"Shape of aorta_normalized: {aorta_normalized.shape}")
print(f"Shape of brachial_normalized: {brachial_normalized.shape}")

# Ensure the shapes match before splitting
assert aorta_normalized.shape[0] == len(aorta_targets), "Mismatch between Aorta features and targets length."
assert brachial_normalized.shape[0] == len(brachial_targets), "Mismatch between Brachial features and targets length."

# Split the training data into train and validation sets
X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, aorta_targets, test_size=0.2, random_state=42)

# Reshape the input data to fit the LSTM input format (samples, timesteps, features)
X_train_aorta_reshaped = X_train_aorta[..., np.newaxis]
X_train_brachial_reshaped = X_train_brachial[..., np.newaxis]
X_val_aorta_reshaped = X_val_aorta[..., np.newaxis]
X_val_brachial_reshaped = X_val_brachial[..., np.newaxis]

# Build the model
aorta_input = Input(shape=(X_train_aorta_reshaped.shape[1], 1), name='aorta_input')
brachial_input = Input(shape=(X_train_brachial_reshaped.shape[1], 1), name='brachial_input')

aorta_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(aorta_input)
brachial_lstm = LSTM(units=64, activation='tanh', return_sequences=False)(brachial_input)

combined = Concatenate()([aorta_lstm, brachial_lstm])

combined = Dropout(0.3)(combined)

output = Dense(6, activation='softmax')(combined)

model = Model(inputs=[aorta_input, brachial_input], outputs=output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit([X_train_aorta_reshaped, X_train_brachial_reshaped], y_train, epochs=20, batch_size=32, validation_data=([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val))

# Evaluate the model
val_loss, val_accuracy = model.evaluate([X_val_aorta_reshaped, X_val_brachial_reshaped], y_val)
print(f'Validation Accuracy: {val_accuracy:.4f}')

# Handle missing values in the test set, similar to the training set
aorta_test_df = handle_nans_knn(aorta_test_df, 'Aorta Test Dataset')
brachial_test_df = handle_nans_knn(brachial_test_df, 'Brachial Test Dataset')

# Normalize the test data with the scaler used to fit the training data
# Apply the same scaler to the test data (DO NOT fit the scaler again, just transform)
aorta_test_normalized = scaler.transform(aorta_test_df)  # Transform the test data
brachial_test_normalized = scaler.transform(brachial_test_df)  # Transform the test data

# Reshape the test data to fit the LSTM input format (samples, timesteps, features)
aorta_test_reshaped = aorta_test_normalized[..., np.newaxis]  # Add the new axis for LSTM
brachial_test_reshaped = brachial_test_normalized[..., np.newaxis]  # Add the new axis for LSTM

# Make predictions on the test dataset
predictions = model.predict([aorta_test_reshaped, brachial_test_reshaped])

# Convert predictions to encoded class (integer values)
predicted_classes = np.argmax(predictions, axis=1)

# Save the predictions in the required JSON format
output_dict = {i: int(predicted_classes[i]) for i in range(len(predicted_classes))}

# Save the results to a JSON file
with open('predictions.json', 'w') as json_file:
    json.dump(output_dict, json_file)

print("Predictions saved to 'predictions.json'")
