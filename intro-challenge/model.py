import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

# Load your training and test datasets
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')
aorta_test_df = pd.read_csv('aortaP_test_data.csv')  # Replace with actual test file path
brachial_test_df = pd.read_csv('brachP_test_data.csv')  # Replace with actual test file path

# Function to handle NaNs by replacing them with row-wise median
def handle_nans(df, dataset_name):
    nan_count = df.isna().sum().sum()
    print(f"{dataset_name} - Total NaNs: {nan_count}")
    df_filled = df.apply(lambda row: row.fillna(row.median()), axis=1)
    return df_filled

# Handle NaNs for both datasets
aorta_df = handle_nans(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans(brachial_df, 'Brachial Dataset')

# Check if 'target' exists and split it out
if 'target' in aorta_df.columns:
    targets = aorta_df['target']
    aorta_df = aorta_df.drop(columns=['target'])

# Normalize the datasets
scaler = StandardScaler()
aorta_normalized = scaler.fit_transform(aorta_df)
brachial_normalized = scaler.fit_transform(brachial_df)

# Split the training data into train and validation sets
X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, targets, test_size=0.4, random_state=42)

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

# Align test set columns to match training set (remove target column if it exists)
# Align the columns of the test set to match the columns of the training set
# First, get the columns for the aorta and brachial training data
aorta_columns = aorta_df.columns
brachial_columns = brachial_df.columns

# Now align the test data columns to the training data columns
aorta_test_df = aorta_test_df[aorta_columns]  # Align aorta test data
brachial_test_df = brachial_test_df[brachial_columns]  # Align brachial test data

# If there are columns in the test set that are not in the training set, you can fill them with NaNs or remove them
# Example: drop columns in test set that are not in training
aorta_test_df = aorta_test_df[aorta_columns.intersection(aorta_test_df.columns)]
brachial_test_df = brachial_test_df[brachial_columns.intersection(brachial_test_df.columns)]

# Handle missing values in the test set, similar to the training set
aorta_test_df = handle_nans(aorta_test_df, 'Aorta Test Dataset')
brachial_test_df = handle_nans(brachial_test_df, 'Brachial Test Dataset')

# Normalize the test data with the scaler used to fit the training data
aorta_test_normalized = scaler.transform(aorta_test_df)
brachial_test_normalized = scaler.transform(brachial_test_df)

# Reshape the test data
aorta_test_reshaped = aorta_test_normalized[..., np.newaxis]
brachial_test_reshaped = brachial_test_normalized[..., np.newaxis]

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

