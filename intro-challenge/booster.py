import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb
import numpy as np
import json
from matplotlib import pyplot as plt

# Read the datasets
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')
aorta_test_df = pd.read_csv('aortaP_test_data.csv')
brachial_test_df = pd.read_csv('brachP_test_data.csv')

def apply_log_transformation(df):
    # Apply log1p (log(x + 1)) to each column, since we want to handle zeros gracefully
    df_log_transformed = df.apply(lambda x: np.log1p(x) if x.min() >= 0 else x)
    return df_log_transformed

# Preprocessing Functions (same as in your code)
def handle_nans_knn(df, dataset_name, n_neighbors=5):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    nan_count_after = df_imputed.isna().sum().sum()
    print(f"{dataset_name} - Total NaNs after KNN imputation: {nan_count_after}")
    return df_imputed

def add_median_difference(df_aorta, df_brachial):
    aorta_medians = df_aorta.median(axis=1)
    brachial_medians = df_brachial.median(axis=1)
    median_difference = brachial_medians - aorta_medians
    df_aorta['median_difference'] = median_difference
    df_brachial['median_difference'] = median_difference
    return df_aorta, df_brachial

def add_pressure_std(df_aorta, df_brachial):
    df_aorta['pressure_std'] = df_aorta.std(axis=1)
    df_brachial['pressure_std'] = df_brachial.std(axis=1)
    return df_aorta, df_brachial

def add_max_min_pressure(df_aorta, df_brachial):
    # Calculate max and min pressure for each subject (row-wise)
    df_aorta['max_pressure'] = df_aorta.max(axis=1)
    df_brachial['max_pressure'] = df_brachial.max(axis=1)
    df_aorta['min_pressure'] = df_aorta.min(axis=1)
    df_brachial['min_pressure'] = df_brachial.min(axis=1)

    # You can also calculate the range as a new feature: max - min
    #df_aorta['pressure_range'] = df_aorta['max_pressure'] - df_aorta['min_pressure']
    #df_brachial['pressure_range'] = df_brachial['max_pressure'] - df_brachial['min_pressure']

    return df_aorta, df_brachial

def add_pressure_slope(df_aorta, df_brachial):
    df_aorta['pressure_slope'] = df_aorta.diff(axis=1).mean(axis=1)
    df_brachial['pressure_slope'] = df_brachial.diff(axis=1).mean(axis=1)
    return df_aorta, df_brachial


def add_fourier_transform(df_aorta, df_brachial, num_components=5):
    def compute_fourier_features(df):
        # Perform FFT for each row (subject)
        fft_features = []
        for row in df.values:
            # Apply FFT to the pressure data
            fft_result = np.fft.fft(row)
            # Compute the magnitudes of the first few frequency components
            magnitudes = np.abs(fft_result[:num_components])  # Take the first 'num_components' components
            fft_features.append(magnitudes)
        return np.array(fft_features)

    # Compute Fourier features for aorta and brachial datasets
    aorta_fft_features = compute_fourier_features(df_aorta)
    brachial_fft_features = compute_fourier_features(df_brachial)

    # Add the FFT magnitudes as new columns to the datasets
    for i in range(aorta_fft_features.shape[1]):
        df_aorta[f'fft_mag_{i + 1}'] = aorta_fft_features[:, i]
        df_brachial[f'fft_mag_{i + 1}'] = brachial_fft_features[:, i]

    return df_aorta, df_brachial


def add_pressure_ratio(df_aorta, df_brachial):
    df_aorta['pressure_ratio'] = df_brachial.mean(axis=1) / df_aorta.mean(axis=1)
    return df_aorta, df_brachial

def add_vascular_health_index(df_aorta, df_brachial):
    df_aorta['vascular_health_index'] = 0.5 * df_aorta['median_difference'] + 0.3 * df_aorta['pressure_std']
    df_brachial['vascular_health_index'] = 0.5 * df_brachial['median_difference'] + 0.3 * df_brachial['pressure_std']
    return df_aorta, df_brachial

def add_moving_average_pressure(df_aorta, df_brachial, window_size=3):
    df_aorta['moving_avg'] = df_aorta.rolling(window=window_size, axis=1).mean().mean(axis=1)
    df_brachial['moving_avg'] = df_brachial.rolling(window=window_size, axis=1).mean().mean(axis=1)
    return df_aorta, df_brachial

from scipy.stats import skew

def add_pressure_skew(df_aorta, df_brachial):
    df_aorta['pressure_skew'] = df_aorta.apply(lambda row: skew(row), axis=1)
    df_brachial['pressure_skew'] = df_brachial.apply(lambda row: skew(row), axis=1)
    return df_aorta, df_brachial


def add_pressure_asymmetry(df_aorta, df_brachial):
    # Calculate the median pressure for each row (subject) in both datasets
    df_aorta['median_pressure'] = df_aorta.median(axis=1)
    df_brachial['median_pressure'] = df_brachial.median(axis=1)

    # Calculate pressure asymmetry
    df_aorta['pressure_asymmetry'] = (df_aorta['max_pressure'] - df_aorta['min_pressure']) / df_aorta['median_pressure']
    df_brachial['pressure_asymmetry'] = (df_brachial['max_pressure'] - df_brachial['min_pressure']) / df_brachial[
        'median_pressure']

    return df_aorta, df_brachial


# Drop columns matching the pattern 'aorta_t_*' or 'brachial_t_*'
def drop_time_series_columns(df):
    # Use regex to match columns that follow the 'aorta_t_*' or 'brachial_t_*' pattern
    time_series_columns = [col for col in df.columns if col.startswith('aorta_t_') or col.startswith('brachial_t_')]
    df = df.drop(columns=time_series_columns)
    return df

# Preprocess the training data (with the time series columns removed)
aorta_df = handle_nans_knn(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans_knn(brachial_df, 'Brachial Dataset')

if 'target' in aorta_df.columns:
    aorta_targets = aorta_df['target']
    aorta_df = aorta_df.drop(columns=['target'])

if 'target' in brachial_df.columns:
    brachial_targets = brachial_df['target']
    brachial_df = brachial_df.drop(columns=['target'])

aorta_df = apply_log_transformation(aorta_df)
brachial_df = apply_log_transformation(brachial_df)

# Add the engineered features
aorta_df, brachial_df = add_median_difference(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_std(aorta_df, brachial_df)
aorta_df, brachial_df = add_max_min_pressure(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_ratio(aorta_df, brachial_df)
aorta_df, brachial_df = add_moving_average_pressure(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_slope(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_asymmetry(aorta_df, brachial_df)
aorta_df, brachial_df = add_fourier_transform(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_skew(aorta_df, brachial_df)

# Drop time series columns after feature engineering
aorta_df = drop_time_series_columns(aorta_df)
brachial_df = drop_time_series_columns(brachial_df)

poly = PolynomialFeatures(degree=2)
aorta_poly = poly.fit_transform(aorta_df)
brachial_poly = poly.fit_transform(brachial_df)

pca = PCA(n_components=5)
aorta_pca = pca.fit_transform(aorta_df)
brachial_pca = pca.fit_transform(brachial_df)

# Handle target column

# Normalize the datasets (after dropping time columns)
scaler = StandardScaler()
aorta_normalized = scaler.fit_transform(aorta_df)
brachial_normalized = scaler.fit_transform(brachial_df)

# Split the training data into train and validation sets
X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, aorta_targets, test_size=0.2, random_state=42)

# Flatten the features for XGBoost
X_train_aorta_flat = X_train_aorta.reshape(X_train_aorta.shape[0], -1)
X_train_brachial_flat = X_train_brachial.reshape(X_train_brachial.shape[0], -1)
X_val_aorta_flat = X_val_aorta.reshape(X_val_aorta.shape[0], -1)
X_val_brachial_flat = X_val_brachial.reshape(X_val_brachial.shape[0], -1)

# Combine the flattened data for XGBoost
X_train_combined = np.concatenate([X_train_aorta_flat, X_train_brachial_flat], axis=1)
X_val_combined = np.concatenate([X_val_aorta_flat, X_val_brachial_flat], axis=1)

# Train an XGBoost model
xgboost_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    max_depth=15,  # Increase tree depth further
    min_child_weight=1,  # Finer splits
    gamma=0.05,  # More splits
    learning_rate=0.02,  # Smaller step size
    subsample=0.7,  # Use 70% of data for each tree
    colsample_bytree=0.7,  # Use 70% of features for each tree
    n_estimators=500,  # Increase number of trees
    reg_lambda=0.5,  # L2 regularization on leaf weights
    reg_alpha=0.1,  # L1 regularization
    max_delta_step=1,  # Prevent large optimization steps
)

xgboost_model.fit(X_train_combined, y_train)

# Make predictions
y_val_pred = xgboost_model.predict(X_val_combined)

# Calculate accuracy
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy (XGBoost): {val_accuracy:.4f}')

# Generate confusion matrix
cm = confusion_matrix(y_val, y_val_pred)

# Print confusion matrix
print("Confusion Matrix (XGBoost):")
print(cm)

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_val_pred), yticklabels=np.unique(y_val_pred))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (XGBoost)')
plt.show()

# Get feature importances
importances = xgboost_model.get_booster().get_score(importance_type='weight')

# Sort the importances in descending order
sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

# Print the sorted feature importances
for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")


# Preprocess the test data (with the same approach)
aorta_test_df = handle_nans_knn(aorta_test_df, 'Aorta Test Dataset')
brachial_test_df = handle_nans_knn(brachial_test_df, 'Brachial Test Dataset')

# Add engineered features to the test set
aorta_test_df, brachial_test_df = add_median_difference(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_std(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_max_min_pressure(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_ratio(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_moving_average_pressure(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_slope(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_asymmetry(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_fourier_transform(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_skew(aorta_test_df, brachial_test_df)

# Drop the time series columns for the test set
aorta_test_df = drop_time_series_columns(aorta_test_df)
brachial_test_df = drop_time_series_columns(brachial_test_df)

# Normalize the test data
aorta_test_normalized = scaler.transform(aorta_test_df)
brachial_test_normalized = scaler.transform(brachial_test_df)

# Flatten the test features for XGBoost
X_test_aorta_flat = aorta_test_normalized.reshape(aorta_test_normalized.shape[0], -1)
X_test_brachial_flat = brachial_test_normalized.reshape(brachial_test_normalized.shape[0], -1)

# Combine the flattened test data
X_test_combined = np.concatenate([X_test_aorta_flat, X_test_brachial_flat], axis=1)

# Make predictions on the test set
test_predictions = xgboost_model.predict(X_test_combined)

# Save predictions to JSON
output_dict = {i: int(test_predictions[i]) for i in range(len(test_predictions))}

# Save the results in JSON format
with open('xgboost_predictions.json', 'w') as json_file:
    json.dump(output_dict, json_file)

print("Predictions saved to 'xgboost_predictions.json'")

