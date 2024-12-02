import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import xgboost as xgb
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.svm import SVC

aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')
aorta_test_df = pd.read_csv('aortaP_test_data.csv')
brachial_test_df = pd.read_csv('brachP_test_data.csv')

def apply_log_transformation(df):
    df_log_transformed = df.apply(lambda x: np.log1p(x) if x.min() >= 0 else x)
    return df_log_transformed

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
    df_aorta['max_pressure'] = df_aorta.max(axis=1)
    df_brachial['max_pressure'] = df_brachial.max(axis=1)
    df_aorta['min_pressure'] = df_aorta.min(axis=1)
    df_brachial['min_pressure'] = df_brachial.min(axis=1)

    return df_aorta, df_brachial

def add_pressure_slope(df_aorta, df_brachial):
    df_aorta['pressure_slope'] = df_aorta.diff(axis=1).mean(axis=1)
    df_brachial['pressure_slope'] = df_brachial.diff(axis=1).mean(axis=1)
    return df_aorta, df_brachial


def add_fourier_transform(df_aorta, df_brachial, num_components=5):
    def compute_fourier_features(df):
        fft_features = []
        for row in df.values:
            fft_result = np.fft.fft(row)
            magnitudes = np.abs(fft_result[:num_components])
            fft_features.append(magnitudes)
        return np.array(fft_features)

    aorta_fft_features = compute_fourier_features(df_aorta)
    brachial_fft_features = compute_fourier_features(df_brachial)

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
    df_aorta['median_pressure'] = df_aorta.median(axis=1)
    df_brachial['median_pressure'] = df_brachial.median(axis=1)

    df_aorta['pressure_asymmetry'] = (df_aorta['max_pressure'] - df_aorta['min_pressure']) / df_aorta['median_pressure']
    df_brachial['pressure_asymmetry'] = (df_brachial['max_pressure'] - df_brachial['min_pressure']) / df_brachial[
        'median_pressure']

    return df_aorta, df_brachial


def drop_time_series_columns(df):
    time_series_columns = [col for col in df.columns if col.startswith('aorta_t_') or col.startswith('brachial_t_')]
    df = df.drop(columns=time_series_columns)
    return df

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

aorta_df, brachial_df = add_median_difference(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_std(aorta_df, brachial_df)
aorta_df, brachial_df = add_max_min_pressure(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_ratio(aorta_df, brachial_df)
aorta_df, brachial_df = add_moving_average_pressure(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_slope(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_asymmetry(aorta_df, brachial_df)
aorta_df, brachial_df = add_fourier_transform(aorta_df, brachial_df)
aorta_df, brachial_df = add_pressure_skew(aorta_df, brachial_df)

aorta_df = drop_time_series_columns(aorta_df)
brachial_df = drop_time_series_columns(brachial_df)

poly = PolynomialFeatures(degree=2)
aorta_poly = poly.fit_transform(aorta_df)
brachial_poly = poly.fit_transform(brachial_df)

pca = PCA(n_components=5)
aorta_pca = pca.fit_transform(aorta_df)
brachial_pca = pca.fit_transform(brachial_df)


scaler = StandardScaler()
aorta_normalized = scaler.fit_transform(aorta_df)
brachial_normalized = scaler.fit_transform(brachial_df)

X_train_aorta, X_val_aorta, X_train_brachial, X_val_brachial, y_train, y_val = train_test_split(
    aorta_normalized, brachial_normalized, aorta_targets, test_size=0.2, random_state=42)

X_train_aorta_flat = X_train_aorta.reshape(X_train_aorta.shape[0], -1)
X_train_brachial_flat = X_train_brachial.reshape(X_train_brachial.shape[0], -1)
X_val_aorta_flat = X_val_aorta.reshape(X_val_aorta.shape[0], -1)
X_val_brachial_flat = X_val_brachial.reshape(X_val_brachial.shape[0], -1)

X_train_combined = np.concatenate([X_train_aorta_flat, X_train_brachial_flat], axis=1)
X_val_combined = np.concatenate([X_val_aorta_flat, X_val_brachial_flat], axis=1)

xgboost_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    max_depth=15,
    min_child_weight=1,
    gamma=0.05,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.7,
    n_estimators=500,
    reg_lambda=0.5,
    reg_alpha=0.1,
    max_delta_step=1,
)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
logreg_model = LogisticRegression(max_iter=1000, random_state=42)
svc_model = SVC(probability=True, kernel='rbf', random_state=42)

voting_clf = VotingClassifier(estimators=[
    ('xgboost', xgboost_model),
    ('rf', rf_model),
    ('logreg', logreg_model),
    ('svc', svc_model)
], voting='soft')


voting_clf.fit(X_train_combined, y_train)


y_val_pred = voting_clf.predict(X_val_combined)


val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy (Voting Classifier): {val_accuracy:.4f}')


cm = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_val_pred), yticklabels=np.unique(y_val_pred))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Voting Classifier)')
plt.show()


importances = xgboost_model.get_booster().get_score(importance_type='weight')


sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)


for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")


aorta_test_df = handle_nans_knn(aorta_test_df, 'Aorta Test Dataset')
brachial_test_df = handle_nans_knn(brachial_test_df, 'Brachial Test Dataset')


aorta_test_df, brachial_test_df = add_median_difference(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_std(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_max_min_pressure(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_ratio(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_moving_average_pressure(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_slope(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_asymmetry(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_fourier_transform(aorta_test_df, brachial_test_df)
aorta_test_df, brachial_test_df = add_pressure_skew(aorta_test_df, brachial_test_df)

aorta_test_df = drop_time_series_columns(aorta_test_df)
brachial_test_df = drop_time_series_columns(brachial_test_df)


aorta_test_normalized = scaler.transform(aorta_test_df)
brachial_test_normalized = scaler.transform(brachial_test_df)

X_test_aorta_flat = aorta_test_normalized.reshape(aorta_test_normalized.shape[0], -1)
X_test_brachial_flat = brachial_test_normalized.reshape(brachial_test_normalized.shape[0], -1)


X_test_combined = np.concatenate([X_test_aorta_flat, X_test_brachial_flat], axis=1)


test_predictions = xgboost_model.predict(X_test_combined)


output_dict = {i: int(test_predictions[i]) for i in range(len(test_predictions))}

with open('xgboost_predictions.json', 'w') as json_file:
    json.dump(output_dict, json_file)

print("Predictions saved to 'xgboost_predictions.json'")

