#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.impute import KNNImputer

# Load the datasets
aorta_df = pd.read_csv('aortaP_test_data.csv')
brachial_df = pd.read_csv('brachP_test_data.csv')

# Display the first few rows of each dataset
print(aorta_df.head())
print(brachial_df.head())


# In[2]:


# Check the dimensions and columns
print(aorta_df.shape)
print(brachial_df.shape)

# Check the columns of the datasets
print(aorta_df.columns)
print(brachial_df.columns)


# In[3]:


import matplotlib.pyplot as plt

# Plot a single subject's waveform for aorta and brachial pressure
subject_index = 0  # Index of the subject you want to visualize (e.g., first subject)

plt.figure(figsize=(12, 6))

# Plot aorta pressure waveform
plt.subplot(1, 2, 1)
plt.plot(aorta_df.iloc[subject_index, :], label='Aorta Pressure')
plt.title(f"Aorta Pressure - Subject {subject_index}")
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.grid(True)

# Plot brachial pressure waveform
plt.subplot(1, 2, 2)
plt.plot(brachial_df.iloc[subject_index, :], label='Brachial Pressure', color='orange')
plt.title(f"Brachial Pressure - Subject {subject_index}")
plt.xlabel('Time')
plt.ylabel('Pressure')
plt.grid(True)

plt.tight_layout()
plt.show()


# In[4]:


# Plot waveforms for multiple subjects (e.g., first 5 subjects)
plt.figure(figsize=(12, 10))

for i in range(5):  # First 5 subjects
    plt.subplot(5, 2, 2*i + 1)
    plt.plot(aorta_df.iloc[i, :], label=f"Subject {i} Aorta")
    plt.title(f"Aorta Pressure - Subject {i}")
    plt.grid(True)

    plt.subplot(5, 2, 2*i + 2)
    plt.plot(brachial_df.iloc[i, :], label=f"Subject {i} Brachial", color='orange')
    plt.title(f"Brachial Pressure - Subject {i}")
    plt.grid(True)

plt.tight_layout()
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets again (if necessary)
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')


def handle_nans_knn(df, dataset_name, n_neighbors=5):
    # Create a KNN imputer instance with specified number of neighbors
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)

    # Perform the imputation
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

    # Count the number of NaNs after imputation to verify
    nan_count_after = df_imputed.isna().sum().sum()
    print(f"{dataset_name} - Total NaNs after KNN imputation: {nan_count_after}")

    return df_imputed

aorta_df = handle_nans_knn(aorta_df, 'Aorta Dataset')
brachial_df = handle_nans_knn(brachial_df, 'Brachial Dataset')

# Plot waveforms for multiple subjects (e.g., first 5 subjects)
plt.figure(figsize=(12, 10))

for i in range(5):  # First 5 subjects
    plt.subplot(5, 2, 2*i + 1)
    plt.plot(aorta_df.iloc[i, :], label=f"Subject {i} Aorta")
    plt.title(f"Aorta Pressure - Subject {i}")
    plt.grid(True)

    plt.subplot(5, 2, 2*i + 2)
    plt.plot(brachial_df.iloc[i, :], label=f"Subject {i} Brachial", color='orange')
    plt.title(f"Brachial Pressure - Subject {i}")
    plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate the median pressure for each subject (across all columns)
aorta_median = aorta_df.median(axis=1)  # Median for each subject
brachial_median = brachial_df.median(axis=1)  # Median for each subject

# Plot histogram to compare the median pressures
plt.figure(figsize=(10, 6))

# Plotting both aorta and brachial median pressures on the same histogram
plt.hist(aorta_median, bins=20, alpha=0.5, label='Aorta Pressure', color='blue')
plt.hist(brachial_median, bins=20, alpha=0.5, label='Brachial Pressure', color='orange')

# Adding labels and title
plt.title('Histogram of Median Pressures for Aorta and Brachial Encodings')
plt.xlabel('Median Pressure')
plt.ylabel('Frequency')
plt.legend()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets again (if necessary)
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')

# Fill missing values with the median of each column
aorta_df = aorta_df.apply(lambda col: col.fillna(col.median()), axis=0)
brachial_df = brachial_df.apply(lambda col: col.fillna(col.median()), axis=0)

# Assuming the 'target' columns are named 'target' in both datasets. If different, adjust column names
aorta_target = aorta_df['target']  # Replace 'target' with the actual column name if different
brachial_target = brachial_df['target']  # Same here

# Plot histogram to compare the target pressures
plt.figure(figsize=(10, 6))

# Plotting both aorta and brachial target pressures on the same histogram
plt.hist(aorta_target, bins=20, alpha=0.5, label='Aorta Target Pressure', color='blue')
plt.hist(brachial_target, bins=20, alpha=0.5, label='Brachial Target Pressure', color='orange')

# Adding labels and title
plt.title('Histogram of Target Pressures for Aorta and Brachial Encodings')
plt.xlabel('Pressure')
plt.ylabel('Frequency')
plt.legend()

plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
aorta_df = pd.read_csv('aortaP_train_data.csv')
brachial_df = pd.read_csv('brachP_train_data.csv')

# Fill missing values with the median of each column (optional, depends on your data)
aorta_df = aorta_df.apply(lambda col: col.fillna(col.median()), axis=0)
brachial_df = brachial_df.apply(lambda col: col.fillna(col.median()), axis=0)

# Assuming 'target' is the column representing the categorical target variable in both datasets
# Replace 'target' with the actual column name if it's different
aorta_target = aorta_df['target']  # Replace 'target' with the actual target column name
brachial_target = brachial_df['target']  # Same here

# Define the relevant pressure columns for both datasets (excluding 'target')
aorta_pressure_columns = [col for col in aorta_df.columns if col != 'target']
brachial_pressure_columns = [col for col in brachial_df.columns if col != 'target']

# Calculate the median for each target group across all pressure columns
aorta_medians = aorta_df.groupby('target')[aorta_pressure_columns].median()  # Median for each target group in aorta
brachial_medians = brachial_df.groupby('target')[brachial_pressure_columns].median()  # Median for each target group in brachial

# Now calculate the median of the medians for each target group (i.e., median of all pressure columns for each target)
aorta_target_medians = aorta_medians.median(axis=1)  # Median of all pressure columns for each target group
brachial_target_medians = brachial_medians.median(axis=1)  # Same for brachial

# Plotting the medians for comparison
plt.figure(figsize=(12, 6))

# Plotting for aorta and brachial medians for each target group
x = range(len(aorta_target_medians))  # X-axis positions for the target groups (0, 1, 2, etc.)

# Plotting the bar charts for aorta and brachial medians
plt.bar(x, aorta_target_medians, width=0.4, label='Aorta Pressure', color='blue', align='center')
plt.bar(x, brachial_target_medians, width=0.4, label='Brachial Pressure', color='orange', align='edge')

# Adding labels and title
plt.title('Median of Medians for Aorta and Brachial Pressure by Target')
plt.xlabel('Target Group')
plt.ylabel('Median Pressure')
plt.xticks(x, aorta_target_medians.index)  # Set target values as x-axis labels (0, 1, 2, etc.)
plt.legend()

plt.show()









