import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_csv('train.csv', header=None)

print(df.shape)
print(df.isnull().sum())
print(df.isnull().sum().sum())

# Step 2: Drop columns with high percentage of missing values (%50)
missing_percentages = df.isnull().sum() / len(df) * 100
high_missing_cols = missing_percentages[missing_percentages > 50].index
df.drop(high_missing_cols, axis=1, inplace=True)

print(df.shape)
print(df.isnull().sum().sum())

numeric_cols = df.select_dtypes(include=['int', 'float']).columns

# Select columns with missing values less than or equal to 50%
low_missing_cols = missing_percentages[missing_percentages <= 50].index.intersection(numeric_cols)

# Iterate over the selected columns and fill missing values with their respective mean value
for col in low_missing_cols:
    col_mean = df[col].mean()
    df[col].fillna(col_mean, inplace=True)

# Selecting columns with non-numeric data type
non_numeric_cols = df.select_dtypes(exclude=['int', 'float']).columns

# Iterating over the selected columns and checking for missing values
for col in non_numeric_cols:
    if df[col].isnull().sum() > 0:
        # Computing the frequency distribution of column values
        col_freq = df[col].value_counts(normalize=True)
        top_freq = col_freq.iloc[0]
        # Checking if the difference between the top two frequency values is less than 10%
        second_freq = col_freq.iloc[1] if len(col_freq) > 1 else 0
        if top_freq - second_freq < 0.1:

            # If the top two frequency values are close, fill the missing values with the two most frequent values
            top2_freq = col_freq.iloc[:2]
            fill_values = top2_freq.index.tolist()
            fill_probs = top2_freq.values
            df[col].fillna(pd.Series(np.random.choice(fill_values, size=len(df[col]), p=fill_probs)), inplace=True)
        else:
            
            # Otherwise, fill the missing values with the mode value
            col_mode = df[col].mode()[0]
            df[col].fillna(col_mode, inplace=True)

# Check for missing values in the DataFrame and binary label column
print("Total number of missing values in DataFrame:", df.isnull().sum().sum())
print("Total number of missing values in binary_label column:", df.iloc[:, -1].isnull().sum())
print(f"Non-numerical columns number is: {len(non_numeric_cols)}")
print(f"Numerical columns number is: {len(numeric_cols)}")
print(f"Total number of rows in the DataFrame: {len(df)}")
print(f"Total number of columns in the DataFrame: {len(df.columns)}")

new_column_names = [str(i) for i in range(df.shape[1])]
df.columns = new_column_names

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
X = df.iloc[:, :-1].values  # Extract the values from the DataFrame
X = ordinal_encoder.fit_transform(X).astype('float')
df.iloc[:, :-1] = X
df.info()
corr_matrix = df.corr()

# Print the correlation matrix
print(corr_matrix)

# Compute correlation matrix
corr = df.corr()
print(corr)

# Generate heatmap
rel_cols = df[['26', '53', '56', '58', '71', '74', '78']].corr()

sns.heatmap(rel_cols, cmap='coolwarm', annot=True)
plt.show()

"""This corelation information can be useful in feature selection, identifying multicollinearity, and understanding the relationships between variables in the dataset."""

# Select the first numerical column
num_col = df.iloc[:, 0]

# Plot a histogram
plt.figure(figsize=(10,5))
plt.hist(num_col, bins=100)
#plt.title('Histogram of Column 0')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()


num_col = df.iloc[:, 15]

# Plot a histogram
plt.figure(figsize=(10,5))
plt.hist(num_col, bins=100)
#plt.title('Histogram of Column 15')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

num_col1 = df.iloc[:, 50]
num_col2 = df.iloc[:, 64]

# Plot a scatter plot
plt.figure(figsize=(10,5))
plt.scatter(num_col1, num_col2)
#plt.title('Scatter Plot of Columns 50 and 64')
plt.xlabel('Column 0')
plt.ylabel('Column 1')
plt.show()

rel_cols =df[['26','53','56','58','71','74','78']].corr()
sns.heatmap(rel_cols, cmap='coolwarm', annot=True)

from sklearn.model_selection import train_test_split

X = df.drop(df.columns[-1], axis=1)
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#0.7 Train Size

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Training data set size:", X_train.shape)
print("Test data set size:", X_test.shape)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
def evaluate(X_train, X_test, y_train, y_test):
    model1 = XGBClassifier()
    model2 = RandomForestClassifier()
    model3 = ExtraTreesClassifier()
    model4 = GradientBoostingClassifier()

    model_name_list = ['XGB Classifier','Random Forest', 
                       'Extra Trees', 'Gradient Boosted']
    results = pd.DataFrame(columns=['precision', 'recall', 'f1-score', 'support'], index=model_name_list)

    for i, model in enumerate([model1, model2, model3, model4]):
        print(model)
        model.fit(X_train, y_train)
        test_predictions = model.predict(X_test)

        report = classification_report(y_test, test_predictions, output_dict=True)
        results.loc[model_name_list[i]] = report['weighted avg']

    return results

evaluation_results = evaluate(X_train, X_test, y_train, y_test)
print(evaluation_results)
