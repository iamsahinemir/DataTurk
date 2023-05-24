import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Load the data
df = pd.read_csv('train.csv', header=None, low_memory=False)

# Drop columns with high percentage of missing values (%50)
df = df.loc[:, (df.isnull().sum() / len(df) * 100 <= 50)]

# Fill missing values in numeric columns with their respective mean value
numeric_cols = df.select_dtypes(include=['int', 'float']).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))

# Selecting columns with non-numeric data type
non_numeric_cols = df.select_dtypes(exclude=['int', 'float']).columns

# Iterating over the selected columns and checking for missing values
for col in non_numeric_cols:
    if df[col].isnull().sum() > 0:
        # Computing the frequency distribution of column values
        col_freq = df[col].value_counts(normalize=True)
        #in order of frequency, the first value is the most frequently repeated value
        top_freq = col_freq.iloc[0]
        # Checking if the difference between the top two frequency values is less than 10%
        second_freq = col_freq.iloc[1] if len(col_freq) > 1 else 0
        if top_freq - second_freq < 0.1:

            # If the top two frequency values are close, fill the missing values with the two most frequent values
            top2_freq = col_freq.iloc[:2]
            fill_values = top2_freq.index.tolist()
            fill_probs = top2_freq.values
            fill_probs = fill_probs / fill_probs.sum()  # Olasılıkları normalize etme

            # Olasılıkları normalize ettikten sonra, eksik değerleri doldurma
            missing_indices = df[col].isnull()  # Eksik değerlere karşılık gelen indeksleri alın

            # Normalize edilmiş olasılıkları eksik değer sayısı kadar tekrarlayın
            random_choices = np.random.choice(fill_values, size=missing_indices.sum(), p=fill_probs.repeat(fill_values.size))

            # Eksik değerleri doldurma
            df.loc[missing_indices, col] = random_choices

            #fill_probs = top2_freq.values
            #df[col].fillna(pd.Series(np.random.choice(fill_values, size=len(df[col]), p=fill_probs)), inplace=True)
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
print(new_column_names)

# Ordinal encoding for categorical features
ordinal_encoder = OrdinalEncoder()
df.iloc[:, :-1] = ordinal_encoder.fit_transform(df.iloc[:, :-1])

# Splitting the data into training and testing sets
X = df.iloc[:, :-1].values  # Extract the values from the DataFrame with the values property to get the NumPy arrays
X = ordinal_encoder.fit_transform(X).astype('int') #independent variables
y = df.iloc[:, -1] #dependent variables
#stratify = y ensures that the overall label ratio is maintained on test and train data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,stratify=y,random_state=44)
print(y)

def evaluate(X_train, X_test, y_train, y_test): #X represents train data and y represents test data.
    model1 = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model2 = RandomForestClassifier(class_weight={0: 1, 1: 10})
    model3 = ExtraTreesClassifier()
    model4 = GradientBoostingClassifier()

    model_name_list = ['XGB Classifier', 'Random Forest', 'Extra Trees', 'Gradient Boosted']
    results = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'cohen_kappa'], index=model_name_list)

    for i, model in enumerate([model1, model2, model3, model4]):
        print(model)
        model.fit(X_train, y_train)
        test_probs = model.predict_proba(X_test)[:, 1]

        # Apply threshold of 0.3 to get the binary predictions
        test_predictions = np.where(test_probs > 0.3, 1, 0)
        print(test_predictions)


        #df_res = pd.DataFrame(test_predictions, columns=['kaggle_id','target'],index=False)
        if model == model4:

            name = '/Users/hasancoskun/desktop/machine-learning-project/DataTürk/DataTurk/' + model_name_list[i] + '.csv'
            #df_res.to_excel(name)
            df_res = pd.DataFrame(test_predictions, columns = ['target'])
            df_res.index.names = ['kaggle_id']
            df_res.to_csv(name)



        # Ensure the target is binary for ROC AUC
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)

        accuracy = accuracy_score(y_test, test_predictions)
        precision = precision_score(y_test, test_predictions, average='weighted')
        recall = recall_score(y_test, test_predictions, average='weighted')
        f1 = f1_score(y_test, test_predictions, average=None)
        _auc = roc_auc_score(y_test_bin, test_probs)
        kappa = cohen_kappa_score(y_test, test_predictions)

        results.loc[model_name_list[i]] = [accuracy, precision, recall, f1, _auc, kappa]

    return results

evaluation_results = evaluate(X_train, X_test, y_train, y_test)
print(evaluation_results)







