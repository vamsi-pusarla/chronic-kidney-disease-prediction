#importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
warnings.filterwarnings("ignore")
#importing dataset
df=pd.read_csv("kidney_disease_dataset.csv")
df.drop('id', axis = 1, inplace = True)
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'aanemia', 'class']

# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']
# replace incorrect values

df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')

df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})
# checking for null values

df.isna().sum().sort_values(ascending = False)
from sklearn.impute import KNNImputer
def impute_null(col):
    impute = KNNImputer(n_neighbors=3)
    col_data = df[col].values.reshape(-1, 1)
    df[col] = impute.fit_transform(col_data)
for col in num_cols:
    impute_null(col)
#mode imputation for categorical features
def impute_mode(col):
    imputer = SimpleImputer(strategy='most_frequent')
    col_data = df[col].values.reshape(-1, 1)
    df[col] = imputer.fit_transform(col_data)
for col in cat_cols:
    impute_mode(col)
#encoding independent variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
#dividing into independent and dependent variables
ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler = MinMaxScaler()
scaler.fit(X)
new_features = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#extra tree
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier(random_state=67)
etc.fit(X_train, y_train)

# accuracy score, confusion matrix and classification report of extra trees classifier

etc_acc = accuracy_score(y_test, etc.predict(X_test))

print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train))}")
print(f"Test Accuracy of Extra Trees Classifier is {etc_acc} \n")
print(f"Classification Report :- \n {classification_report(y_test, etc.predict(X_test))}")
fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, etc.predict(X_test)))
plt.show()
with open('etc1.pkl', 'wb') as file:
    pickle.dump(etc, file)
model=pickle.load(open('etc1.pkl','rb'))