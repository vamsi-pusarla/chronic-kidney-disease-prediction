import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

warnings.filterwarnings("ignore")

# Importing dataset
df = pd.read_csv("kidney_disease_dataset.csv")
df.drop('id', axis=1, inplace=True)
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
              'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
              'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
              'anemia', 'class']

# Extracting categorical and numerical columns
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

# Replace incorrect values
df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})

# Checking for null values
df.isna().sum().sort_values(ascending=False)

# Impute null values
def impute_null(col):
    impute = KNNImputer(n_neighbors=3)
    col_data = df[col].values.reshape(-1, 1)
    df[col] = impute.fit_transform(col_data)

for col in num_cols:
    impute_null(col)

# Mode imputation for categorical features
def impute_mode(col):
    imputer = SimpleImputer(strategy='most_frequent')
    col_data = df[col].values.reshape(-1, 1)
    df[col] = imputer.fit_transform(col_data)

for col in cat_cols:
    impute_mode(col)

# Encoding independent variables
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Dividing into independent and dependent variables
ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]

scaler = MinMaxScaler()
scaler.fit(X)
new_features = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(new_features)
# Explained variance ratio
print(X_pca)
explained_variance_ratio = pca.explained_variance_ratio_
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# Plotting explained variance ratio
plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.5, align='center')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio by Principal Components')
plt.show()

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show()

# Initializing the ANN
model_S = Sequential()

# Adding the input layer and the first hidden layer
model_S.add(Dense(units=32, kernel_initializer='uniform', activation='relu', input_dim=24))
model_S.add(Dropout(rate=0.1))

# Adding the second hidden layer
model_S.add(Dense(units=16, kernel_initializer='uniform', activation='relu'))
model_S.add(Dropout(rate=0.1))

# Adding the third hidden layer
model_S.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))
model_S.add(Dropout(rate=0.1))

# Adding the output layer
model_S.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
model_S.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_S.summary()
plot_losses = PlotLosses()
history = model_S.fit(X_train, y_train, epochs=50, batch_size=8, verbose=2, validation_split=0.2, callbacks=[plot_losses])
score1 = model_S.evaluate(X_train, y_train, verbose=0)
print("\nTrain accuracy: %.1f%%" % (100.0 * score1[1]))
'''with open('etc1.pkl', 'wb') as file:
    pickle.dump(etc, file)
model=pickle.load(open('etc1.pkl','rb'))'''