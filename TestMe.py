from sklearn import model_selection
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

filename = sys.argv[1]

df = pd.read_csv(filename, decimal='.', sep=',' )
df_bi = df.copy()
df_multi = df.copy()

def print_results(y_true, y_pred):

    print("\n\n------------------------------------Results--------------------------------------------")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Pecision:", precision_score(y_true, y_pred, average=None))
    print("Recall:", recall_score(y_true, y_pred, average=None))
    print("Macro-Pecision:", precision_score(y_true, y_pred, average='macro'))
    print("Macro-Recall:", recall_score(y_true, y_pred, average='macro'))
    print("Macro-F1-Score:", f1_score(y_true, y_pred, average='macro'))
    print("Confusion Matrix:", confusion_matrix(y_true, y_pred))
    print("---------------------------------------------------------------------------------------")

Binary_model = pickle.load(open('BinaryModel.sav', 'rb'))
MultiClass_model = pickle.load(open('MultiClassModel.sav', 'rb'))


#**************************BinaryProblem**********************************

#ADD COLUMN MORETHAN2 TO THE DATASET
df_bi['MoreThan2'] = (df['Persons']>2).astype(int)

#DELETE TIME AND DATA
df_bi = df_bi.drop(['Time', 'Date', 'Persons'], axis=1)

#Create x and y
y = df_bi.pop('MoreThan2')
x = df_bi

#Predict and Print Results
y_pred = Binary_model.predict(x).reshape(-1,1)
print_results(y, y_pred)


#**************************MultiClassProblem**********************************

#DELETE TIME AND DATA
df_multi = df_multi.drop(['Time', 'Date'], axis=1)

#Create x and y
y = df_multi.pop('Persons')
x = df_multi

#Predict and Print Results
y_pred = MultiClass_model.predict(x).reshape(-1,1)
print_results(y, y_pred)
