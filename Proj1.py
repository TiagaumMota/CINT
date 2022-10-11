import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from scipy import stats

def remove_outliers1(df):
    outliers=[]

    for index in range(2,11):
        column=df.iloc[:,index].to_numpy()
        q3, q1 = np.percentile(column, [75 ,25])
        iqr=q3-q1

        upper=np.where(column >= q3 + (1.5*iqr))
        lower=np.where(column <= q1 - (1.5*iqr))

        outliers.append(upper[0])
        outliers.append(lower[0])
    
    print(outliers)


def remove_outliers(df):
        threshold = 6
        df_new=df

        for index in range(2,11):
            column=df.iloc[:,index].to_numpy()
            z = np.abs(stats.zscore(column))
            
            outliers=np.where(z > threshold)
            df_new.drop(outliers[0], inplace=True)

        return df_new
    
    

#Read data from file
df = pd.read_csv('Proj1_Dataset.csv',decimal='.', sep=',' )

#Include the binary classification output 
df['MoreThan2']=(df['Persons']>2).astype(int)

#Data Preprocessing
#Filling all the missing values
clean_df=df.interpolate(option='time')

#Find some outliers and remove them 
clean_df=remove_outliers(clean_df)
print(clean_df.shape)

#Resolve inconsistencies?
inconsistencies=np.where((clean_df['Persons']==0 ) & (clean_df['PIR1']+clean_df['PIR2']!=0))
#print(inconsistencies[0].shape)
#print(clean_df.iloc[inconsistencies[0], :])

#Baseline MLP
print("Problem a)")

X=  clean_df.iloc[:, 2:11].to_numpy()
Y=np.reshape(clean_df['MoreThan2'].to_numpy(), (-1,1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2 , shuffle = False)

clf = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',solver='sgd',random_state=1, max_iter=300)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test).reshape(-1,1)

print("Accuracy:", accuracy_score(y_test, y_pred ))
print("Pecision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

print("Problem b)")
Y=np.reshape(clean_df['Persons'].to_numpy(), (-1,1))

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2 , shuffle = False)

clf = MLPClassifier(hidden_layer_sizes=(10,),activation='logistic',solver='sgd',random_state=1, max_iter=300)
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test).reshape(-1,1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Pecision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1-Score:", f1_score(y_test, y_pred, average='macro'))
print("Confusion Matrix:", confusion_matrix(y_test, y_pred))

