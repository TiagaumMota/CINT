from distutils.command.clean import clean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from scipy import stats
from sklearn.preprocessing import StandardScaler

def remove_outliers(df):
        threshold = 6
        df_new=df

        for index in range(2,11):
            column = df_new.iloc[:,index].to_numpy()
            z = np.abs(stats.zscore(column))
            
            outliers = np.where(z > threshold)
            df_new.drop(df_new.index[outliers[0]],axis=0, inplace=True)
       
        return df_new
    
    
def print_confM(conf_matrix):
    sns.set(font_scale=1.5)
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')

    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    plt.show()

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



#Read data from file
df = pd.read_csv('Proj1_Dataset.csv',decimal='.', sep=',' )

#Include the binary classification output 
df['MoreThan2'] = (df['Persons']>2).astype(int)

#Data Preprocessing
#Filling all the missing values
clean_df = df.interpolate(option='time')

#Find some outliers and remove them 
clean_df = remove_outliers(clean_df)

#Include daytime
clean_df['DayTime'] = ((clean_df['Time'] >= '07:51:00') & (clean_df['Time'] <= '17:40:00')).astype(int)


#........................................................Binary.................................................................
fuzzy_df = clean_df.drop(['Date', 'Time', 'S1Temp', 'S2Temp', 'S3Temp','CO2', 'PIR1', 'PIR2', 'Persons'],  axis=1)


Y = fuzzy_df.pop('MoreThan2')
X = fuzzy_df

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

pipeline = imbpipeline(steps = [['smote', SMOTE(random_state=42)],
                                ['scaler', StandardScaler()],
                                ['classifier', MLPClassifier(random_state=42)]])

stratified_kfold = StratifiedKFold(n_splits=4,
                                       shuffle=True,
                                       random_state=42)

param_grid = {'classifier__solver': ['adam'],
        'classifier__max_iter': [ 300, 500, 1000],
        'classifier__hidden_layer_sizes': [(10,)],
        'classifier__activation': ['relu'],
        'classifier__alpha': [0.00001, 0.0001, 0.001]
        }

grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='f1',
                           cv=stratified_kfold,
                           n_jobs=-1)

grid_search.fit(X_train, y_train)
 
print(grid_search.best_params_)
y_pred=grid_search.predict(X_test).reshape(-1,1)
print_results(y_test, y_pred)

"""
sns.set(font_scale=1.5)
ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues', fmt='g')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
plt.show()
"""