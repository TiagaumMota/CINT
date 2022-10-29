import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from scipy import stats

import skfuzzy as fuzz
from skfuzzy import control as ctrl


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

    #ax.xaxis.set_ticklabels(['0','2', '3'])
    #ax.yaxis.set_ticklabels(['0','2','3'])

    plt.show()



#Read data from file
df = pd.read_csv('Proj1_Dataset.csv',decimal='.', sep=',' )

#Include the binary classification output 
df['MoreThan2'] = (df['Persons']>2).astype(int)

#Data Preprocessing
#Filling all the missing values
clean_df = df.interpolate(option='time')

#Find some outliers and remove them 
clean_df = remove_outliers(clean_df)


train_ratio = 0.80
test_ratio = 0.20

fuzzy_df = clean_df.drop(['Date','S1Temp', 'S2Temp', 'S3Temp'],  axis=1)
fuzzy_df['DayTime'] = ((fuzzy_df['Time'] >= '07:51:00') & (fuzzy_df['Time'] <= '17:40:00')).astype(int)

"""
ADD CLAUDY DAY FEATURE
"""

#fuzzy_df['ClaudyDay']= (((fuzzy_df['Time'] >= '07:51:00') & (fuzzy_df['Time'] <= '17:40:00')) & (fuzzy_df['S1Light']<200)& (fuzzy_df['S2Light']<200)& (fuzzy_df['S3Light']<200)).astype(int)
#fuzzy_df.to_csv('ChangedData1.csv', decimal=',', sep=';', index=False)


"""
#ADD CO2 Derivative
"""

"""CO2_diff = fuzzy_df[["CO2"]]
CO2_diff = CO2_diff.diff(periods=1)
CO2_diff = CO2_diff.fillna(0.0)
CO2_diff = CO2_diff.astype(int)
fuzzy_df.insert(loc=len(fuzzy_df.columns)-1, column = "CO2_dif", value = CO2_diff)"""

##########################PLOTS#######################################
"""plt.figure()
plt.subplot(3,1,1)
plt.scatter(fuzzy_df['S1Light'],fuzzy_df['Persons'])
plt.title("S1")


plt.subplot(3,1,2)
plt.scatter(fuzzy_df['S2Light'],fuzzy_df['Persons'])
plt.title("S2")


plt.subplot(3,1,3)
plt.scatter(fuzzy_df['S3Light'],fuzzy_df['Persons'])
plt.title("S3")
plt.show()

plt.figure()
plt.subplot(2,1,1)
plt.scatter(fuzzy_df['PIR1'],fuzzy_df['Persons'])
plt.title("PIR1")


plt.subplot(2,1,2)
plt.scatter(fuzzy_df['PIR2'],fuzzy_df['Persons'])
plt.title("PIR2")
plt.show()

plt.subplot(1,1,1)
plt.scatter(fuzzy_df['CO2_dif'],fuzzy_df['Persons'])
plt.title("CO2_dif")
plt.show()"""

"""plot_df = fuzzy_df.loc[(fuzzy_df['S1Light'] < 400) & (fuzzy_df['S1Light'] > 100) & (fuzzy_df['S2Light'] > 300)]

sns.relplot(x="S1Light", y = "S2Light", hue = "MoreThan2", data = plot_df, size = "MoreThan2", palette = 'bright', kind = "scatter", height = 5, aspect = 1)
plt.show()

sns.relplot(x="S1Light", y = "S2Light", hue = "MoreThan2", data = fuzzy_df, size = "MoreThan2", palette = 'bright', kind = "scatter", height = 5, aspect = 1)
plt.show()
sns.relplot(x="S1Light", y = "S3Light", hue = "MoreThan2", data = fuzzy_df, size = "MoreThan2", palette = 'bright', kind = "scatter", height = 5, aspect = 1)
plt.show()
sns.relplot(x="S2Light", y = "S3Light", hue = "MoreThan2", data = fuzzy_df, size = "MoreThan2", palette = 'bright', kind = "scatter", height = 5, aspect = 1)
plt.show()"""
#**************************************** IMPLEMENTAR FUZZY ****************************************** 

"""
print(x_vector['S1Light'].min()) #0
print(x_vector['S2Light'].min()) #0
print(x_vector['S3Light'].min()) #0

print(x_vector['S1Light'].max()) #489
print(x_vector['S2Light'].max()) #516
print(x_vector['S3Light'].max()) #560
"""

S1Light = ctrl.Antecedent(np.arange(0, 490, 1), 'S1Light') #input1
S2Light = ctrl.Antecedent(np.arange(0, 517, 1), 'S2Light') #input2
S3Light = ctrl.Antecedent(np.arange(0, 561, 1), 'S3Light') #input3
DayTime = ctrl.Antecedent(np.arange(0, 2, 1), 'DayTime') #input4

#Only For Testing Reasons
PIR1 = ctrl.Antecedent(np.arange(0, 2, 1), 'PIR1') #input5
PIR2 = ctrl.Antecedent(np.arange(0, 2, 1), 'PIR2') #input6
Claudy = ctrl.Antecedent(np.arange(0, 2, 1), 'Claudy') #input4


MoreThan2 = ctrl.Consequent(np.arange(0, 2, 1), 'MoreThan2') #Output

#Linguistic terms for input1
S1Light['low'] = fuzz.trapmf(S1Light.universe, [0,0,100,200])
S1Light['medium'] = fuzz.trapmf(S1Light.universe, [100,200,270,350])
S1Light['high'] = fuzz.trapmf(S1Light.universe, [270,350,490,490])

"""
S1Light['low'] = fuzz.trapmf(S1Light.universe, [0,0,270,350])
S1Light['high'] = fuzz.trapmf(S1Light.universe, [270,350,490,490])"""


#Linguistic terms for input2
S2Light['low'] = fuzz.trapmf(S2Light.universe, [0,0,200,300])
S2Light['high'] = fuzz.trapmf(S2Light.universe, [200,300,517,517])

#Linguistic terms for input3
S3Light['low'] = fuzz.trapmf(S3Light.universe, [0,0,200,300])
S3Light['high'] = fuzz.trapmf(S3Light.universe, [200,300,500,560])

#Linguistic terms for input4
DayTime.automf(2, names = ['No','Yes'])

#Linguistic terms for input5
PIR1.automf(2, names = ['No','Yes'])

#Linguistic terms for input6
PIR2.automf(2, names = ['No','Yes'])

#Linguistic terms for input7
Claudy.automf(2, names = ['No','Yes'])

#Linguistic terms for the output
MoreThan2.automf(2, names = ['False','True'])
MoreThan2.defuzzify_method = "mom"


#Membership functions
"""
S1Light.view()
S2Light.view()
S3Light.view()
PIR2.view()
MoreThan2.view()
plt.show()
"""

#Rule set
rule1 = ctrl.Rule(S1Light['medium'] & S2Light['low'], MoreThan2['False'])
rule2 = ctrl.Rule(S1Light['high'] & S2Light['low'], MoreThan2['False'])
rule3 = ctrl.Rule(S1Light['high'] & S2Light['high'], MoreThan2['True'])
rule4 = ctrl.Rule(S1Light['low'] & S2Light['high'], MoreThan2['True'])
rule5 = ctrl.Rule(S1Light['medium'] & S3Light['low'], MoreThan2['False'])
rule6 = ctrl.Rule(S1Light['low'] & S3Light['high'], MoreThan2['True'])
rule7 = ctrl.Rule(S2Light['low'] & S3Light['high'], MoreThan2['False'])

#DayTime Rule
rule8 = ctrl.Rule(DayTime['No'] & S1Light['high'] & S3Light['high'], MoreThan2['True'])
rule9 = ctrl.Rule(DayTime['No'] & S2Light['high'] & S3Light['high'], MoreThan2['True'])
rule10 = ctrl.Rule(DayTime['No'] & S2Light['low'] & S3Light['low'], MoreThan2['False'])

#Possible rules to be deleted
rule11 = ctrl.Rule(S1Light['low'] & S2Light['low'], MoreThan2['False'])
rule12 = ctrl.Rule(S2Light['high'] & S3Light['low'], MoreThan2['False'])
rule13 = ctrl.Rule(S2Light['high'] & S3Light['high'], MoreThan2['True'])

#Claudy Rules
#rule14 = ctrl.Rule(Claudy['Yes'] & (PIR1['Yes'] | PIR2['Yes']), MoreThan2['False'])


people_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6,
                                 rule7, rule8, rule9, rule10, rule11, rule12, rule13])


people_model = ctrl.ControlSystemSimulation(people_ctrl)


fuzzy_df = fuzzy_df.drop(['Time', 'CO2', 'PIR1', 'PIR2', 'Persons'],  axis=1)
#fuzzy_df = fuzzy_df.drop(['Time', 'CO2','Persons'],  axis=1)
fuzzy_df.to_csv('ChangedData1.csv', decimal=',', sep=';', index=False)

y_vector = fuzzy_df.pop('MoreThan2')
x_vector = fuzzy_df


"""df_out = pd.DataFrame(x_vector)
df_out.to_csv('ChangedData1.csv', decimal=',', sep=';', index=False)"""


#create vector containing predictions
y_pred= np.zeros(len(y_vector))
y_fuzzy= np.zeros(len(y_vector))
for i in range(len(y_vector)):
        
    people_model.input['S1Light'] = x_vector.iloc[i,0]
    people_model.input['S2Light'] = x_vector.iloc[i,1]
    people_model.input['S3Light'] = x_vector.iloc[i,2]
    people_model.input['DayTime'] = x_vector.iloc[i,3]

    people_model.compute()
    people_prob=people_model.output['MoreThan2']
    y_fuzzy[i]= people_prob
    
    if people_prob > 0.5:
        y_pred[i]=1
    else:
        y_pred[i]=0

"""
df_out2 = pd.DataFrame(y_fuzzy)
df_out2.to_csv('yfuzzy.csv', decimal=',', sep=';', index=False)
"""

print("\n\n------------------------------------Results--------------------------------------------")
print("Accuracy:", accuracy_score(y_vector, y_pred))
print("Pecision:", precision_score(y_vector, y_pred, average=None))
print("Recall:", recall_score(y_vector, y_pred, average=None))
print("Macro-Pecision:", precision_score(y_vector, y_pred, average='macro'))
print("Macro-Recall:", recall_score(y_vector, y_pred, average='macro'))
print("Macro-F1-Score:", f1_score(y_vector, y_pred, average='macro'))
print("Confusion Matrix:", confusion_matrix(y_vector, y_pred))
print("---------------------------------------------------------------------------------------")