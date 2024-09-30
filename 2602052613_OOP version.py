# 2602052613 - Jovian Yanto

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.output_df = None

    def loadData(self):
        self.data = pd.read_csv(self.file_path)

    def createInputOutput(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)

    def hotEncode(self, column):
        self.data = pd.get_dummies(self.data, columns=column)

    def dropColumn(self, col):
        self.data = self.data.drop(col, axis=1)


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def splitData(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)

    def cekMissing(self, data):
        print(data.isnull().sum())

    def checkOutlier(self, data, col):
        boxplot = data.boxplot(column=col)
        plt.show()

    def colMean(self, data, col):
        return np.mean(data[col])

    def colMedian(self, data, col):
        return np.nanmedian(data[col])

    def inputMissVal(self, data, column, value):
        data[column].fillna(value, inplace=True)

    def createModel(self, min_samples_split= 2, max_depth= None, n_estimators= 100, min_samples_leaf= 1):
        self.model = RandomForestClassifier(min_samples_split= min_samples_split, max_depth= max_depth, n_estimators= n_estimators, min_samples_leaf= min_samples_leaf)

    def predictModel(self):
        self.y_predict = self.model.predict(self.x_test)

    def trainModel(self):
        self.model.fit(self.x_train, self.y_train)

    def createReport(self):
        print("\nClassification Report")
        print(classification_report(self.y_test, self.y_predict))

    def tuningParams(self):
        params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 8],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [2, 4]
        }
        RF_class = RandomForestClassifier()
        RF_class = GridSearchCV(RF_class, param_grid = params, scoring='accuracy', cv=5)
        RF_class.fit(self.x_train, self.y_train)
        print("Tuned Hyperparameters :", RF_class.best_params_)
        print("Accuracy :",RF_class.best_score_)

        self.createModel(min_samples_split= RF_class.best_params_['min_samples_split'],
                         max_depth= RF_class.best_params_['max_depth'],
                         n_estimators= RF_class.best_params_['n_estimators'], 
                         min_samples_leaf= RF_class.best_params_['min_samples_leaf'])

    def saveModelToFile(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

file_path = 'data_A.csv'
data_handler = DataHandler(file_path)
data_handler.loadData()

data_handler.dropColumn(['Unnamed: 0', 'id', 'CustomerId', 'Surname'])
data_handler.hotEncode(['Geography', 'Gender'])

data_handler.createInputOutput('churn')

# training
input_df = data_handler.input_df
output_df = data_handler.output_df


model_handler = ModelHandler(input_df, output_df)
model_handler.splitData()

# check outlier
# model_handler.checkOutlier(model_handler.x_train, 'CreditScore')

model_handler.inputMissVal(model_handler.x_train, 'CreditScore', model_handler.colMedian(model_handler.x_train, 'CreditScore'))
model_handler.inputMissVal(model_handler.x_test, 'CreditScore', model_handler.colMedian(model_handler.x_test, 'CreditScore'))

print("===== Untuned Params =====")
model_handler.trainModel()

model_handler.predictModel()
model_handler.createReport()


print("===== Tuned")
model_handler.tuningParams()
model_handler.trainModel()
model_handler.predictModel()
model_handler.createReport()

model_handler.saveModelToFile('RF_class.pkl') 
