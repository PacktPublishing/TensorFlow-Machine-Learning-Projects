import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import zipfile


from parameters import *

def read_data():
    zf = zipfile.ZipFile(os.path.join(DATA_DIR,"creditcardfraud.zip"))
    data = pd.read_csv(zf.open("creditcard.csv"))
    return data


def preprocess_data(data):
    data = data.drop(['Time'], axis=1)
    data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    return data


def load_and_preprocess_data():
    data = read_data()
    processed_data = preprocess_data(data)
    return processed_data

def get_train_and_test_data(processed_data):
    X_train, X_test = train_test_split(processed_data, test_size=0.25, random_state=RANDOM_SEED)
    X_train = X_train[X_train.Class == 0]
    X_train = X_train.drop(['Class'], axis=1)

    y_test = X_test['Class']
    X_test = X_test.drop(['Class'], axis=1)

    X_train = X_train.values
    X_test = X_test.values
    return X_train, X_test,y_test



















