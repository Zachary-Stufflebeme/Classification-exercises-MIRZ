from pydataset import data
import pandas as pd
import numpy as np
import plotly as py
from Acquire_exercise import get_iris_data, read_csv, get_titanic_data, get_telco_data
from sklearn.model_selection import train_test_split

def prep_iris(fresh_iris):
    iris = fresh_iris.drop(columns = {'species_id','measurement_id','Unnamed: 0'})
    iris = iris.rename(columns = {'species_name':'species'})
    iris['encoded_species'] = iris['species'].map({'setosa':1,'versicolor':2,'virginica':3})
    return iris

def prep_titanic(FreshTitanic):
    Titanic = FreshTitanic.drop(columns = {'passenger_id', 'Unnamed: 0'})
    Titanic['encoded_class'] = FreshTitanic['class'].map({'First': 1, 'Second': 2, 'Third': 3})
    return Titanic

def prep_telco_data(FreshTelco):
    df = FreshTelco.T.drop_duplicates().T
    df = df.drop(columns = {'internet_service_type', 'customer_id','Unnamed: 0'})
    df['is_female'] = df.gender.map({'Female': 1, 'Male': 0})
    df['has_partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['has_dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['has_phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['uses_paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['did_churn'] = df.churn.map({'Yes': 1, 'No': 0})
    df['has_mult_lines'] = df.multiple_lines.map({'No' : 0, 'Yes' : 1,'No phone': 2 })
    df['has_online_security'] = df.online_security.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    df['has_online_backup'] = df.online_backup.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    df['has_device_protection'] = df.device_protection.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    df['has_tech_support'] = df.tech_support.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    df['can_stream_tv'] = df.streaming_tv.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    df['can_stream_movie'] = df.streaming_movies.map({'No' : 0, 'Yes' : 1, 'No Internet Service' : 2})
    return df

def my_train_test_split(df, target):

    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=.25, random_state=123, stratify=train[target])

    return train, validate, test