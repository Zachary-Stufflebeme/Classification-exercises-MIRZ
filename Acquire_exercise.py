mport pandas as pd
import numpy as np
from env import host, username, password
import os

def get_connection(db, user = username, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_titanic_data():
    return pd.read_sql('''SELECT * FROM passengers''' , get_connection('titanic_db'))

def new_iris_data():
    return pd.read_sql('''
    Select * from species 
    join measurements using(species_id)''', get_connection('iris_db'))

def new_telco_data():
    return pd.read_sql('''
SELECT * FROM customers
join customer_payments using(customer_id)
join contract_types using(contract_type_id)
join internet_service_types using(internet_service_type_id
''', get_connection('telco_churn'))

def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = new_titanic_data()
        df.to_csv(filename)
        return df

def get_iris_data():
    filename = 'iris.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = new_iris_data()
        df.to_csv(filename)
        return df

def get_telco_data():
    filename = 'telco.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = new_telco_data()
        df.to_csv(filename)
        return df