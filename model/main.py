import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

pd.set_option('display.max_columns', 30)
df = pd.read_csv('../data/bank-additional-full.csv', sep=';')
df.dropna(inplace=True)
X, y = df.drop(columns=['y']), df['y']

integer = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
           'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

categorical = ['job', 'marital', 'education', 'default', 
               'housing', 'loan', 'contact', 'month', 
               'day_of_week', 'poutcome']

ct = ColumnTransformer([('scaler', StandardScaler(), integer),
                        ('ordinal', OrdinalEncoder(), categorical)], remainder='passthrough')
X_transformed = pd.DataFrame(ct.fit_transform(X), columns=ct.get_feature_names_out())

