import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

## Data Preprocessing ##

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

sampler = SMOTE()
X_sampled, y_sampled = sampler.fit_resample(X_transformed, y)

## Model Training ##

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.20, random_state=42)

clf = GradientBoostingClassifier(learning_rate=0.8, n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

## Model Testing ##

clf_report = classification_report(y_test, y_pred)
print('Classification report: \n')
print(clf_report)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap='Blues')
plt.show()