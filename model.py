import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import pickle

df = pd.read_csv('train.csv')
#parch lo quito de aqui y lo dejo para entrenar tambien
df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1, inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].astype('int64')

le = LabelEncoder()
sex = le.fit_transform(df['Sex'])
df['Sex'] = sex

print(df.head())

SEEDD = 42
X = df.drop('Survived', axis=1).values
y = df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEEDD)
rf = RandomForestClassifier(n_estimators=160, max_depth=5, random_state=SEEDD)

rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))



pickle.dump(rf, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[4, 300, 500]]))