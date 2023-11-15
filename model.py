import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
import pickle

df = pd.read_csv('datasets/train.csv')

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Age'] = df['Age'].astype('int64')

le = LabelEncoder()

sex = le.fit_transform(df['Sex'])
df['Sex'] = sex

SEED = 42

X = df.drop('Survived', axis=1).values
y = df['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

kf = KFold(n_splits=70, shuffle=True, random_state=SEED)
rf = RandomForestClassifier()
gs = RandomizedSearchCV(rf, param_distributions={'max_depth': range(2, 10),"n_estimators":range(10, 200, 10), "random_state":range(1, 50)}, cv=kf, scoring='accuracy', refit=True)

gs.fit(X_train, y_train)
gs.best_params_

pickle.dump(gs, open('model.pkl', 'wb'))