#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

# Statistical Lotto/Lottery Prediction 7/39
# GradientBoostingRegressor

import numpy as np
import pandas as pd


from scipy import stats
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from qiskit_machine_learning.utils import algorithm_globals
import random


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score


# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# 1. Učitaj loto podatke
df = pd.read_csv("/Users/milan/Desktop/GHQ/data/loto7h_4502_k85.csv")





# Pretpostavljamo da prve 7 kolona sadrže brojeve lutrije
df = df.iloc[:, :7]

# Kreiranje ulaznih (X) i izlaznih (y) podataka
X = df.shift(1).dropna().values
y = df.iloc[1:].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=39)




# sns.distplot(df.sum(axis=1), fit=stats.gamma)




mpl.rc("figure", figsize=(12, 12))

sns.countplot(x="Num1", data=df)
plt.show()
sns.countplot(x="Num2", data=df)
plt.show()
sns.countplot(x="Num3", data=df)
plt.show()
sns.countplot(x="Num4", data=df)
plt.show()
sns.countplot(x="Num5", data=df)
plt.show()
sns.countplot(x="Num6", data=df)
plt.show()
sns.countplot(x="Num7", data=df)
plt.show()

###################################



X = np.array([df["Num1"], df["Num2"], df["Num3"], df["Num4"], df["Num5"], df["Num6"], df["Num7"]] )
bindex = 0
final = []

print()
for ball in [ df["Num1"], df["Num2"], df["Num3"], df["Num4"], df["Num5"], df["Num6"], df["Num7"] ]:
    Y = np.array(ball.values.tolist())
    X_train, X_test, y_train, y_test = train_test_split(X.transpose(), Y, test_size=0.8, random_state=None)
    reg = GradientBoostingRegressor()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)               
    # accuracy = accuracy_score(y_test, y_pred)  
    final.append(y_pred[bindex])
    if len(final)!=len(set(final)):
        Y = np.array(ball.values.tolist())
        X_train, X_test, y_train, y_test = train_test_split(X.transpose(), Y, test_size=0.9, random_state=None)
        reg = GradientBoostingRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)               
        # accuracy = accuracy_score(y_test, y_pred)  
        final.append(y_pred[bindex])        
        
    print(f"Prediction of Ball {bindex + 1} is [{y_pred[bindex]}] ")
    bindex = bindex + 1  

    
print()



"""
Prediction of Ball 1 is [9.999869050876402] 
Prediction of Ball 2 is [10.000000696496336] 
Prediction of Ball 3 is [12.000071664060362] 
Prediction of Ball 4 is [16.000132567052486] 
Prediction of Ball 5 is [19.000210213115942] 
Prediction of Ball 6 is [31.999976514048733] 
Prediction of Ball 7 is [29.000159474567052] 
"""

print()
final.sort()
print("Predicted Numbers:", np.round(final).astype(int).tolist())
print()
S = sum(final)
print(f"Sum of numbers: {S}")
print(f"Sum is good!") if S >= 120 and S <= 190 else print(f"Sum of prediction is out of ideal range. Re-run prediction.")
print()

"""
Predicted Numbers: [10, 10, 12, 16, 19, 29, 32]

Sum of numbers: 128.00042018021733
Sum is good!
"""





"""

source ~/Desktop/GHQ/qiskit_novi/bin/activate

source ~/Desktop/GHQ/qiskit_stari/bin/activate

deactivate

"""






