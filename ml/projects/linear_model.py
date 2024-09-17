# Credit: HOML by Aurelien Geron

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

life_sat = pd.read_csv("data/lifesat.csv")
X = life_sat[["GDP per capita (USD)"]].values
y = life_sat[["Life satisfaction"]].values

life_sat.plot(kind="scatter", x="GDP per capita (USD)", y="Life satisfaction", grid=True)
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Linear model

model1 = LinearRegression()

model1.fit(X, y)

X_new = [[30000]]
print(model1.predict(X_new))

# K Nearest Neighbours

model2 = KNeighborsRegressor()

model2.fit(X, y)
print(model2.predict(X_new))