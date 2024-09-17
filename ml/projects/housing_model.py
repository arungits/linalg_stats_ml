import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/housing.csv")
# print(df.info())
# print(df.describe()[["median_income", "median_house_value"]])
print(df.head())

df.plot(kind="scatter", x="median_income", y="median_house_value", grid=True)
plt.axis([0,16,14_000,600_000])
plt.show()