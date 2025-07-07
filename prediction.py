import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

df=pd.read_csv("veriler.csv")

X=df[["yas", "marka"]]
Y=df["fiyat"]

ct=ColumnTransformer(transformers=[("encoder",OneHotEncoder(handle_unknown="ignore"),["marka"])],remainder="passthrough")

X_encoded=ct.fit_transform(X)

poly=PolynomialFeatures(degree=4,include_bias=False)
X_poly=poly.fit_transform(X_encoded)

model=LinearRegression()
model.fit(X_poly,Y)

y_pred=model.predict(X_poly)
print(y_pred)

yeni_veri=pd.DataFrame({"yas":[3],"marka":["MERCEDES"]})
yeni_veri_encoded=ct.transform(yeni_veri)
yeni_veri_poly=poly.transform(yeni_veri_encoded)
tahmin=model.predict(yeni_veri_poly)
print("mercedes için fiyath tahmini: ",int(tahmin[0]))

plt.figure(figsize=(8,5))
plt.scatter(X["yas"], Y, color='blue', label='Gerçek Fiyat')
plt.scatter(X["yas"], y_pred, color='red', label='Tahmin Edilen Fiyat', marker='x')
plt.xlabel("Yaş")
plt.ylabel("Fiyat")
plt.title("Yaş ve Fiyat İlişkisi")
plt.legend()
plt.grid(True)
plt.show()