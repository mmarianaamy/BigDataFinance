import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLasso
from sklearn.linear_model import Lars

#Utiliza el archivo que ya fue limpiado
x = pd.read_csv("archivo_limpiado.csv")

#Covierte todas las fechas al mismo formato de pandas
x["fecha_transaccion"] = pd.to_datetime(x["fecha_transaccion"])

#Elige el cliente deseado
yOriginal = x[x["id_cliente"] == 463]
#Selecciona todos los giros
giros = yOriginal["giro_nombre"].unique()

#Ingresa los intervalos que se consideran para los gastos. En este caso, son semanales
intervalos = pd.date_range(start='2022-11-01 00:00:00', end='2023-03-08 00:24:00', freq="W")

ints = len(intervalos)

pointsyOriginal = [[0] for i in range(ints - 1)]

#Suma todos los gastos de cada uno de las categorias
for i in giros:
    y = yOriginal[yOriginal["giro_nombre"] == i]
    for j in range(ints - 1):
        z = y[y['fecha_transaccion'].between(intervalos[j], intervalos[j + 1], inclusive="left")]
        gastos = z["monto_transaccion"].sum()
        pointsyOriginal[j] += gastos

#Grafica los datos dados
xxOriginal = np.array(list(range(1, ints))).reshape(-1, 1)
plt.plot(xxOriginal, pointsyOriginal)

#Selecciona los puntos para entrenar al modelo. Usamos los primeros en vez de unos arbitrarios para tener una idea lineal del historial
pointsy = pointsyOriginal[:int(len(pointsyOriginal) / 1.5)]
xx = xxOriginal[:int(len(pointsyOriginal) / 1.5)]

#Definicion de modelos
modelo1 = LinearRegression()
modelo1.fit(xx, pointsy)

modelo2 = ElasticNet(random_state=0)
modelo2.fit(xx, pointsy)

modelo3 = MultiTaskLasso(alpha=0.1)
modelo3.fit(xx, pointsy)

modelo4 = Lars(n_nonzero_coefs=1)
modelo4.fit(xx, pointsy)

#Define los puntos usdos para el modelo polinomial
xx2 = np.array(xxOriginal[int(len(pointsyOriginal) / 1.5):]).reshape(-1, 1)
py = pointsyOriginal[int(len(pointsyOriginal) / 1.5):]

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(xx)
poly_reg_model = LinearRegression()
poly_reg_model.fit(poly_features, pointsy)
poly_features2 = poly.fit_transform(xx2)
deg = 2

past_maei = 2
current_maei = 1

#Prueba los diferentes modelos hasta bajar lo mas posible. Esto debido a evitar caer en ajustar de mas o de menos el modelo
while past_maei > current_maei:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    poly_features = poly.fit_transform(xx)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, pointsy)
    poly_features2 = poly.fit_transform(xx2)
    if deg == 2:
        current_maei = mean_absolute_error(poly_reg_model.predict(poly_features2), py)
        past_maei = current_maei + 1
    else:
        past_maei = current_maei
        current_maei = mean_absolute_error(poly_reg_model.predict(poly_features2), py)
    deg += 1

#Predicciones de los modelos
predicciones = [modelo1.predict(xx2), modelo2.predict(xx2), modelo3.predict(xx2), modelo4.predict(xx2)]
#Cálculo de los errores
maei = [mean_absolute_error(predicciones[0], py), mean_absolute_error(predicciones[1], py), mean_absolute_error(predicciones[2], py), mean_absolute_error(predicciones[3], py), past_maei]
xx3 = np.array(range(ints - 1, ints + ints//5)).reshape(-1, 1)

#Define el modelo definitivo, y se usan todos los datos para ajustar todo el modelo.
if min(maei) == maei[0]:
    modeloDef = LinearRegression()
    modeloDef.fit(xxOriginal, pointsyOriginal)
    yy3 = modeloDef.predict(xx3)
elif min(maei) == maei[1]:
    modeloDef = ElasticNet(random_state=0)
    modeloDef.fit(xxOriginal, pointsyOriginal)
    yy3 = modeloDef.predict(xx3)
elif min(maei) == maei[2]:
    modeloDef = MultiTaskLasso(alpha=0.5)
    modeloDef.fit(xxOriginal, pointsyOriginal)
    yy3 = modeloDef.predict(xx3)
elif min(maei) == maei[3]:
    modeloDef = Lars(n_nonzero_coefs=1)
    modeloDef.fit(xxOriginal, pointsyOriginal)
    yy3 = modeloDef.predict(xx3)
else:
    poly = PolynomialFeatures(degree=deg-1, include_bias=False)
    poly_features = poly.fit_transform(xxOriginal)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, pointsyOriginal)
    poly_features2 = poly.fit_transform(xx3)
    yy3 = poly_reg_model.predict(poly_features2)

# Se pone a 0 como el límite inferior de los valores en y
yy3 = [j if j >= 0 else 0 for j in yy3]

#Graficacion
intervalos2 = [d.strftime('%Y-%m-%d') for d in pd.date_range(start='2022-11-01 00:00:00', end='2023-03-08 00:24:00', freq="W")]
plt.plot(xx3, yy3, ls = ':')

x1 = np.array([xxOriginal[-1], xx3[0]])
y1 = np.array([pointsyOriginal[-1], yy3[0]])
plt.plot(x1, y1, ls = ':')

plt.xticks(range(len(xxOriginal) + 1), intervalos2)
plt.xticks(rotation=45)
plt.title("Total")
plt.show()