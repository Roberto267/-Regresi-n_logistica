import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
df = pd.read_csv("Employee.csv")

# Codificación one-hot y mapeo de 'EverBenched'
df_encoded = pd.get_dummies(df, columns=['Education', 'City', 'Gender'])
df_encoded['EverBenched'] = df_encoded['EverBenched'].map({'No': 0, 'Yes': 1})

# Dividir datos en conjuntos de entrenamiento y prueba
train_data = df_encoded.sample(frac=0.8, random_state=42)
test_data = df_encoded.drop(train_data.index)

# Matrices X y y
X_train = (train_data.drop('LeaveOrNot', axis=1) - np.mean(train_data.drop('LeaveOrNot', axis=1), axis=0)) / np.std(train_data.drop('LeaveOrNot', axis=1), axis=0)
y_train = train_data['LeaveOrNot']
X_test = (test_data.drop('LeaveOrNot', axis=1) - np.mean(test_data.drop('LeaveOrNot', axis=1), axis=0)) / np.std(test_data.drop('LeaveOrNot', axis=1), axis=0)
y_test = test_data['LeaveOrNot']

# Inicializar parámetros y función sigmoide mejorada
theta = np.random.randn(X_train.shape[1]) * 0.01

# Funciones
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_cost_regularized(X, y, theta, lambda_):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-10
    cost = -1 / m * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))
    regularization_term = lambda_ / (2 * m) * np.sum(theta[1:]**2)
    cost += regularization_term
    return cost

def gradient_descent_regularized(X, y, theta, learning_rate, iterations, lambda_):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        regularization_term = lambda_ / m * np.concatenate(([0], theta[1:]))
        gradient += regularization_term
        theta -= learning_rate * gradient
        cost_history[i] = calculate_cost_regularized(X, y, theta, lambda_)

    return theta, cost_history

# Entrenar el modelo con regularización L2 (Ridge)
learning_rate = 0.01
iterations = 1000
lambda_ = 0.1
theta, cost_history = gradient_descent_regularized(X_train.values, y_train.values, theta, learning_rate, iterations, lambda_)

# Evaluar el modelo en los datos de prueba
y_pred_prob = sigmoid(X_test.values @ theta)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Imprimir la precisión del modelo
accuracy = np.mean(y_pred == y_test)
print(f'Precisión del modelo: {accuracy:.2f}')

# Visualizar la función de costo a lo largo de las iteraciones
plt.plot(range(1, iterations + 1), cost_history, color='blue', label='Función de costo')
plt.scatter(range(1, iterations + 1), cost_history, color='red', s=40, label='Puntos de datos')
plt.rcParams["figure.figsize"] = (10, 6)
plt.grid()
plt.xlabel('Número de iteraciones')
plt.ylabel('Costo')
plt.title('Regresión  logistica\nPrecision del modelo: ' + str(accuracy))
plt.legend()
plt.show()