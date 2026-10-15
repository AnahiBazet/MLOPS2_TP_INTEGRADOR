import numpy as np


class LogisticRegressionModel:
    """Regresión logística binaria para entrenamiento federado."""

    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    # El cliente envía los parámetros a Flower.
    def get_parameters(self):
        return [self.weights.copy(), np.array([self.bias])]

    # Actualiza el modelo local con los parámetros recibidos desde el servidor.
    def set_parameters(self, parameters):
        self.weights = np.asarray(parameters[0], dtype=np.float64).copy()
        self.bias = float(np.asarray(parameters[1]).item())

    def fit(self, X, y, epochs=5):
        """Entrena mediante descenso de gradiente por lotes completos."""
        n = len(y)

        for _ in range(epochs):
            probabilities = self.predict_proba(X)
            errors = probabilities - y

            gradient_w = X.T @ errors / n
            gradient_b = np.mean(errors)

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b