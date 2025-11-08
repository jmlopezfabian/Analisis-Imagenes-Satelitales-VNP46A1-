import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def get_elasticity(x: np.ndarray, y: np.ndarray, plot: bool = False) -> float:
    """
    Calcula la elasticidad promedio en escala log-log:
    E = Δlog(y) / Δlog(x)
    """
    X = np.array(x)
    Y = np.array(y)

    # Diferencias entre observaciones consecutivas
    delta_x = X[1:] - X[:-1]
    delta_y = Y[1:] - Y[:-1]

    # Evitar divisiones por valores muy pequeños
    mask = np.abs(delta_x) > 1e-3

    elasticity = delta_y[mask] / delta_x[mask]
    elasticity_avg = np.mean(elasticity)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(elasticity[mask], label="Elasticidad instantánea", color='blue')
        plt.axhline(elasticity_avg, color='red', linestyle='--', label=f'Promedio: {elasticity_avg:.3f}')
        plt.legend()
        plt.title("Elasticidad (escala log-log)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return elasticity_avg

def prueba_adf(x: np.ndarray) -> bool:
    """
    Prueba la hipótesis nula de que la serie es no estacionaria.
    """
    result = adfuller(x)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4e}")
    print(f"Valores críticos:")
    for key, value in result[4].items():
        print(f"{key}: {value:.4f}")
    if result[1] < 0.05:
        print("Rechazamos la hipótesis nula, la serie es estacionaria")
    else:
        print("No rechazamos la hipótesis nula, la serie es no estacionaria")
