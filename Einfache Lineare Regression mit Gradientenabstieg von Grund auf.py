
import pandas as pd  # Importieren der pandas-Bibliothek zur Datenverarbeitung

# Kostenfunktion MSE
def cost_function(Y, b, w, X):
    """
    Berechnet den mittleren quadratischen Fehler (MSE) für eine Reihe von Beobachtungen.

    Args:
        Y (array-like): Die tatsächlichen Beobachtungswerte.
        b (float): Der Bias-Wert.
        w (float): Der Gewichtswert.
        X (array-like): Die unabhängigen Variablen.

    Returns:
        float: Der mittlere quadratische Fehler.
    """
    m = len(Y)  # Anzahl der Beobachtungen
    sse = 0   # sse = Summe der quadratischen Fehler

    # Berechnung des quadratischen Fehlers für jede Beobachtung
    for i in range(0, m):
        y_hat = b + w * X[i]  # Berechnung des geschätzten Werts
        y = Y[i]  # Tatsächlicher Wert
        sse += (y_hat - y) ** 2  # Quadratischer Fehler hinzufügen

    mse = sse / m  # Berechnung des mittleren quadratischen Fehlers
    return mse

# Gewichtsaktualisierung
def update_weights(Y, b, w, X, learning_rate):
    """
    Aktualisiert die Gewichte basierend auf den partiellen Ableitungen der Kostenfunktion.

    Args:
        Y (array-like): Die tatsächlichen Beobachtungswerte.
        b (float): Der aktuelle Bias-Wert.
        w (float): Der aktuelle Gewichtswert.
        X (array-like): Die unabhängigen Variablen.
        learning_rate (float): Die Lernrate.

    Returns:
        tuple: Die aktualisierten Bias- und Gewichtswerte.
    """
    m = len(Y)  # Anzahl der Beobachtungen
    b_deriv_sum = 0  # Summe der partiellen Ableitung bzgl. des Bias
    w_deriv_sum = 0  # Summe der partiellen Ableitung bzgl. des Gewichts

    # Berechnung der Summen der partiellen Ableitungen für jeden Datenpunkt
    for i in range(0, m):
        y_hat = b + w * X[i]  # Berechnung des geschätzten Werts
        y = Y[i]  # Tatsächlicher Wert
        b_deriv_sum += (y_hat - y)  # Berechnung der Ableitung bzgl. des Bias
        w_deriv_sum += (y_hat - y) * X[i]  # Berechnung der Ableitung bzgl. des Gewichts

    # Aktualisierung der Gewichte mithilfe des Gradientenabstiegs
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w

# Trainingsfunktion
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    """
    Trainiert das Modell durch Anpassung der Gewichte für eine bestimmte Anzahl von Iterationen.

    Args:
        Y (array-like): Die tatsächlichen Beobachtungswerte.
        initial_b (float): Der anfängliche Bias-Wert.
        initial_w (float): Der anfängliche Gewichtswert.
        X (array-like): Die unabhängigen Variablen.
        learning_rate (float): Die Lernrate.
        num_iters (int): Die Anzahl der Iterationen.

    Returns:
        tuple: Eine Liste der Verlaufswerte der Kostenfunktion und die finalen Bias- und Gewichtswerte.
    """
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    # Iteration über die festgelegte Anzahl von Iterationen
    for i in range(num_iters):
        # Aktualisierung der Gewichte mithilfe des Gradientenabstiegs
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)  # Berechnung des aktuellen mittleren quadratischen Fehlers
        cost_history.append(mse)  # Hinzufügen des aktuellen Fehlers zur Verlaufsliste

        if i % 100 == 0:
            # Ausgabe des Fortschritts alle 100 Iterationen
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

