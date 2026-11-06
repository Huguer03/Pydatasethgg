"""Escalado de variables, curva ROC, AUC e información mutua.

Equivalente a R/estadistica.R (normalize, standardize, rates, roc, auc,
auc.variable y mutual.info).
"""
import numpy as np
import pandas as pd

from .discretizacion import entropy


def normalize(values):
    """Normaliza un vector al rango [0, 1] (reescalado min-max).

    Aplica  x' = (x - min) / (max - min). Si todos los valores son iguales
    devuelve un vector de ceros.

    Parámetros:
        values (array-like): valores numéricos.

    Devuelve:
        np.ndarray: valores normalizados (mínimo 0, máximo 1).
    """
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    return np.zeros_like(v) if hi == lo else (v - lo) / (hi - lo)


def standardize(values):
    """Estandariza un vector (puntuaciones Z) con la desviación típica muestral.

    Aplica  x' = (x - media) / sd,  con sd calculada con ddof=1 (como el sd de R).
    Si sd es 0 devuelve x - media.

    Parámetros:
        values (array-like): valores numéricos.

    Devuelve:
        np.ndarray: valores estandarizados (media 0, desviación típica 1).
    """
    v = np.asarray(values, dtype=float)
    sd = v.std(ddof=1)
    return v - v.mean() if sd == 0 else (v - v.mean()) / sd


def normalizacion(x):
    """Normaliza al rango [0, 1]: (x - min) / (max - min).

    Parámetros:
        x (array-like o pd.Series): valores numéricos.

    Devuelve:
        Valores normalizados, del mismo tipo que `x`.
    """
    return((x - np.min(x)) / (np.max(x) - np.min(x)))


def estandarizacion(x):
    """Estandariza: (x - media) / desviación típica.

    Parámetros:
        x (array-like o pd.Series): valores numéricos.

    Devuelve:
        Valores estandarizados, del mismo tipo que `x`.
    """
    return((x - np.mean(x)) / np.std(x))


def normalizar_df(df):
    """Normaliza (min-max) todas las columnas de un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas numéricas.

    Devuelve:
        pd.DataFrame con cada columna en el rango [0, 1].
    """
    return(df.apply(lambda x: normalizacion(x),axis=0))


def estandarizar_df(df):
    """Estandariza todas las columnas de un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas numéricas.

    Devuelve:
        pd.DataFrame con cada columna estandarizada.
    """
    return(df.apply(lambda x: estandarizacion(x),axis=0))


def rates(cut_value, df, shorted=False):
    """Tasas TPR y FPR para un punto de corte.

    Una observación se clasifica como positiva si Val >= cut_value.

        TPR = TP / (TP + FN)        FPR = FP / (FP + TN)

    Parámetros:
        cut_value (float): punto de corte.
        df (pd.DataFrame): columnas "Val" (puntuación numérica) y "Etiqueta"
            (booleana, True = positivo).
        shorted (bool): si es False (por defecto) se ordena df por Val de
            forma decreciente; si es True se asume ya ordenado.

    Devuelve:
        dict: {"TPR": ..., "FPR": ...}
    """
    if not shorted:
        df = df.sort_values("Val", ascending=False, ignore_index=True)

    etiqueta = df["Etiqueta"].astype(bool)
    positivo = df["Val"] >= cut_value

    TP = (etiqueta & positivo).sum()
    FP = (~etiqueta & positivo).sum()
    TN = (~etiqueta & ~positivo).sum()
    FN = (etiqueta & ~positivo).sum()

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return {"TPR": TPR, "FPR": FPR}


def roc(df):
    """Curva ROC de una variable frente a una etiqueta binaria.

    Calcula (TPR, FPR) para cada valor distinto de "Val" como punto de corte,
    añade los extremos (0, 0) y (1, 1), y deja un único punto por cada FPR
    (el de mayor TPR).

    Parámetros:
        df (pd.DataFrame): columnas "Val" (numérica) y "Etiqueta" (booleana).

    Devuelve:
        pd.DataFrame con columnas "TPR" y "FPR", ordenado por FPR creciente.
    """
    df = df.sort_values("Val", ignore_index=True)
    cv = df["Val"].unique()

    curve = pd.DataFrame([rates(c, df, shorted=True) for c in cv])

    extremos = pd.DataFrame({"TPR": [0.0, 1.0], "FPR": [0.0, 1.0]})
    curve = pd.concat([curve, extremos], ignore_index=True)

    curve = curve.sort_values(["FPR", "TPR"], ascending=[True, False])
    curve = curve.drop_duplicates(subset="FPR", keep="first")
    return curve.reset_index(drop=True)


def auc(curva):
    """Área bajo la curva ROC (AUC) por la regla del trapecio.

    Parámetros:
        curva (pd.DataFrame): columnas "FPR" y "TPR", como la que devuelve roc().

    Devuelve:
        float: el AUC (entre 0 y 1).
    """
    x = curva["FPR"].to_numpy()
    y = curva["TPR"].to_numpy()
    return np.nansum(np.diff(x) * (y[:-1] + y[1:]) / 2)


def auc_variable(values, labels):
    """AUC de una variable numérica frente a una etiqueta binaria.

    Es una función de conveniencia que combina roc() y auc().

    Parámetros:
        values (array-like): valores numéricos de la variable.
        labels (array-like): etiqueta booleana (True = positivo).

    Devuelve:
        float: el AUC; NaN si falta alguna de las dos clases.
    """
    v = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    return float(auc(roc(pd.DataFrame({"Val": v, "Etiqueta": y}))))


def mutual_info(x, y):
    """Información mutua (en bits) entre dos variables categóricas.

    Se calcula como  I(X,Y) = H(X) + H(Y) - H(X,Y).

    Parámetros:
        x (array-like): valores de la primera variable.
        y (array-like): valores de la segunda variable (misma longitud).

    Devuelve:
        float: la información mutua en bits.
    """
    x, y = np.asarray(x), np.asarray(y)
    union = np.array([f"{a}|{b}" for a, b in zip(x, y)])
    return entropy(x) + entropy(y) - entropy(union)


def matriz_cor(df):
    """Matriz de correlaciones entre las columnas de un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas numéricas.

    Devuelve:
        pd.DataFrame: matriz de correlaciones de Pearson.
    """
    return df.corr()
