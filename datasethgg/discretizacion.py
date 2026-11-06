"""Discretización de variables numéricas y entropía.

Equivalente a R/discretizacion.R (discretizeEW, discretizeEF, discretize y entropy).
"""
import numpy as np
import pandas as pd


def discretize_ew(values, num_bins):
    """Discretización de igual anchura (versión para atributos de un Dataset).

    Parámetros:
        values (array-like): valores numéricos.
        num_bins (int): número de intervalos.

    Devuelve:
        pd.Categorical o pd.Series categórica con los intervalos como
        categorías. Si la variable es constante, devuelve una única categoría.
    """
    v = np.asarray(values, dtype=float)
    lo, hi = v.min(), v.max()
    if hi == lo:
        return pd.Categorical([str(lo)] * len(v))
    cut_points = np.linspace(lo, hi, num_bins + 1)
    return pd.cut(v, cut_points, include_lowest=True)


def discretize_ef(values, num_bins):
    """Discretización de igual frecuencia (versión para atributos de un Dataset).

    Usa pd.qcut con duplicates="drop": si hay cuantiles repetidos se obtienen
    menos intervalos que `num_bins`.

    Parámetros:
        values (array-like): valores numéricos.
        num_bins (int): número de intervalos.

    Devuelve:
        pd.Categorical con los intervalos como categorías.
    """
    v = np.asarray(values, dtype=float)
    return pd.qcut(v, num_bins, duplicates="drop")


def discretizeEW(x, num_bins):
    """Discretización de igual anchura (equal width).

    Divide el rango [min, max] de `x` en `num_bins` intervalos de la misma
    anchura (cerrados por la derecha; el primero incluye el mínimo).

    Parámetros:
        x (array-like): valores numéricos.
        num_bins (int): número de intervalos.

    Devuelve:
        tuple: (x_discretized, cut_points), donde x_discretized es la variable
        discretizada (categorías = intervalos) y cut_points son los
        num_bins + 1 límites de los intervalos (incluidos los extremos).
    """
    x_max, x_min = np.max(x), np.min(x)
    cut_points = np.linspace(x_min, x_max, num_bins + 1)
    x_discretized = pd.cut(x, cut_points, include_lowest=True)
    return(x_discretized, cut_points)


def discretizeEF(x, num_bins):
    """Discretización de igual frecuencia (equal frequency).

    Usa como puntos de corte los cuantiles 0, 1/k, 2/k, ..., 1, de modo que
    cada intervalo contiene (aproximadamente) el mismo número de observaciones.
    Si hay cuantiles repetidos pd.cut lanza un ValueError.

    Parámetros:
        x (array-like): valores numéricos.
        num_bins (int): número de intervalos.

    Devuelve:
        tuple: (x_discretized, cut_points), con la variable discretizada y los
        num_bins + 1 puntos de corte (incluidos los extremos).
    """
    cut_points = np.quantile(x, q=np.arange(0,num_bins+1)/num_bins)
    x_discretized = pd.cut(x, cut_points, include_lowest=True)
    return(x_discretized, cut_points)


def discretize(x, cut_points):
    """Discretiza una variable con unos puntos de corte ya conocidos.

    Los valores fuera de [min(cut_points), max(cut_points)] quedan como NaN;
    para evitarlo se pueden poner -inf e inf como primer y último punto.

    Parámetros:
        x (array-like): valores numéricos.
        cut_points (array-like): puntos de corte ordenados, incluidos los extremos.

    Devuelve:
        tuple: (x_discretized, cut_points).
    """
    x_discretized = pd.cut(x, cut_points, include_lowest=True)
    return(x_discretized, cut_points)


def discretize_df(df, method='EW', cut_points=[]):
    """Discretiza todas las columnas de un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas numéricas.
        method (str): "EW" (igual anchura, 10 intervalos), "EF" (igual
            frecuencia, 10 intervalos) o cualquier otro valor para usar los
            puntos de corte dados en `cut_points`.
        cut_points (list): puntos de corte (solo si method no es EW ni EF).

    Devuelve:
        pd.DataFrame con cada columna discretizada.
    """
    if method == 'EW':
        return df.apply(lambda x: discretizeEW(x, 10)[0],axis=0)
    elif method == 'EF':
        return df.apply(lambda x: discretizeEF(x, 10)[0],axis=0)
    else:
        return df.apply(lambda x: discretize(x, cut_points)[0],axis=0)


def entropy(values):
    """Entropía de Shannon (en bits) de una variable categórica.

    Calcula  H = -sum(p_i * log2(p_i)),  donde p_i es la frecuencia relativa
    de cada valor distinto.

    Parámetros:
        values (array-like): valores de la variable (lista, array o categórica).

    Devuelve:
        float: la entropía en bits.
    """
    _, counts = np.unique(np.asarray(values), return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))


def entropia_df(df):
    """Entropía de cada columna de un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas categóricas.

    Devuelve:
        pd.DataFrame de una fila con la entropía de cada columna.
    """
    return (pd.DataFrame({col: [entropy(df[col].tolist())] for col in df.columns}))
