"""Utilidades para crear, convertir, leer y escribir datasets.

Equivalente a R/dataset_utils.R (es.num, crear.dataset, dataset.a.df,
leer.dataset y escribir.dataset).
"""
import numpy as np
import pandas as pd

from .dataset_atributos import NumericAttribute, CategoricalAttribute
from .dataset import Dataset


def is_numeric(a):
    """Comprueba si un atributo es numérico. Equivalente a es.num() en R.

    Parámetros:
        a (Attribute): atributo a comprobar.

    Devuelve:
        bool: True si `a` es un NumericAttribute, False en caso contrario.
    """
    return isinstance(a, NumericAttribute)


def from_dataframe(df, class_name=None):
    """Crea un Dataset a partir de un DataFrame. Equivalente a crear.dataset().

    Las columnas numéricas (no booleanas) pasan a ser NumericAttribute y el
    resto CategoricalAttribute.

    Parámetros:
        df (pd.DataFrame): datos.
        class_name (str, opcional): nombre de la columna que actúa como clase.

    Devuelve:
        Dataset: el dataset creado.
    """
    attrs = []
    for name in df.columns:
        col = df[name]
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            attrs.append(NumericAttribute(name, col.to_numpy()))
        else:
            attrs.append(CategoricalAttribute(name, col))
    return Dataset(attrs, class_name)


def to_dataframe(ds):
    """Convierte un Dataset en DataFrame. Equivalente a dataset.a.df().

    Parámetros:
        ds (Dataset): dataset a convertir.

    Devuelve:
        pd.DataFrame: una columna por atributo (se pierde la información de
        cuál es la clase).
    """
    return pd.DataFrame({name: a.values for name, a in ds.attributes.items()})


def read_dataset(path, class_name=None, max_val=5):
    """Lee un dataset desde un fichero CSV. Equivalente a leer.dataset().

    Una columna con menos de `max_val` valores distintos (o de tipo texto) se
    considera categórica; el resto, numérica.

    Parámetros:
        path (str): ruta del fichero CSV.
        class_name (str, opcional): nombre de la columna que actúa como clase.
        max_val (int): número de valores distintos por debajo del cual una
            columna se considera categórica (por defecto 5).

    Devuelve:
        Dataset: el dataset leído.
    """
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].nunique() < max_val or df[col].dtype == object:
            df[col] = df[col].astype("category")
    return from_dataframe(df, class_name)


def write_dataset(ds, path):
    """Escribe un dataset en un fichero CSV. Equivalente a escribir.dataset().

    La variable clase no se guarda en el CSV: hay que indicarla de nuevo al leer.

    Parámetros:
        ds (Dataset): dataset a guardar.
        path (str): ruta del fichero de salida.

    Devuelve:
        None. Escribe el fichero.
    """
    to_dataframe(ds).to_csv(path, index=False)
