"""Clases que representan los atributos (columnas) de un dataset.

Equivalente a R/dataset_atributos.R (clases atributo, atributo.num y atributo.cat).
"""
import numpy as np
import pandas as pd


class Attribute:
    """Atributo genérico de un dataset (clase base).

    Parámetros del constructor:
        name (str): nombre del atributo.
        values: valores del atributo.

    Campos:
        name (str): nombre del atributo.
        values: valores (los convierte cada subclase).
        kind (str): tipo de atributo ("base", "numeric" o "categorical").
    """
    kind = "base"

    def __init__(self, name, values):
        self.name = name
        self.values = values

    def __len__(self):
        """Devuelve el número de observaciones del atributo."""
        return len(self.values)

    def __repr__(self):
        """Devuelve una descripción corta: tipo de objeto, nombre y longitud."""
        return f"<{type(self).__name__} {self.name!r} n={len(self)}>"


class NumericAttribute(Attribute):
    """Atributo con valores numéricos (enteros o reales).

    Parámetros del constructor:
        name (str): nombre del atributo.
        values (array-like): valores numéricos; se guardan como np.ndarray de float.
    """
    kind = "numeric"

    def __init__(self, name, values):
        super().__init__(name, np.asarray(values, dtype=float))


class CategoricalAttribute(Attribute):
    """Atributo con valores categóricos (strings, booleanos o códigos).

    Usa pd.Categorical para conservar los niveles al estilo de un factor de R.

    Parámetros del constructor:
        name (str): nombre del atributo.
        values (array-like): valores; se guardan como pd.Categorical.
    """
    kind = "categorical"

    def __init__(self, name, values):
        super().__init__(name, pd.Categorical(values))
