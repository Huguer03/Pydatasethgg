"""Funciones de preprocesado que trabajan sobre un Dataset completo.

Equivalente a R/dataset_funciones.R (aplica.num, discretizar.dataset,
normalizar.dataset, estandarizar.dataset, metricas.dataset, filtrar.dataset,
matriz.cor y matriz.mi).

NOTA: las funciones que transforman un Dataset (apply_numeric,
discretize_dataset, normalize_dataset, standardize_dataset y filter_dataset)
MODIFICAN el dataset que reciben y lo devuelven. Para conservar el original
hay que pasarles una copia: copy.deepcopy(ds).
"""
import numpy as np
import pandas as pd

from .dataset_atributos import NumericAttribute, CategoricalAttribute
from .dataset_utils import is_numeric, to_dataframe
from .discretizacion import discretize_ew, discretize_ef, entropy
from .estadistica import normalize, standardize, auc_variable, mutual_info


def apply_numeric(ds, f):
    """Aplica una función a todos los atributos numéricos de un Dataset.

    Equivalente a aplica.num() en R.

    Parámetros:
        ds (Dataset): dataset a transformar (se modifica).
        f (callable): función que recibe un NumericAttribute y devuelve otro
            Attribute (NumericAttribute o CategoricalAttribute).

    Devuelve:
        Dataset: el mismo dataset, con los atributos numéricos transformados
        por `f`. Los categóricos no se modifican.
    """
    for name, a in list(ds.attributes.items()):
        if is_numeric(a):
            ds.attributes[name] = f(a)
    ds._validate()
    return ds


def discretize_dataset(ds, num_bins, method="EF"):
    """Discretiza los atributos numéricos de un Dataset.

    Equivalente a discretizar.dataset() en R.

    Parámetros:
        ds (Dataset): dataset a transformar (se modifica).
        num_bins (int): número de intervalos.
        method (str): "EW" (igual anchura) o "EF" (igual frecuencia, por defecto).

    Devuelve:
        Dataset: con los atributos numéricos convertidos en CategoricalAttribute.
    """
    f = discretize_ew if method == "EW" else discretize_ef
    return apply_numeric(ds, lambda a: CategoricalAttribute(a.name, f(a.values, num_bins)))


def normalize_dataset(ds):
    """Normaliza al rango [0, 1] los atributos numéricos de un Dataset.

    Parámetros:
        ds (Dataset): dataset a transformar (se modifica).

    Devuelve:
        Dataset: con los atributos numéricos normalizados.
    """
    return apply_numeric(ds, lambda a: NumericAttribute(a.name, normalize(a.values)))


def standardize_dataset(ds):
    """Estandariza (puntuaciones Z) los atributos numéricos de un Dataset.

    Parámetros:
        ds (Dataset): dataset a transformar (se modifica).

    Devuelve:
        Dataset: con los atributos numéricos estandarizados.
    """
    return apply_numeric(ds, lambda a: NumericAttribute(a.name, standardize(a.values)))


def dataset_metrics(ds):
    """Métricas por variable de un Dataset. Equivalente a metricas.dataset().

    Calcula varianza y AUC para las variables numéricas y entropía para las
    categóricas. La variable clase no se incluye. El AUC solo se calcula si el
    dataset tiene clase con exactamente 2 categorías (el positivo es la segunda).

    Parámetros:
        ds (Dataset): dataset a analizar.

    Devuelve:
        pd.DataFrame con una fila por variable y columnas "variable", "tipo",
        "varianza", "auc" y "entropia" (NaN donde la métrica no aplica).
    """
    names = [n for n in ds.attributes if n != ds.class_name]

    labels = None
    if ds.class_name:
        cl = ds.attributes[ds.class_name].values.remove_unused_categories()
        if len(cl.categories) == 2:
            labels = np.asarray(cl) == cl.categories[1]

    rows = []
    for n in names:
        a = ds.attributes[n]
        row = {"variable": n, "tipo": a.kind,
               "varianza": np.nan, "auc": np.nan, "entropia": np.nan}
        if is_numeric(a):
            row["varianza"] = float(a.values.var(ddof=1))
            if labels is not None:
                row["auc"] = auc_variable(a.values, labels)
        else:
            row["entropia"] = entropy(a.values)
        rows.append(row)
    return pd.DataFrame(rows)


def filter_dataset(ds, metrica, umbral, mayor=True):
    """Filtra las variables de un Dataset según una métrica.

    Equivalente a filtrar.dataset() en R. Las variables a las que no aplica la
    métrica y la variable clase no se eliminan.

    Parámetros:
        ds (Dataset): dataset a filtrar (se modifica).
        metrica (str): "varianza", "auc" o "entropia".
        umbral (float): valor umbral.
        mayor (bool): si es True (por defecto) se conservan las variables con
            métrica >= umbral; si es False, las de métrica <= umbral.

    Devuelve:
        Dataset: sin las variables que no cumplen el criterio.
    """
    m = dataset_metrics(ds)
    v = m[metrica]
    cumple = v >= umbral if mayor else v <= umbral
    quitar = set(m.loc[v.notna() & ~cumple, "variable"])
    ds.attributes = {n: a for n, a in ds.attributes.items() if n not in quitar}
    ds._validate()
    return ds


def cor_matrix(ds):
    """Matriz de correlaciones entre las variables numéricas de un Dataset.

    Equivalente a matriz.cor() en R.

    Parámetros:
        ds (Dataset): dataset a analizar.

    Devuelve:
        pd.DataFrame: matriz de correlaciones de Pearson (filas y columnas con
        los nombres de las variables numéricas).
    """
    df = to_dataframe(ds)
    num = [n for n, a in ds.attributes.items() if is_numeric(a)]
    return df[num].corr()


def mi_matrix(ds, num_bins=None):
    """Matriz de información mutua entre variables categóricas de un Dataset.

    Equivalente a matriz.mi() en R. La diagonal contiene la entropía de cada
    variable.

    Parámetros:
        ds (Dataset): dataset a analizar.
        num_bins (int, opcional): si se indica, las variables numéricas se
            discretizan con igual frecuencia en `num_bins` intervalos y entran
            también en el cálculo. Si es None (por defecto) se ignoran.

    Devuelve:
        pd.DataFrame: matriz cuadrada y simétrica con la información mutua (bits).
    """
    facs = {}
    for n, a in ds.attributes.items():
        if not is_numeric(a):
            facs[n] = np.asarray(a.values)
        elif num_bins is not None:
            facs[n] = np.asarray(discretize_ef(a.values, num_bins))

    names = list(facs)
    m = np.zeros((len(names), len(names)))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            m[i, j] = mutual_info(facs[ni], facs[nj])
    return pd.DataFrame(m, index=names, columns=names)
