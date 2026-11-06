"""Gráficos (plotnine) para curvas ROC, AUC, entropía y matrices.

Equivalente a R/graficar.R (graf.roc, graf.auc, graf.entropy, graf.matriz,
graf.cor y graf.mi). Todas las funciones devuelven un objeto ggplot de
plotnine, al que se le pueden añadir capas o temas (por ejemplo
`+ theme_light()`).
"""
import numpy as np
import pandas as pd
from plotnine import *
from .discretizacion import entropia_df
from .estadistica import roc, matriz_cor
from .dataset_funciones import dataset_metrics, mi_matrix


def graf_roc(df, name="Curva ROC", ref=True):
    """Gráfico de la curva ROC de una variable.

    Parámetros:
        df (pd.DataFrame): columnas "Val" (numérica) y "Etiqueta" (booleana).
        name (str): nombre de la curva (leyenda).
        ref (bool): si es True (por defecto) añade la diagonal de referencia.

    Devuelve:
        ggplot: la curva ROC.
    """
    curva = roc(df)
    dfp = pd.DataFrame({'FPR': curva['FPR'],
                        'TPR': curva['TPR'],
                        'ROC': name})
    g = ggplot(dfp, aes(x='FPR', y='TPR', col='ROC')) + geom_line(aes(colour='ROC')) + coord_cartesian(ylim = (0,1))
    g = g + geom_abline(slope = 1, intercept = 0, linetype = "dashed") if (ref) else g
    return(g)


def graf_auc(ds):
    """Gráfico de barras con el AUC de cada variable numérica de un Dataset.

    Se añade una línea discontinua en AUC = 0.5. Solo aparecen las variables
    numéricas para las que se ha podido calcular el AUC (clase binaria).

    Parámetros:
        ds (Dataset): dataset con clase binaria.

    Devuelve:
        ggplot: el gráfico de barras.
    """
    m = dataset_metrics(ds)
    m = m[m["auc"].notna()]
    return (ggplot(m, aes(x="variable", y="auc"))
            + geom_bar(stat="identity", fill="steelblue")
            + geom_hline(yintercept=0.5, linetype="dashed")
            + labs(x="", y="AUC")
            + coord_flip()
            + theme_minimal())


def graf_entropia(df):
    """Gráfico de barras con la entropía normalizada de cada columna.

    La entropía se divide entre log2(k), con k el número de valores distintos
    de la columna (si k = 1 la entropía normalizada es 0).

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas categóricas.

    Devuelve:
        ggplot: el gráfico de barras.
    """
    ent = entropia_df(df).iloc[0]

    ent = ent / df.apply(lambda x: np.log2(x.nunique())).replace(0, np.nan)
    ent = ent.fillna(0.0)

    df_plot = pd.DataFrame({"Variable": ent.index, "Entropy": ent.values})

    return (ggplot(df_plot, aes(x="Variable", y="Entropy"))
            + geom_bar(stat="identity", fill="steelblue")
            + labs(x="", y="Entropy")
            + coord_flip()
            + theme_minimal())


def graf_matriz(m, titulo=""):
    """Mapa de calor de una matriz con nombres de fila y columna.

    Sirve tanto para matrices de correlación como de información mutua. Los
    valores se muestran redondeados a dos decimales.

    Parámetros:
        m (pd.DataFrame): matriz cuadrada con índice y columnas con nombres.
        titulo (str): título del gráfico.

    Devuelve:
        ggplot: el mapa de calor.
    """
    df_plot = m.stack().reset_index()
    df_plot.columns = ["Var1", "Var2", "value"]
    df_plot["label"] = df_plot["value"].round(2).astype(str)

    return (ggplot(df_plot, aes(x="Var1", y="Var2", fill="value")) +
             geom_tile() +
             geom_text(aes(label="label"), size=7) +
             labs(title=titulo, x="", y=""))


def graf_cor(df):
    """Mapa de calor de las correlaciones de un DataFrame.

    Es una envoltura de graf_matriz() aplicada a matriz_cor(df).

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas numéricas.

    Devuelve:
        ggplot: el mapa de calor.
    """
    return graf_matriz(matriz_cor(df), "Correlaciones")


def graf_mi(ds, num_bins=None):
    """Mapa de calor de la información mutua entre las variables de un Dataset.

    Es una envoltura de graf_matriz() aplicada a mi_matrix(ds, num_bins).

    Parámetros:
        ds (Dataset): dataset a analizar.
        num_bins (int, opcional): si se indica, las numéricas se discretizan
            (igual frecuencia) y entran en el cálculo.

    Devuelve:
        ggplot: el mapa de calor.
    """
    return graf_matriz(mi_matrix(ds, num_bins), "Informacion mutua (bits)")
