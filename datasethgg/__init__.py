"""datasethgg: manejo y preprocesado de datasets (versión Python).

Paquete equivalente al paquete de R `datasethgg`. Define las clases
Attribute / NumericAttribute / CategoricalAttribute y Dataset, y funciones de
discretización, escalado, métricas (varianza, entropía, AUC), filtrado,
correlación, información mutua y gráficos.

Todas las funciones se pueden usar directamente tras:

    from datasethgg import *
"""
from .dataset_atributos import Attribute, NumericAttribute, CategoricalAttribute
from .dataset import Dataset
from .dataset_utils import (is_numeric, from_dataframe, to_dataframe,
                            read_dataset, write_dataset)
from .discretizacion import (discretize_ew, discretize_ef, discretizeEW,
                             discretizeEF, discretize, discretize_df,
                             entropy, entropia_df)
from .estadistica import (normalize, standardize, normalizacion,
                          estandarizacion, normalizar_df, estandarizar_df,
                          rates, roc, auc, auc_variable, mutual_info,
                          matriz_cor)
from .dataset_funciones import (apply_numeric, discretize_dataset,
                                normalize_dataset, standardize_dataset,
                                dataset_metrics, filter_dataset,
                                cor_matrix, mi_matrix)
from .graficar import (graf_roc, graf_auc, graf_entropia, graf_matriz,
                       graf_cor, graf_mi)

__version__ = "1.0.0"

__all__ = [
    # objetos
    "Attribute", "NumericAttribute", "CategoricalAttribute", "Dataset",
    # utilidades de datasets
    "is_numeric", "from_dataframe", "to_dataframe", "read_dataset", "write_dataset",
    # discretizacion y entropia
    "discretize_ew", "discretize_ef", "discretizeEW", "discretizeEF",
    "discretize", "discretize_df", "entropy", "entropia_df",
    # estadistica
    "normalize", "standardize", "normalizacion", "estandarizacion",
    "normalizar_df", "estandarizar_df", "rates", "roc", "auc", "auc_variable",
    "mutual_info", "matriz_cor",
    # funciones sobre datasets
    "apply_numeric", "discretize_dataset", "normalize_dataset",
    "standardize_dataset", "dataset_metrics", "filter_dataset",
    "cor_matrix", "mi_matrix",
    # graficos
    "graf_roc", "graf_auc", "graf_entropia", "graf_matriz", "graf_cor", "graf_mi",
]
