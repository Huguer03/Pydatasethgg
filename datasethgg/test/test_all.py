"""Tests del paquete datasethgg. Se ejecutan con:  pytest --pyargs datasethgg"""
import copy
import os
import tempfile

import numpy as np
import pandas as pd

from datasethgg import *


def _datos():
    """Pequeño DataFrame mixto con una clase binaria."""
    rng = np.random.default_rng(1)
    n = 60
    clase = np.where(rng.random(n) < 0.5, "no", "si")
    return pd.DataFrame({
        "x1": rng.normal(size=n) + (clase == "si") * 2,   # relacionada con la clase
        "x2": rng.normal(size=n),
        "c1": rng.choice(["a", "b", "c"], size=n),
        "clase": clase,
    })


def test_objetos():
    df = _datos()
    ds = from_dataframe(df, class_name="clase")
    assert isinstance(ds, Dataset)
    assert isinstance(ds.attributes["x1"], NumericAttribute)
    assert isinstance(ds.attributes["c1"], CategoricalAttribute)
    assert ds.class_name == "clase"
    assert list(to_dataframe(ds).columns) == list(df.columns)
    # la clase debe ser categorica y existir
    for mala in ("x1", "no_existe"):
        try:
            from_dataframe(df, class_name=mala)
        except ValueError:
            pass
        else:
            raise AssertionError("debia lanzar ValueError con class_name=" + mala)


def test_discretizacion():
    x = np.arange(9, dtype=float)
    x_ew, cp = discretizeEW(x, 4)
    assert list(cp) == [0, 2, 4, 6, 8]
    assert len(x_ew.categories) == 4
    x_ef, _ = discretizeEF(np.arange(1, 101, dtype=float), 4)
    assert list(pd.Series(x_ef).value_counts()) == [25, 25, 25, 25]
    d, _ = discretize([1, 5, 9], [-np.inf, 3, 6, np.inf])
    assert list(d.codes) == [0, 1, 2]


def test_escalado():
    x = np.array([-0.29, 2.11, -3.84, 10.69, 5.30])
    n = normalize(x)
    assert n.min() == 0 and n.max() == 1
    z = standardize(x)
    assert abs(z.mean()) < 1e-12 and abs(z.std(ddof=1) - 1) < 1e-12


def test_entropia():
    assert abs(entropy(["a", "a", "c", "c", "c"]) - 0.9709506) < 1e-6


def test_roc_auc():
    val = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 1.0]
    eti = [False] * 4 + [True] * 3
    assert auc(roc(pd.DataFrame({"Val": val, "Etiqueta": eti}))) == 1
    assert auc_variable(val, eti) == 1           # variable perfecta
    # sin poder discriminante (empates entre clases): AUC = 0.5
    assert abs(auc_variable([1, 1, 2, 2], [False, True, False, True]) - 0.5) < 1e-12
    # si falta una de las clases no se puede calcular
    assert np.isnan(auc_variable([1, 2, 3], [True, True, True]))
    # una variable relacionada con la clase tiene AUC alto
    d = _datos()
    assert auc_variable(d["x1"], d["clase"] == "si") > 0.8


def test_metricas_y_filtrado():
    ds = from_dataframe(_datos(), class_name="clase")
    m = dataset_metrics(ds)
    assert list(m["variable"]) == ["x1", "x2", "c1"]      # sin la clase
    assert m.loc[0, "auc"] > 0.8                          # x1 discrimina
    assert np.isnan(m.loc[2, "auc"]) and not np.isnan(m.loc[2, "entropia"])
    f = filter_dataset(copy.deepcopy(ds), "auc", 0.8)
    assert "x1" in f.attributes and "x2" not in f.attributes
    assert "clase" in f.attributes and "c1" in f.attributes   # no se tocan


def test_dataset_completo():
    ds = from_dataframe(_datos(), class_name="clase")
    d = discretize_dataset(copy.deepcopy(ds), 3, "EW")
    assert all(a.kind == "categorical" for a in d.attributes.values())
    n = normalize_dataset(copy.deepcopy(ds))
    assert n.attributes["x1"].values.max() == 1
    s = standardize_dataset(copy.deepcopy(ds))
    assert abs(s.attributes["x2"].values.mean()) < 1e-12
    assert ds.attributes["x1"].kind == "numeric"          # el original no cambia


def test_asociacion():
    ds = from_dataframe(_datos(), class_name="clase")
    c = cor_matrix(ds)
    assert c.shape == (2, 2) and abs(c.loc["x1", "x1"] - 1) < 1e-12
    mi = mi_matrix(ds)
    assert list(mi.columns) == ["c1", "clase"]
    assert abs(mi.loc["c1", "c1"] - entropy(ds.attributes["c1"].values)) < 1e-12
    assert mi_matrix(ds, num_bins=3).shape == (4, 4)
    a = ["x", "x", "y", "y"]
    assert abs(mutual_info(a, ["u", "v", "u", "v"])) < 1e-12   # independientes
    assert abs(mutual_info(a, a) - 1) < 1e-12                  # H(X) = 1 bit


def test_graficos():
    ds = from_dataframe(_datos(), class_name="clase")
    df = _datos()
    roc_df = pd.DataFrame({"Val": df["x1"], "Etiqueta": df["clase"] == "si"})
    assert graf_roc(roc_df, "x1") is not None
    assert graf_auc(ds) is not None
    assert graf_entropia(to_dataframe(discretize_dataset(copy.deepcopy(ds), 3))) is not None
    assert graf_cor(df[["x1", "x2"]]) is not None
    assert graf_mi(ds, num_bins=3) is not None


def test_lectura_escritura():
    ds = from_dataframe(_datos(), class_name="clase")
    ruta = os.path.join(tempfile.mkdtemp(), "datos.csv")
    write_dataset(ds, ruta)
    ds2 = read_dataset(ruta, class_name="clase")
    assert list(ds2.attributes) == list(ds.attributes)
    assert ds2.attributes["x1"].kind == "numeric"
    assert ds2.attributes["c1"].kind == "categorical"
    assert ds2.class_name == "clase"


def test_all():
    """Comprueba que todo lo declarado en __all__ existe."""
    import datasethgg
    for nombre in datasethgg.__all__:
        assert hasattr(datasethgg, nombre), nombre
