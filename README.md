# datasethgg (Python)

Versión en Python del paquete de R `datasethgg`: manejo y preprocesado de
datasets con clases propias (`NumericAttribute`, `CategoricalAttribute`,
`Dataset`). Las implementaciones de discretización, entropía, información
mutua, curva ROC y AUC son propias; solo se usan `numpy`, `pandas` y
`plotnine` (gráficos).

## Instalación

Desde el repositorio de GitHub:

```bash
pip install git+https://github.com/Huguer03/datasethgg-py.git
```

O desde una copia local (modo desarrollo), situándose en la carpeta del paquete:

```bash
pip install -e .
```

## Documentación

Cada función está documentada con su docstring (entradas y salidas):

```python
help(discretizeEW)
```

## Autor

Hugo González González
