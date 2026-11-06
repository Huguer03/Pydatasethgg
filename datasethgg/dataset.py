"""Clase Dataset: conjunto de atributos y, opcionalmente, una variable clase.

Equivalente a R/dataset.R (clase dataset).
"""
from .dataset_atributos import CategoricalAttribute


class Dataset:
    """Conjunto de datos formado por atributos y, opcionalmente, la variable clase.

    Parámetros del constructor:
        attributes (list): lista de objetos Attribute (NumericAttribute o
            CategoricalAttribute). Todos deben tener la misma longitud.
        class_name (str, opcional): nombre del atributo que actúa como clase.
            Debe existir en el dataset y ser un CategoricalAttribute. Si es None,
            el dataset no tiene clase.

    Campos:
        attributes (dict): diccionario nombre -> Attribute.
        class_name (str o None): nombre de la variable clase.

    Lanza:
        ValueError: si los atributos tienen distinta longitud, si la clase no
            está en el dataset o si la clase no es categórica.
    """

    def __init__(self, attributes, class_name=None):
        self.attributes = {a.name: a for a in attributes}
        self.class_name = class_name
        self._validate()

    def _validate(self):
        """Comprueba que el dataset es válido (misma longitud y clase correcta).

        Lanza ValueError si no lo es. No devuelve nada.
        """
        lengths = {len(a) for a in self.attributes.values()}
        if len(lengths) > 1:
            raise ValueError("Los atributos tienen distinta longitud")
        if self.class_name is not None:
            if self.class_name not in self.attributes:
                raise ValueError(f"La clase {self.class_name!r} no esta en el dataset")
            if not isinstance(self.attributes[self.class_name], CategoricalAttribute):
                raise ValueError("La clase debe ser categorica")

    def __repr__(self):
        """Devuelve un resumen: número de variables, clase y tipo de cada variable."""
        lines = [f"Dataset: {len(self.attributes)} variables"]
        lines.append(f"Clase: {self.class_name if self.class_name else '(ninguna)'}")
        for a in self.attributes.values():
            lines.append(f" - {a.name}: {a.kind}")
        return "\n".join(lines)
