from setuptools import setup

setup(
    name='datasethgg',
    version='1.0.0',
    author='Hugo Gonzalez Gonzalez',
    author_email='hgonzalez031@ikasle.ehu.eus',
    packages=['datasethgg', 'datasethgg.test'],
    url='https://github.com/Huguer03/datasethgg-py',
    license='GPL-2.0-or-later',
    description='Manejo y preprocesado de datasets: discretizacion, escalado, metricas, filtrado, correlacion, informacion mutua y graficos',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    tests_require=['pytest'],
    install_requires=[
        "pandas >= 1.3.0",
        "numpy >= 1.20.0",
        "plotnine >= 0.10.0",
    ],
    extras_require={
        # solo para ejecutar el notebook de ejemplo (docs/ejemplo.ipynb)
        "ejemplos": ["scikit-learn", "jupyter"],
    },
)
