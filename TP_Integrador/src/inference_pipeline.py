"""
inference_pipeline.py

DESCRIPCIÓN: Corre los scripts de feature engineering y predict para
realizar predicciones sobre un conjunto de datos de entrada.

AUTOR: Ezequiel Scordamaglia y Santiago González Achaval
FECHA: 10/08/2023
"""

import subprocess

subprocess.run(['Python', 'feature_engineering.py', 'test'], check=False)

subprocess.run(['Python', 'predict.py'], check=False)
