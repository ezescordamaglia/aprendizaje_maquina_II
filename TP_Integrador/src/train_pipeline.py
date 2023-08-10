"""
train_pipeline.py

DESCRIPCIÓN: Corre los scripts de feature engineering y train para
realizar el entrenamiento de un modelo de ML.

AUTOR: Ezequiel Scordamaglia y Santiago González Achaval
FECHA: 10/08/2023
"""

import subprocess

subprocess.run(['Python', 'feature_engineering.py', 'train'], check=False)

subprocess.run(['Python', 'train.py'], check=False)
