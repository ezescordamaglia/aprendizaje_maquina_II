# Aprendizaje de Maquina 2
# Especialización de Inteligencia Artificial
# CEIA

# TP Integrador 2023

# Entrenamiento de modelo de ML basado en archivo de Train y prediccion de variable objetivo sobre un archivo de Test

# Instrucciones de uso
Para utilizar el pipeline de entrenamiento, es necesario contar con un archivo llamado Train_BigMart.csv dentro de la carpeta ../data/

Para entrenar al modelo se debe utilizar la siguiente instrucción:

TP_Integrador\src> python train_pipeline.py

Esta instrucción generará dos artefactos:
- ../data/Transformed/Train_BigMart_Prepared.csv
- ../model/model.pkl

Luego para generar predicciones sobre el modelo entrenado, es necesario contar con un archivo llamado Test_BigMart.csv dentro de la carpeta ../data/ y un modelo entrenado dentro de la carpeta ../model/

Para generar las predicciones debe utilizar la siguiente instrucción:

TP_Integrador\src> python inference_pipeline.py

Esta instrucción generará dos artefactos:
- ../data/Transformed/Test_BigMart_Prepared.csv
- ../data/Test_BigMart_Predictions.csv



