"""
predict.py

DESCRIPCIÓN: Usando un modelo entrenado, se realizan predicciones sobre 
             un conjunto de datos de entrada.
AUTOR: Ezquiel Scordamaglia - Santiago Gonzalez Achaval
FECHA: 10/8/2023
"""

import pickle as pkl
import os
import logging as log
import pandas as pd

log.basicConfig(
    filename='./predict.log',
    level=log.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
class MakePredictionPipeline():
    """
    Clase que carga un modelo entrenado y realiza predicciones sobre 
    un conjunto de datos de entrada.
    """

    def __init__(self, input_path, output_path, model_path: str = None):
        self.input_path = input_path
        self.output_path = output_path
        self.model_path = model_path
        self.model = None

    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos de entrada y los retorna como un DataFrame.

        :return: El DataFrame con los datos de entrada.
        """

        try:
            log.info("Iniciando la lectura de datos desde %s", self.input_path)
            data = pd.read_csv(self.input_path,sep=",")
            log.info("Datos leídos: Filas=%d, Columnas=%d",
                    data.shape[0], data.shape[1])
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e_lectura:
            log.info("Error %s al importar dataframe", e_lectura)
        return data

    def load_model(self) -> None:
        """
        Carga el modelo entrenado.
        """

        with open(self.model_path, 'rb') as model_file:
            self.model = pkl.load(model_file)

        log.info("Cargando modelo entrenado..")
        try:
            with open(self.model_path, 'rb') as model_file:
                self.model = pkl.load(model_file)
            log.info("Modelo cargado")
        except (FileNotFoundError, PermissionError, pkl.PickleError) as e_guardado:
            log.error("Un error ocurrio al cargar el modelo: %s", str(e_guardado))
    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones sobre el conjunto de datos de entrada.
        """
        log.info("Realizando predicciones..")
        new_data = data.drop(columns=['Item_Outlet_Sales'])
        new_data['Item_Outlet_Sales'] = self.model.predict(new_data)

        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Escribe las predicciones en el directorio de salida.
        """
        log.info("Guardando predicciones en el dataset")
        predicted_data.to_csv(self.output_path, index=False)

    def run(self):
        """
        Llama a los metodos.
        """

        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)

if __name__ == "__main__":

    log.info("Script de predicción iniciado")
    current_directory = os.path.dirname(os.path.abspath(__file__))

    in_path = os.path.join(current_directory,
                           "..", 
                           "data","Transformed",
                           "Test_BigMart_Prepared.csv")

    out_path = os.path.join(current_directory,
                           "..", 
                           "data",
                           "Test_BigMart_Predictions.csv")

    mod_path = os.path.join(current_directory, "..", "model", "model.pkl")

    MakePredictionPipeline(input_path = in_path,
                            output_path = out_path,
                            model_path = mod_path).run()
    log.info("Script de predicción finalizado")
  