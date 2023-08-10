"""
predict.py

DESCRIPCIÓN: Usando un modelo entrenado, se realizan predicciones sobre 
             un conjunto de datos de entrada.
AUTOR: Ezquiel Scordamaglia - Santiago Gonzalez Achaval
FECHA: 10/8/2023
"""

import pickle as pkl
import os
import pandas as pd

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

        data = pd.read_csv(self.input_path,sep=",")

        return data

    def load_model(self) -> None:
        """
        Carga el modelo entrenado.
        """

        with open(self.model_path, 'rb') as model_file:
            self.model = pkl.load(model_file)

    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones sobre el conjunto de datos de entrada.
        """

        new_data = data.drop(columns=['Item_Outlet_Sales'])
        new_data['Item_Outlet_Sales'] = self.model.predict(new_data)

        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        Escribe las predicciones en el directorio de salida.
        """

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
  