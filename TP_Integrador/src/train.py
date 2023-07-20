"""
train.py

DESCRIPCIÓN: Entrenamiento de modelo con dataframe final. Se exporta modelo entrenado .pkl.
AUTOR: Ezquiel Scordamaglia - Santiago Gonzalez Achaval
FECHA: 17/7/2023
"""

# Imports
import pickle as pkl
import os
import pandas as pd
from sklearn.linear_model import LinearRegression


class ModelTrainingPipeline:
    """
    Clase que toma el dataframe de la ruta seleccionada y entrena un modelo
    utilizando el modelo linear regression.
    """
    def __init__(self, input_path, model_path):
        """
        Toma las ubicaciones de entrada y salida.

        :return: Los paths de entrada y salida.
        :rtype: pd.dataframe
        """
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:

        """
        Leer los datos del DataLake y retornarlos como un DataFrame.

        :return: El DataFrame con los datos del DataLake.
        :rtype: pd.DataFrame
        """
        df_bigmart = pd.read_csv(self.input_path,sep=";")

        return df_bigmart

    def model_training(self, df_bigmart: pd.DataFrame) -> pd.DataFrame:

        """
        Entrenar el modelo usando linear regression.
        Usamos el dataset entero al no precisar de validacion en este punto.

        :return: Modelo entrenado con el dataframe
        :rtype: pandas.LinearRegression
        """

        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        x_train = df_bigmart.drop(columns=['Item_Outlet_Sales'])
        y_train = df_bigmart['Item_Outlet_Sales']

        # Entrenamiento del modelo
        model_trained = model.fit(x_train,y_train)

        return model_trained

    def model_dump(self, model_trained) -> None:

        """
        Guarda el modelo entrenado en un archivo o ubicación específica.

        :param model_trained: El modelo entrenado que se desea guardar.
        :type model_trained: None
        """
        file_path = self.model_path
        with open(file_path, 'wb') as f_pkl:
            pkl.dump(model_trained, f_pkl)

    def run(self):
        """
        Llama a los metodos.
 
        """

        df_bigmart = self.read_data()
        model_trained = self.model_training(df_bigmart)
        self.model_dump(model_trained)

if __name__ == "__main__":

    current_directory = os.path.dirname(os.path.abspath(__file__))

    in_path = os.path.join(current_directory,
                           "..", 
                           "data","Transformed",
                           "Train_BigMart_Prepared.csv")
    mod_path = os.path.join(current_directory, "..", "model.pkl")

    ModelTrainingPipeline(input_path = in_path,
                           model_path = mod_path).run()
    