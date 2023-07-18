"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: Entrenamiento de modelo con dataframe final. Se exporta modelo entrenado .pkl.
AUTOR: Ezquiel Scordamaglia - Santiago Gonzalez Achaval
FECHA: 17/7/2023
"""

# Imports
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import pickle as pkl
import os
class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        Leer los datos del DataLake y retornarlos como un DataFrame.

        :return: El DataFrame con los datos del DataLake.
        :rtype: pd.DataFrame
        """
            
        # COMPLETAR CON CÓDIGO
        pandas_df = pd.read_csv(f"{self.input_path}/train_final.csv")

        return pandas_df

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """
        COMPLETAR DOCSTRING
        Entrenar el modelo usando linear regression.
        Usamos el dataset entero al no precisar de validacion en este punto.
        """
        
        # COMPLETAR CON CÓDIGO
        seed = 28
        model = LinearRegression()

        # División de dataset de entrenaimento y validación
        X = df.drop(columns='Item_Outlet_Sales') #[['Item_Weight', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type']] # .drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'], test_size = 0, random_state=seed)

        # Entrenamiento del modelo
        model_trained = model.fit(x_train,y_train)

        return model_trained

    def model_dump(self, model_trained) -> None:
        
        """
        Guarda el modelo entrenado en un archivo o ubicación específica.

        :param model_trained: El modelo entrenado que se desea guardar.
        :type model_trained: Any
        """
            
        # COMPLETAR CON CÓDIGO PARA GUARDAR EL MODELO
        file_path = f"{self.model_path}/trained_model.pkl"
        with open(file_path, 'wb') as f:
            pkl.dump(model_trained, f)
        
        return None

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = '../final_data',
                          model_path = '../model').run()