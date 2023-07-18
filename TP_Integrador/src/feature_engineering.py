"""
feature_engineering.py

DESCRIPCIÓN: Este script contiene la clase FeatureEngineeringPipeline, 
que se encarga de realizar la ingeniería de features sobre
los datos de entrada y escribir el resultado en un archivo
de salida.

AUTOR: Ezequiel Scordamaglia y Santiago González Achaval
FECHA: 17/07/2023
"""

import pandas as pd

class FeatureEngineeringPipeline:
    """
    Clase para manejar el pipeline de feature engineering.
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        Este metodo se encarga de leer los datos de entrada
        y devolverlos en un DataFrame de pandas.

        :return pandas_df: DataFrame de pandas con los datos de entrada.
        :rtype: pd.DataFrame
        """
        pandas_df = pd.read_csv(self.input_path)

        return pandas_df

    def data_transformation(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Este método se encarga de realizar la ingeniería de features
        sobre los datos de entrada y devolverlos en un DataFrame de pandas.
        
        :param df_raw: DataFrame de pandas con los datos de entrada.

        :return df_transformed: DataFrame de pandas transformado.
        :rtype: pd.DataFrame
        """

        df_transformed = df_raw.copy()

        # print(df_transformed.head())

        # FEATURES ENGINEERING: nueva columna para marcar registros de entrenamiento
        df_transformed['Set'] = 'train'

        # FEATURES ENGINEERING: para los años del establecimiento
        # Cálculo de años de vida de la tienda en base al año de establecimiento
        # y el año actual (se asume que es data del actual año 2020)
        df_transformed['Outlet_Establishment_Year'] = 2020 - \
            df_transformed['Outlet_Establishment_Year']

        # LIMPIEZA: Unificando etiquetas para 'Item_Fat_Content'
        df_transformed['Item_Fat_Content'] = df_transformed['Item_Fat_Content'].replace(
            {'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        # LIMPIEZA: de faltantes en el peso de los productos
        productos = list(df_transformed[
            df_transformed['Item_Weight'].isnull()
            ]['Item_Identifier'].unique())

        for producto in productos:
            # Revisar si existe un registro con el mismo identificador de producto
            # y que tenga el peso cargado
            if len(df_transformed[(df_transformed['Item_Identifier'] == producto)
                                  & (~df_transformed['Item_Weight'].isnull())]) > 0:
                moda = (df_transformed[
                    df_transformed['Item_Identifier'] == producto
                    ][['Item_Weight']]).mode().iloc[0,0]
                df_transformed.loc[
                    df_transformed['Item_Identifier'] == producto, 'Item_Weight'
                    ] = moda

        # Eliminar registros con valores perdidos en 'Item_Weight'
        df_transformed = df_transformed[~df_transformed['Item_Weight'].isnull()]

        outlets = list(df_transformed[
            df_transformed['Outlet_Size'].isnull()
            ]['Outlet_Identifier'].unique())

        # LIMPIEZA: de faltantes en el tamaño de las tiendas
        for outlet in outlets:
            df_transformed.loc[
                df_transformed['Outlet_Identifier'] == outlet, 'Outlet_Size'
                ] =  'Small'

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Household', 'Item_Fat_Content'
            ] = 'NA'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Health and Hygiene', 'Item_Fat_Content'
            ] = 'NA'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Hard Drinks', 'Item_Fat_Content'
            ] = 'NA'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Soft Drinks', 'Item_Fat_Content'
            ] = 'NA'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Fruits and Vegetables', 'Item_Fat_Content'
            ] = 'NA'

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df_transformed['Item_Type'] = df_transformed['Item_Type'].replace(
            {'Others': 'Non perishable',
            'Health and Hygiene': 'Non perishable',
            'Household': 'Non perishable',
            'Seafood': 'Meats',
            'Meat': 'Meats',
            'Baking Goods': 'Processed Foods',
            'Frozen Foods': 'Processed Foods',
            'Canned': 'Processed Foods',
            'Snack Foods': 'Processed Foods',
            'Breads': 'Starchy Foods',
            'Breakfast': 'Starchy Foods',
            'Soft Drinks': 'Drinks',
            'Hard Drinks': 'Drinks',
            'Dairy': 'Drinks'})

        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Non perishable', 'Item_Fat_Content'
            ] = 'NA'

        # FEATURES ENGINEERING: Codificando los niveles de precios de los productos
        df_transformed['Item_MRP'] = pd.qcut(
            df_transformed['Item_MRP'], 4, labels = [1, 2, 3, 4]
            )

        # FEATURES ENGINEERING: Se eliminan las columnas que no se utilizarán
        df_transformed = df_transformed.drop(columns=['Item_Type', 'Item_Fat_Content'])

        # FEATURES ENGINEERING: Codificación de variables ordinales
        df_transformed['Outlet_Size'] = df_transformed['Outlet_Size'] \
            .replace({'High': 2, 'Medium': 1, 'Small': 0})

        df_transformed['Outlet_Location_Type'] = df_transformed['Outlet_Location_Type'] \
            .replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})

        # FEATURES ENGINEERING: Codificación de variables nominales
        df_transformed = pd.get_dummies(df_transformed, columns=['Outlet_Type'], dtype=int)

        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Este método se encarga de escribir los datos de salida
        en un archivo csv.
        
        :param transformed_dataframe: DataFrame de pandas transformado.
        """

        transformed_dataframe.to_csv(self.output_path, index=False, sep=';')

    def run(self):
        """
        Este metodo ejecuta los pasos para transformar los datos de entrada
        y escribirlos en un archivo de salida.
        """

        df_raw = self.read_data()
        df_transformed = self.data_transformation(df_raw)
        self.write_prepared_data(df_transformed)

if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = r"..\data\Train_BigMart.csv",
                                output_path =
                                r"..\data\Transformed\Train_BigMart_Prepared.csv") \
                                .run()
