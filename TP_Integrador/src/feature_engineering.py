"""
feature_engineering.py

DESCRIPCIÓN: Este script contiene la clase FeatureEngineeringPipeline, 
que se encarga de realizar la ingeniería de features sobre
los datos de entrada y escribir el resultado en un archivo
de salida.

AUTOR: Ezequiel Scordamaglia y Santiago González Achaval
FECHA: 17/07/2023
"""

import argparse
import logging as log
import pandas as pd
import numpy as np

log.basicConfig(
    filename='./feat_ing.log',
    level=log.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

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
        try:
            log.info("Iniciando la lectura de datos desde %s", self.input_path)
            pandas_df = pd.read_csv(self.input_path)
            log.info("Datos leídos: Filas=%d, Columnas=%d", pandas_df.shape[0], pandas_df.shape[1])
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e_lectura:
            log.info("Error %s al importar dataframe", e_lectura)
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

        # FEATURES ENGINEERING: para los años del establecimiento
        # Cálculo de años de vida de la tienda en base al año de establecimiento
        # y el año actual (se asume que es data del actual año 2020)
        log.info("Aplicando transformación de años de vida de la tienda")
        df_transformed['Outlet_Establishment_Year'] = 2020 - \
            df_transformed['Outlet_Establishment_Year']

        log.info("Unificando etiquetas para 'Item_Fat_Content")
        # LIMPIEZA: Unificando etiquetas para 'Item_Fat_Content'
        df_transformed['Item_Fat_Content'] = df_transformed['Item_Fat_Content'].replace(
            {'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        log.info("Limpieza de faltantes en el peso de los productos")
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

        log.info("Limpieza de faltantes en el tamaño de tiendas")
        # LIMPIEZA: de faltantes en el tamaño de las tiendas
        for outlet in outlets:
            df_transformed.loc[
                df_transformed['Outlet_Identifier'] == outlet, 'Outlet_Size'
                ] =  'Small'

        log.info("Asignación de nueva categorías para 'Item_Fat_Content")
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

        log.info("Creando categorías para 'Item_Type")
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

        log.info("Asignación de nueva categorías para Item_Fat_Content")
        # FEATURES ENGINEERING: asignación de nueva categorías para 'Item_Fat_Content'
        df_transformed.loc[
            df_transformed['Item_Type'] == 'Non perishable', 'Item_Fat_Content'
            ] = 'NA'

        log.info("Codificando los niveles de precios de los productos")
        # FEATURES ENGINEERING: Codificando los niveles de precios de los productos
        df_transformed['Item_MRP'] = pd.qcut(
            df_transformed['Item_MRP'], 4, labels = [1, 2, 3, 4]
            )

        log.info("Se eliminan columnas 'Item_Type' y 'Item_Fat_Content'")
        # FEATURES ENGINEERING: Se eliminan las columnas que no se utilizarán
        df_transformed = df_transformed.drop(columns=['Item_Type', 'Item_Fat_Content'])

        log.info("Codificación de variables ordinales")
        # FEATURES ENGINEERING: Codificación de variables ordinales
        df_transformed['Outlet_Size'] = df_transformed['Outlet_Size'] \
            .replace({'High': 2, 'Medium': 1, 'Small': 0})

        df_transformed['Outlet_Location_Type'] = df_transformed['Outlet_Location_Type'] \
            .replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})

        log.info("Codificación de variables nominales")
        # FEATURES ENGINEERING: Codificación de variables nominales
        df_transformed = pd.get_dummies(df_transformed, columns=['Outlet_Type'], dtype=int)

        log.info("Eliminacion de columnas 'Item_Identifier' y 'Outlet_Identifier'")
        # FEATURES ENGINEERING: Eliminacion de columnas
        df_transformed = df_transformed.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        # Mover al final la columna de Item_Outlet_Sales
        if 'Item_Outlet_Sales' in df_transformed.columns:
            # Mover columna al final
            cols = list(df_transformed.columns)
            cols.remove('Item_Outlet_Sales')
            cols.append('Item_Outlet_Sales')
            df_transformed = df_transformed[cols]

        log.info("Agregando columna de 'Item_Outlet_Sales' si no existe")
        # Agregar columna de Item_Outlet_Sales si no existe
        if 'Item_Outlet_Sales' not in df_transformed.columns:
            df_transformed['Item_Outlet_Sales'] = np.nan
        log.info("Dataframe transformed")
        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        Este método se encarga de escribir los datos de salida
        en un archivo csv.
        
        :param transformed_dataframe: DataFrame de pandas transformado.
        """
        try:
            transformed_dataframe.to_csv(self.output_path, index=False, sep=',')
            log.info("Dataframe transformado exportado en: %s",(self.output_path))
        except (FileNotFoundError, PermissionError, pd.errors.DtypeWarning) as e_escritura:
            log.info("An error occurred while writing to CSV: %s", str(e_escritura))
    def run(self):
        """
        Este metodo ejecuta los pasos para transformar los datos de entrada
        y escribirlos en un archivo de salida.
        """

        df_raw = self.read_data()
        df_transformed = self.data_transformation(df_raw)
        self.write_prepared_data(df_transformed)

if __name__ == "__main__":
    log.info("Script de feature engineering Iniciado")
    parser = argparse.ArgumentParser()
    parser.add_argument('modo', type=str, help='Modo de ejecución: train o test')
    args = parser.parse_args()

    modo = args.modo

    if modo == 'train':
        IN_PATH = r"..\data\Train_BigMart.csv"
        OUT_PATH = r"..\data\Transformed\Train_BigMart_Prepared.csv"
    else:
        IN_PATH = r"..\data\Test_BigMart.csv"
        OUT_PATH = r"..\data\Transformed\Test_BigMart_Prepared.csv"

    FeatureEngineeringPipeline(input_path = IN_PATH,
                                output_path = OUT_PATH).run()
    log.info("Script de feature engineering completado")
