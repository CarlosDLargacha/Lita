from models.model_base import ModelBase
from training.prophet_training import ProphetTraining
import matplotlib.pyplot as plt
from darts.models import Prophet
from darts import TimeSeries
from prophet.diagnostics import cross_validation
from utils.model_utils import ModelUtils
from utils.data_utils import DataUtils
import polars as pl
import pandas as pd
import os
import numpy as np
from typing import Dict, List

class ProphetModel(ModelBase):
    def __init__(self, interval):
        """
        Inicializa la clase ProphetModel.
        
        :param model_route: Ruta del archivo del modelo (opcional).
        """
        self.prophet_model: Prophet = None  # Modelo de Prophet
        self.df_cross_validation: pd.DataFrame = None  # DataFrame para validación cruzada
        self.prediction_df: pl.DataFrame = None  # DataFrame con las predicciones
        self.training = ProphetTraining(interval)  # Entrenamiento
        self.dataframe: pl.DataFrame = None  # DataFrame con los datos de entrenamiento
        self.interval: str = interval  # Intervalo de tiempo por defecto

    def set_interval(self, interval: str):
        """
        Establece el intervalo de tiempo para las predicciones, y verifica que el intervalo sea válido.
        
        :param interval: Intervalo de tiempo ('15m', '1h', '1d').
        """
        if interval not in ['15m', '1h', '1d']:
            raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")
        self.interval = interval
        self.training.interval = interval
        self.training.model_utils.interval = interval
        self.training.model_utils.path_data = os.path.join('data', 'datasets', 'flights', self.training.interval_mapping[interval])
        self.training.model_utils.path_model = os.path.join('data', 'models', 'flights', self.training.interval_mapping[interval])
        
    def set_amount_of_checkpoints(self, n: int):
        """
        Establece una nueva cantidad de checkpoints a guardar.        
        :param n: nueva cantidad de checkpoints a guardar.
        """
        self.training.model_utils.number_of_checkpoints = n
        
    def set_model_save_route(self, route:str):
        """
        Establece una nueva ruta donde se guardara el modelo entrenado y sus chekpoints.        
        :param route: ruta nueva donde se guardarán los modelos.
        """
        self.training.model_utils.path_model = route
        
    def set_dataframe_save_route(self, route:str):
        """
        Establece una nueva ruta donde se guardaran los dataframes usados en los entrenamoientos.        
        :param route: ruta nueva donde se guardarán los dataframes.
        """
        self.training.model_utils.path_data = route
        
    def custom_seasonalities(self, custom_seasonalities: List[Dict]):
        """
        param custom_seasonalities: 
        """
        self.training.seasonalities = custom_seasonalities
        
    def custom_holidays(self, custom_holidays: pd.DataFrame):
        """
        param custom_holidays: dataframe con los holidays
        """
        self.training.holidays = custom_holidays

    def new_training(self, new_data: pl.DataFrame) -> Prophet:
        """
        Entrenamiento completo del modelo.       
        :param new_data: Dataframe con los datos para el entrenamiento.
        :return: Modelo entrenado.
        """
        # Entrenar el modelo usando self.interval
        self.prophet_model = self.training.full_training(new_data)
        self.dataframe = new_data
        return self.prophet_model
    
    def incremental_training(self, new_data: str) -> Prophet:
        """
        Entrenamiento incremental del modelo.
        :param new_data: Dataframe con los datos para el entrenamiento.
        :return: Modelo actualizado.
        """
        # Entrenar el modelo usando self.interval
        self.prophet_model = self.training.incremental_training(new_data)
        
        self.dataframe = self.training.dataframe
        return self.prophet_model
    
    def p_cross_validation(self, initial_d, period_d, horizon_d)-> pd.DataFrame: 
        """
        Realiza validación cruzada del modelo.
        
        :param initial_d: Período inicial para la validación cruzada.
        :param period_d: Período entre cortes de validación cruzada.
        :param horizon_d: Horizonte de predicción.
        :return: DataFrame con los resultados de la validación cruzada.
        """
        self.df_cross_validation = cross_validation(self.prophet_model.model, initial=initial_d, period=period_d, horizon=horizon_d)
        return self.df_cross_validation
    
    def predict_next(self, days: int) -> pl.DataFrame:
        """
        Realiza una predicción para los próximos días usando Prophet de Darts.
        
        Args:
            days: Número de días a predecir
            num_samples: Número de muestras para calcular intervalos de confianza
            
        Returns:
            DataFrame de Polars con columnas: ds, yhat, yhat_lower, yhat_upper
        """
        
        if self.prophet_model is None: raise ValueError("El modelo no ha sido entrenado")
        
        intrvl = 1        
        if self.interval == "1h": intrvl = 24
        if self.interval == "15m": intrvl = 24 * 4

        # Generar predicción
        forecast = self.prophet_model.predict_raw( n=days * intrvl )
        
        # Guardar la predicción en un dataframe de polars
        self.prediction_df = pl.DataFrame(forecast[["ds", "yhat", "yhat_upper", "yhat_lower"]])
        
        return self.prediction_df

    def predict_range(self, initial_date: str, end_date: str) -> pl.DataFrame:
        """
        Realiza una predicción en un rango de fechas específico usando el modelo Prophet de Darts.
        
        :param initial_date: Fecha inicial en formato 'YYYY-MM-DD HH:MM:SS'.
        :param end_date: Fecha final en formato 'YYYY-MM-DD HH:MM:SS'.
        :return: DataFrame de Polars con las predicciones.
        """
        # Convertir fechas a timestamp
        start_dt = pd.Timestamp(initial_date)
        end_dt = pd.Timestamp(end_date)
        
        # Verificar que el rango sea válido
        if start_dt > end_dt:
            raise ValueError("La fecha inicial no puede ser posterior a la fecha final")
        
        # Obtener la serie temporal de entrenamiento del modelo
        if not hasattr(self.prophet_model, 'model') or not hasattr(self.prophet_model.model, 'history'):
            raise RuntimeError("El modelo no tiene datos de entrenamiento cargados")
        
        # Obtener la última fecha de entrenamiento
        last_train_date = pd.Timestamp(self.prophet_model.model.history['ds'].iloc[-1])
        
        print(start_dt)
        print(end_dt)
        print(last_train_date)
        
        # Verificar que las fechas de predicción sean posteriores al entrenamiento
        if start_dt <= last_train_date:
            raise ValueError(f"La fecha inicial debe ser posterior a {last_train_date.date()}")
        
        # Calcular el horizonte de predicción necesario
        date_range = pd.date_range(start=last_train_date, end=end_dt, freq=self.interval)
        n = len(date_range) - 1
        
        print(n)
        
        if self.prophet_model is None: raise ValueError("El modelo no ha sido entrenado")

        # Generar predicción
        forecast = self.prophet_model.predict_raw(n=n)
        
        df = forecast[forecast['ds'] >= start_dt]
        
        # Guardar la predicción en un dataframe de polars
        self.prediction_df = pl.DataFrame(df[["ds", "yhat", "yhat_upper", "yhat_lower"]])
        
        return self.prediction_df
    
    def load_model(self, file_path_old_model: str):
        """
        Intenta cargar el modelo desde file_path_old_model si existe.
        
        :param file_path_old_model: Ruta del modelo a cargar.
        """
        
        self.training.load_model(file_path_old_model)

        try:
                self.prophet_model = ModelUtils.load_model(file_path_old_model)
        except Exception as e:
                raise ValueError(f"No se pudo cargar el modelo: {e}")
    
    def show_prediction(self):
        """
        Muestra un gráfico de la predicción (requiere haber realizado una predicción primero).
        """
        if self.prediction_df is None:
            raise ValueError("No hay predicciones para mostrar. Ejecute `predict_next` primero.")

        # Convertir a DataFrame de pandas para plotting
        pred_df = self.prediction_df.to_pandas()
        
        # Crear figura
        plt.figure(figsize=(12, 6))
        
        # Graficar predicción
        plt.plot(pred_df['ds'], pred_df['yhat'], label='Predicción', color='blue')
        
        # Graficar intervalo de confianza
        plt.fill_between(
            pred_df['ds'], 
            pred_df['yhat_lower'], 
            pred_df['yhat_upper'],
            color='blue', 
            alpha=0.2,
            label='Intervalo 95%'
        )
        
        # Añadir detalles al gráfico
        plt.title('Predicción de Series Temporales')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)
        
        # Rotar etiquetas de fechas para mejor visualización
        plt.xticks(rotation=45)
        
        # Ajustar layout para que no se corten las etiquetas
        plt.tight_layout()
        plt.savefig('forecast.png')
        plt.show()