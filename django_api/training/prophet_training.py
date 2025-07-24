import os
import polars as pl
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet
from training.training import Training
from utils.data_utils import DataUtils
from utils.model_utils import ModelUtils
from typing import List, Dict

class ProphetTraining(Training):
    def __init__(self, interval = "15m"):
        self.interval_mapping = {'15m': '15min', '1h': 'hourly', '1d': 'daily'}
        self.dataframe:pl.DataFrame = None
        self.model: Prophet = None
        self.interval:str = interval
        self.old_data:pl.DataFrame = None
        self.data_utils: DataUtils = DataUtils()
        self.model_utils: ModelUtils = ModelUtils(
            os.path.join('data', 'models', 'flights', self.interval_mapping[interval]), 
            os.path.join('data', 'datasets', 'flights', self.interval_mapping[interval]),
            self.interval_mapping[interval]
            )
        self.seasonalities: List[Dict] =  [
            # Estacionalidad semanal reforzada
            {
                'name': 'weekly_enhanced',
                'seasonal_periods': 7, 
                'fourier_order': 12,   
                'prior_scale': 15.0,
                'mode': 'multiplicative'
            },
            
            # Estacionalidad mensual
            {
                'name': 'monthly',
                'seasonal_periods': 30.5,  
                'fourier_order': 8,
                'prior_scale': 10.0
            },
            
            # Estacionalidad trimestral 
            {
                'name': 'quarterly',
                'seasonal_periods': 91.25,  
                'fourier_order': 6,
                'prior_scale': 12.0
            }
        ]
        self.holidays:pd.DataFrame = None

    def save_model(self, model):
        """
        Guarda el modelo y los datos en archivos JSON.
        
        :param model: Modelo de Prophet.
        :param data: DataFrame de Polars con los datos de entrenamiento.
        :param data_path: Ruta para guardar los datos.
        """
        self.model_utils.save_model(model=model)
    
    def load_model(self, file_path_old_model: str):
        
        # Cargar el modelo antiguo
        self.model = self.model_utils.load_model(file_path_old_model)      

    def full_training(self, new_data: pl.DataFrame) -> Prophet:
        
        df = self.dataframe_corrections(new_data)
        
        # Entrenar modelo
        model = Prophet(add_seasonalities=self.seasonalities, country_holidays='US', holidays=self.holidays)
        model.fit(df)
        
        # Guardar modelo y datos
        self.dataframe = new_data  # Guardamos los datos originales
        self.save_model(model)
        
        return model    
    
    def incremental_training(self, new_data: pl.DataFrame) -> Prophet:
        
        try:
            df_new = self.dataframe_corrections(new_data)
            old_prophet = self.model.model
            
            # Configurar nuevo modelo con parámetros del anterior
            new_prophet = Prophet(
                growth=old_prophet.growth,
                changepoint_prior_scale=old_prophet.changepoint_prior_scale,
                seasonality_prior_scale=old_prophet.seasonality_prior_scale,
                holidays_prior_scale=old_prophet.holidays_prior_scale,
                seasonality_mode=old_prophet.seasonality_mode,
                yearly_seasonality=old_prophet.yearly_seasonality,
                weekly_seasonality=old_prophet.weekly_seasonality,
                daily_seasonality=old_prophet.daily_seasonality
            )
            
            new_prophet.fit(df_new)
            
            # 5. Guardar y actualizar datos
            self.dataframe = new_data
            self.save_model(new_prophet)
            self.model = new_prophet
            
            return self.model
            
        except Exception as e:
            print(f"Error en incremental_training: {e}. Fallback a full_training.")
            return self.full_training(new_data)
        
    def dataframe_corrections(self, dataframe: pl.DataFrame):
        
        i_m = {'15m': '15min', '1h': 'H', '1d': 'D'}
        
        df = dataframe.to_pandas()
        
        # Asegurar que la columna de tiempo es datetime y está ordenada
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds')
        
        # Crear rango completo de fechas con frecuencia horaria
        full_range = pd.date_range(
            start=df['ds'].min(),
            end=df['ds'].max(),
            freq=i_m[self.interval]  # Frecuencia horaria
        )
        
        # Rellenar fechas faltantes y valores NaN
        df = (
            df.set_index('ds')
            .reindex(full_range)
            .rename_axis('ds')
            .reset_index()
        )
        
        # Imputar valores faltantes (rellenar NaN)
        df['y'] = df['y'].fillna(method='ffill')
        
        # Convertir a TimeSeries (ahora con datos completos)
        series = TimeSeries.from_dataframe(
            df,
            time_col='ds',
            value_cols='y',
            freq=i_m[self.interval]
        )
                
        return series