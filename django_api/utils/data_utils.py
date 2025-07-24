import polars as pl
import json
from datetime import datetime, timedelta

class DataUtils:
    
    def load_data_from_file(self, file_path: str, interval: str)-> pl.DataFrame:
        
        """
        Carga los datos de un dataframe almacenado en un archivo .json o csv
        
        :param file_path: ruta al archivo con el dataframe.
        :param interval: intervalo de tiempo en la datos a cargar.
        :return: Dataframe de polars con los datos del archivo en file_path.
        """
        
        # Cargar y procesar datos
        if file_path.endswith(".json"):
            df = self.json_to_dataframe(file_path)
        elif file_path.endswith(".csv"):
            df = self.csv_to_dataframe(file_path)
        
        if interval == "15m": return self.create_dataset(df, interval)
        if interval == "1h": return self.create_hourly_dataset(df)
        if interval == "1d": return self.create_daily_dataset(df)
        
        else: raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")
        
    @staticmethod
    def json_to_dataframe(file_path: str)-> pl.DataFrame:
        """
        Carga un archivo JSON y lo convierte en un DataFrame de Polars.
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        dataframe = pl.DataFrame(data)
        return dataframe

    @staticmethod
    def csv_to_dataframe(file_path: str)-> pl.DataFrame:
        """
        Carga un archivo CSV y lo convierte en un DataFrame de Polars.
        """
        dataframe = pl.read_csv(file_path)
        return dataframe

    @staticmethod
    def load_old_data(file_path: str) -> pl.DataFrame:
        """
        Carga datos antiguos desde un archivo JSON.
        """
        df = pl.read_json(file_path)
        return df

    @staticmethod
    def create_dataset(dataframe, interval: str = '15m')-> pl.DataFrame:
        """
        Crea un dataset para el modelo según el intervalo especificado.
        """
        dataframe = dataframe.sort('time')
        dataframe = dataframe.with_columns(
            pl.col('time').str.to_datetime(format='%Y-%m-%dT%H:%M:%SZ').alias('time')
        )
        dataframe = dataframe.set_sorted('time')
        
        if interval == '15m':
            dataframe = dataframe.upsample(time_column='time', every='15m').interpolate()
        elif interval == '1h':
            dataframe = dataframe.upsample(time_column='time', every='1h').interpolate()
        elif interval == '1d':
            dataframe = dataframe.upsample(time_column='time', every='1d').interpolate()
        else:
            raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")
        
        dataframe = dataframe.rename({'time': 'ds', 'load': 'y'})
        dataframe = dataframe.with_columns(
            pl.col('ds').dt.replace_time_zone(None).alias('ds')
        )
        return dataframe

    @staticmethod
    def create_hourly_dataset(dataframe)-> pl.DataFrame:
        """
        Crea un dataset para el modelo con datos horarios.
        """
        dataframe = dataframe.sort('Time')
        dataframe = dataframe.with_columns(
            pl.col('Time').str.to_datetime(format='%Y-%m-%d %H:%M:%S').alias('ds')
        )
        dataframe = dataframe.rename({'Total': 'y'})
        dataframe = dataframe.with_columns(
            pl.col('ds').dt.replace_time_zone(None).alias('ds')
        )
        return dataframe

    @staticmethod
    def create_daily_dataset(dataframe) -> pl.DataFrame:
        """
        Crea un dataset para el modelo con datos diarios.
        """
        dataframe = dataframe.sort('Time')
        dataframe = dataframe.with_columns(
            pl.col('Time').str.to_datetime(format='%Y-%m-%d %H:%M:%S').alias('ds')
        )
        dataframe = dataframe.rename({'Total': 'y'})
        dataframe = dataframe.with_columns(
            pl.col('ds').dt.replace_time_zone(None).alias('ds')
        )
        #dataframe = dataframe.groupby('ds').agg(pl.col('y').sum())
        return dataframe

    @staticmethod
    def prediction_dataframe(df, days, interval: str = '15m') -> pl.DataFrame:
        """
        Crea un DataFrame para las predicciones.
        """
        last_date = df['ds'].max()

        if interval == '15m':
            delta = timedelta(minutes=15)
        elif interval == '1h':
            delta = timedelta(hours=1)
        elif interval == '1d':
            delta = timedelta(days=1)
        else:
            raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")

        prediction_dates = []
        current_date = last_date + delta
        end_date = last_date + timedelta(days=days)

        while current_date <= end_date:
            prediction_dates.append(current_date)
            current_date += delta

        prediction_dataset = pl.DataFrame({'ds': prediction_dates})
        return prediction_dataset