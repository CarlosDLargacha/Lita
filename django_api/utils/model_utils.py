from datetime import datetime, timedelta
from darts.models import Prophet
import polars as pl
import os

class ModelUtils:
    
    def __init__(self, path_model:str, path_data:str, interval:str):
        self.path_model:str = path_model
        self.path_data:str = path_data
        self.interval:str = interval
        self.number_of_checkpoints:int = 30
    
    def save_model(self, model):
        """
        Guarda el modelo en un archivo.
        
        :param model: Modelo de Prophet.
        """        
        self._save_checkpoints(model)
                
        model_dir = os.path.join(self.path_model, "prophet_model_data")
        
        # Guardar el modelo en un archivo
        model.save(model_dir)
            
    def _save_checkpoints(self, model):
        
        """Guarda versiones de checkpoint del modelo y datos."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # ===== Guardado de checkpoint del modelo =====
        model_dir = os.path.join(self.path_model, 'checkpoints')
        os.makedirs(model_dir, exist_ok=True)
        
        checkpoint_name = f"prophet_model_checkpoint_{timestamp}"
        full_model_path = os.path.join(model_dir, checkpoint_name)
        
        model.save(full_model_path)
        
        # Limpiar checkpoints antiguos 
        ModelUtils._clean_old_files(model_dir, self.number_of_checkpoints)

    @staticmethod
    def _clean_old_files(directory: str, keep_last: int):
        """Elimina archivos antiguos manteniendo solo los 'keep_last' más recientes."""
        try:
            files = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if os.path.isfile(os.path.join(directory, f))]
            
            if len(files) > keep_last:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                for old_file in files[keep_last:]:
                    try:
                        os.remove(old_file)
                    except Exception as e:
                        print(f"Error eliminando {old_file}: {e}")
        except Exception as e:
            print(f"Error limpiando checkpoints en {directory}: {e}")

    @staticmethod
    def load_model(model_path: str):
        """
        Carga un modelo desde un archivo JSON.
        
        :param model_path: Ruta del archivo JSON del modelo.
        :return: Modelo de Prophet.
        """
        return Prophet.load(model_path)
        
    @staticmethod
    def generate_future_dataset(last_date: datetime, days: int, interval: str) -> pl.DataFrame: 
        """
        Genera un DataFrame de Polars con fechas futuras para predicciones.
        
        :param last_date: Última fecha conocida en los datos de entrenamiento.
        :param days: Número de días a predecir.
        :param interval: Intervalo de tiempo ('15m', '1h', '1d').
        :return: DataFrame de Polars con las fechas futuras.
        """
        if interval == '15m':
            delta = timedelta(minutes=15)
        elif interval == '1h':
            delta = timedelta(hours=1)
        elif interval == '1d':
            delta = timedelta(days=1)
        else:
            raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")

        future_dates = []
        try:
            current_date = last_date + delta
            end_date = last_date + timedelta(days=days)
        except:
            last_date = datetime.strptime(last_date, "%Y-%m-%d %H:%M:%S")
            current_date = last_date + delta
            end_date = last_date + timedelta(days=days)

        while current_date <= end_date:
            future_dates.append(current_date)
            current_date += delta

        return pl.DataFrame({'ds': future_dates})

    @staticmethod
    def generate_range_dataset(initial_date: str, end_date: str, interval: str) -> pl.DataFrame:
        """
        Genera un DataFrame de Polars con fechas en un rango específico.
        
        :param initial_date: Fecha inicial en formato 'YYYY-MM-DD'.
        :param end_date: Fecha final en formato 'YYYY-MM-DD'.
        :param interval: Intervalo de tiempo ('15m', '1h', '1d').
        :return: DataFrame de Polars con las fechas en el rango.
        """
        if interval == '15m':
            delta = timedelta(minutes=15)
        elif interval == '1h':
            delta = timedelta(hours=1)
        elif interval == '1d':
            delta = timedelta(days=1)
        else:
            raise ValueError("Intervalo no soportado. Use '15m', '1h' o '1d'.")

        start_date = datetime.strptime(initial_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        date_range = []
        current_date = start_date

        while current_date <= end_date:
            date_range.append(current_date)
            current_date += delta

        return pl.DataFrame({'ds': date_range})