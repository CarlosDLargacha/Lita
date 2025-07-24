from models.prophet_model import ProphetModel
from utils.data_utils import DataUtils
from metrics.metrics_data import Metrics
import pandas as pd

class ForecastPipeline:
    def __init__(self):
        """
        Inicializa la clase ForecastPipeline.
        """
        self.file_path_new_data: str = "data/datasets/flights/daily/2024.csv"  # Ruta del archivo con los nuevos datos
        self.model = ProphetModel("1d")  # Instancia de ProphetModel
        self.metrics = None  # Métricas de validación cruzada
        self.data_utils = DataUtils()  # Utilidades para manejar datos

    def run(self): 
        """
        Ejecuta el pipeline completo: entrenamiento, validación cruzada, métricas y predicción.
        """
        #cargar dataframe para realizar en entrenamiento inicial        
        df = self.data_utils.load_data_from_file(self.file_path_new_data, "1d")
        
        self.model.custom_seasonalities = [
            {
                'name': 'weekly_enhanced',
                'seasonal_periods': 7, 
                'fourier_order': 12,   
                'prior_scale': 15.0,
                'mode': 'multiplicative'
            }]
        
        self.model.custom_holidays = pd.DataFrame({
            'holiday': ['feriado_personalizado', 'evento_especial'],
            'ds': pd.to_datetime(['2023-12-25', '2023-11-24']),  # Fechas de los eventos
            'lower_window': [-1, 0],  # 1 día antes solo para el primer evento
            'upper_window': [1, 2]    # 1 día después y 2 días después respectivamente
        })
        
        #entrnamiento inicial
        self.model.new_training(df)
        
        #obtener dtaframe de la predicción
        prediction = self.model.predict_next(days=10)
        print(prediction)
        
        #mostrar gráfico de la predicción
        self.model.show_prediction()
        
        #Validación cruzada
        df_cv = self.model.p_cross_validation('388 days', '3days', '3days')
            
        #Métricas
        self.metrics = Metrics(df_cv)
        self.metrics.save_metrcis()
        self.metrics.show_RMSE()
        self.metrics.show_MAPE()
        self.metrics.show_Coverage()
        self.metrics.show_SMAPE()
        self.metrics.show_MDAPE()
        self.metrics.show_comparison(self.model.prophet_model.model)