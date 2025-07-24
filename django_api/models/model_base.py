from abc import ABC, abstractmethod

class ModelBase(ABC):
    """
    Clase abstracta que define la interfaz para los modelos de predicción.
    """
    
    def __init__(self, model_route: str = None):
        """
        Inicializa la clase base con una ruta opcional para cargar el modelo.
        
        :param model_route: Ruta del archivo del modelo.
        """
        self.model_route = model_route
    
    @abstractmethod
    def new_training(self, new_data):
        """
        Método abstracto para el entrenamiento completo del modelo.
        
        :param file_path_new_data: Ruta del archivo con los nuevos datos.
        :param interval: Intervalo de tiempo ('15m' para 15 minutos, '1h' para 1 hora).
        """
        pass
    
    @abstractmethod
    def incremental_training(self, new_data, file_path_old_data):
        """
        Método abstracto para el entrenamiento incremental del modelo.
        
        :param file_path_new_data: Ruta del archivo con los nuevos datos.
        :param file_path_old_data: Ruta del archivo con los datos antiguos.
        :param file_path_old_model: Ruta del archivo con el modelo antiguo.
        :param interval: Intervalo de tiempo ('15m' para 15 minutos, '1h' para 1 hora).
        """
        pass
    
    @abstractmethod
    def p_cross_validation(self, initial_d, period_d, horizon_d):
        """
        Método abstracto para la validación cruzada del modelo.
        
        :param initial_d: Período inicial para la validación cruzada.
        :param period_d: Período entre cortes de validación cruzada.
        :param horizon_d: Horizonte de predicción.
        """
        pass
    
    def predict_next(self, days: int):
        """
        Realiza una predicción para los próximos "days" días.
        
        :param days: Número de días a predecir.
        :return: DataFrame de Polars con las predicciones.
        """
        
        pass
    
    def predict_range(self, initial_date: str, end_date: str):
        """
        Realiza una predicción en un rango de fechas específico.
        
        :param initial_date: Fecha inicial en formato 'YYYY-MM-DD'.
        :param end_date: Fecha final en formato 'YYYY-MM-DD'.
        :return: DataFrame de Polars con las predicciones.
        """
        
        pass
    
    @abstractmethod
    def show_prediction(self):
        """
        Método abstracto para mostrar un gráfico de la predicción.
        """
        pass