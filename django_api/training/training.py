from abc import ABC, abstractmethod

class Training(ABC):
    """
    Clase abstracta que define la interfaz para el entrenamiento de modelos.
    """
    
    @abstractmethod
    def full_training(self, new_data, interval: str):
        """
        Método abstracto para el entrenamiento completo del modelo.
        
        :param file_path_new_data: Ruta del archivo con los nuevos datos.
        :param interval: Intervalo de tiempo ('15m' para 15 minutos, '1h' para 1 hora).
        """
        pass
    
    @abstractmethod
    def incremental_training(
        self,
        file_path_new_data,
        file_path_old_data: str,
        file_path_old_model,
        interval: str,
    ):
        """
        Método abstracto para el entrenamiento incremental del modelo.
        
        :param file_path_new_data: Ruta del archivo con los nuevos datos.
        :param file_path_old_data: Ruta del archivo con los datos antiguos.
        :param file_path_old_model: Ruta del archivo con el modelo antiguo.
        :param interval: Intervalo de tiempo ('15m' para 15 minutos, '1h' para 1 hora).
        """
        pass