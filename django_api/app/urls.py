from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

# Patrones de URL base (sin format_suffix_patterns)
urlpatterns = [
    # Endpoint principal de forecast
    path('forecast/', 
        views.ForecastAPIView.as_view(), 
        name='forecast-api'),
    
    # Endpoint para entrenamiento (descomenta cuando esté implementado)
    # path('forecast/train/', 
    #     views.ForecastTrainView.as_view(), 
    #     name='forecast-train'),
    
    # Endpoint para predicción específica (descomenta cuando esté implementado)
    # path('forecast/predict/<int:days>/', 
    #     views.ForecastPredictView.as_view(), 
    #     name='forecast-predict'),
    
    # Endpoint para validación (descomenta cuando esté implementado)
    # path('forecast/validate/', 
    #     views.ForecastValidateView.as_view(), 
    #     name='forecast-validate'),
    
    # Endpoint para gráficos (descomenta cuando esté implementado)
    # path('forecast/plot/', 
    #     views.ForecastPlotView.as_view(), 
    #     name='forecast-plot'),
]

# Aplicar format_suffix_patterns solo si hay patrones definidos
if urlpatterns:
    urlpatterns = format_suffix_patterns(urlpatterns)