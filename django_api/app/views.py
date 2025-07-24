from rest_framework.views import APIView
from rest_framework.response import Response
from models.prophet_model import ProphetModel
import polars as pl

class ForecastAPIView(APIView):
    def post(self, request):
        # 1. Configurar modelo
        model = ProphetModel(interval='1d')
        model.custom_seasonalities(request.data.get('seasonalities', []))
        
        # 2. Entrenar/predicción
        if 'new_data' in request.data:
            df = pl.DataFrame(request.data['new_data'])
            model.new_training(df)
        
        # 3. Devolver predicción
        forecast = model.predict_next(days=7).to_dicts()
        return Response({"forecast": forecast})