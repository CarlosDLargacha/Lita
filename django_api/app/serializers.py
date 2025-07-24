from rest_framework import serializers
from datetime import datetime
import polars as pl
import pandas as pd

class SeasonalitySerializer(serializers.Serializer):
    """Serializador para personalizar estacionalidades"""
    name = serializers.CharField(max_length=50)
    period = serializers.FloatField(min_value=1)
    fourier_order = serializers.IntegerField(min_value=1)

class HolidaySerializer(serializers.Serializer):
    """Serializador para días festivos personalizados"""
    holiday = serializers.CharField(max_length=100)
    ds = serializers.DateField()
    lower_window = serializers.IntegerField(default=0)
    upper_window = serializers.IntegerField(default=0)

class TrainingDataSerializer(serializers.Serializer):
    """Serializador para datos de entrenamiento"""
    ds = serializers.ListField(
        child=serializers.DateTimeField(),
        allow_empty=False
    )
    y = serializers.ListField(
        child=serializers.FloatField(),
        allow_empty=False
    )
    
    def validate(self, data):
        """Valida que las fechas y valores tengan la misma longitud"""
        if len(data['ds']) != len(data['y']):
            raise serializers.ValidationError("Las listas 'ds' y 'y' deben tener la misma longitud")
        return data
    
    def to_internal_value(self, data):
        """Convierte a DataFrame de Polars"""
        validated_data = super().to_internal_value(data)
        return pl.DataFrame({
            'ds': pd.to_datetime(validated_data['ds']),
            'y': validated_data['y']
        })

class ForecastInputSerializer(serializers.Serializer):
    """Serializador para requests de forecasting"""
    data = TrainingDataSerializer(required=False)
    interval = serializers.ChoiceField(
        choices=['15m', '1h', '1d'],
        default='1d'
    )
    days = serializers.IntegerField(
        min_value=1,
        max_value=365,
        default=7
    )
    seasonalities = SeasonalitySerializer(many=True, required=False)
    holidays = HolidaySerializer(many=True, required=False)
    model_path = serializers.CharField(required=False)

class ValidationSerializer(serializers.Serializer):
    """Serializador para validación cruzada"""
    initial = serializers.CharField(default='365 days')
    period = serializers.CharField(default='90 days')
    horizon = serializers.CharField(default='30 days')
    model_path = serializers.CharField(required=False)

class ModelSaveSerializer(serializers.Serializer):
    """Serializador para guardar modelos"""
    path = serializers.CharField()
    checkpoint_name = serializers.CharField(default='model')