#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from models.prophet_model import ProphetModel  # Importar tu modelo
import polars as pl

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
    
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # Comandos personalizados para interactuar con ProphetModel
    if len(sys.argv) > 1 and sys.argv[1] == "forecast":
        handle_forecast_command(sys.argv[2:])
    else:
        execute_from_command_line(sys.argv)

def handle_forecast_command(args):
    """Maneja comandos personalizados de forecasting."""
    if not args or args[0] == '--help':
        print("Uso: python manage.py forecast [comando]")
        print("Comandos disponibles:")
        print("  train <ruta_csv> --interval=15m|1h|1d  # Entrena un modelo")
        print("  predict <días> --model=<ruta_modelo>   # Genera predicción")
        return

    # Ejemplo: Implementación de comando 'train'
    if args[0] == 'train':
        import polars as pl
        interval = '1d'  # Valor por defecto
        if '--interval' in args:
            interval = args[args.index('--interval') + 1]

        df = pl.read_csv(args[1])
        model = ProphetModel(interval=interval)
        model.new_training(df)
        print(f"Modelo entrenado con intervalo {interval}")

    # Ejemplo: Comando 'predict'
    elif args[0] == 'predict':
        model_path = None
        if '--model' in args:
            model_path = args[args.index('--model') + 1]

        model = ProphetModel(interval='1d')
        if model_path:
            model.load_model(model_path)

        prediction = model.predict_next(days=int(args[1]))
        print(prediction)

if __name__ == '__main__':
    main()