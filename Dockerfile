FROM python:3.12-slim

WORKDIR /app

# Instalar uv y dependencias del sistema (para Prophet)
RUN pip install uv && apt-get update && apt-get install -y gcc python3-dev

# Copiar e instalar librería
COPY ./ForecastingLita /app/ForecastingLita
RUN uv pip install -e /app/ForecastingLita
RUN uv pip install 'u8darts[prophet]'

# Instalar dependencias de Django
COPY ./django_api/requirements.txt .
RUN uv pip install -r requirements.txt

# Copiar proyecto Django
COPY ./django_api /app

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "django_api.wsgi:application"]