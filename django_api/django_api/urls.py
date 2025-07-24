from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse
from rest_framework.documentation import include_docs_urls
from rest_framework.schemas import get_schema_view

def home(request):
    return HttpResponse("Bienvenido a la API de Forecasting")

urlpatterns = [
    path('', home),
    path('admin/', admin.site.urls),
    path('api/', include('app.urls')),  # Rutas de tu app de forecasting
    path('docs/', include_docs_urls(title='Forecasting API')),
    path('schema/', get_schema_view(
        title="Forecasting API",
        description="API para predicciones temporales",
        version="1.0.0"
    ), name='openapi-schema'),
]