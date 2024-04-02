from django.urls import path
from . import views
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

schema_view = get_schema_view(
    openapi.Info(
        title="Verifai API",
        default_version='v1',
        description="API for verifying facts and generating LLM responses",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="cole.thomas.agard@gmail.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=[permissions.AllowAny],
)

urlpatterns = [
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('swagger.<format>/', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('verify/', views.verify_fact, name='verify_fact'),
    path('get_task_status/<str:task_id>/', views.get_task_status, name='get_task_status')
]
