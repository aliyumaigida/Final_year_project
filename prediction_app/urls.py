from django.urls import path
# from . import views
from .views import predict_page, about_page

urlpatterns = [
    path('', predict_page, name='prediction'),  # root of prediction_app
    path('about/', about_page, name='about'), 
]


