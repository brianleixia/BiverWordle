from django.contrib import admin
from django.http.request import validate_host
from django.urls import path, include
from . import views

urlpatterns = [
    path('themeRiver-pyecharts', views.themeRiverVis,
         name='themeRiver-pyecharts'),
]
