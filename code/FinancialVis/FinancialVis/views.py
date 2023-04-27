from django.http import HttpResponse
from django.http.response import HttpResponseRedirect
from django.shortcuts import render


def index(request):
    return render(request, 'index.html')


def wordCloud(request):
    return render(request, 'wordCloud.html')
