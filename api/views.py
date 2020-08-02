from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd
from .models import FilterForm, LocalFilterForm, AreaFilterForm


# Create your views here.
def home(request):
    return render(request, 'home.html',)

def contributors(request):
    return render(request, 'contributors.html')    
  
def data(request):
    if request.method == 'POST':
        form = FilterForm(request.POST)
        if form.is_valid():
            df = pd.read_csv('api/static/data/'+str(form.cleaned_data['placeprices'])+','+str(form.cleaned_data['prices'])+','+str(form.cleaned_data['rooms'])+'.csv')
        # else:
        #     df = pd.read_csv('api/static/data/3,3,0.csv')       
    else:
        form = FilterForm()
        df = pd.read_csv('api/static/data/3,3,0.csv')  
    table = df.to_html()
    table = table[:table.find('<tr')] + table[table.find('<th>'):table.find('</tr>')] +table[table.find('</thead>'):]
    table = table[:table.find('dataframe')] + "table table-hover" + table[table.find('dataframe')+9:]      
    return render(request, 'data.html', {'table': table , 'form':form})
    
def analysisLoc(request):
    if request.method == 'POST':
        form = LocalFilterForm(request.POST)
        if form.is_valid():
            image = str(form.cleaned_data['placeprices'])+'1.jpg'
        else:
            image = '31.jpg'
    else:
        form = LocalFilterForm()   
        image = '31.jpg'   
    image = "http://127.0.0.1:8000/media/img/" + image      
    return render(request, 'graphL.html',{'image': image, 'form':form})    

def analysisArea(request):
    if request.method == 'POST':
        form = AreaFilterForm(request.POST)
        if form.is_valid():
            image = str(form.cleaned_data['prices'])+'2.jpg'
        else:
            image = '32.jpg'
    else:
        form = AreaFilterForm() 
        image = '32.jpg'
    image = "http://127.0.0.1:8000/media/img/" + image
    return render(request, 'graphA.html',{'image': image, 'form':form})     