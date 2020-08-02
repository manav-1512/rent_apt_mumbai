from django.db import models
from django import forms
# Create your models here.

PLACE_CHOICES = [(3,'All Areas'),(0,'Suburbs'),(1,'Downtown'),(2,'Posh')]

PRICE_CHOICES = [(3,'All'),(0,'Economical Apts'),(1,'Comfortable Apts'),(2,'Luxurious Apts')]

ROOM_CHOICES = [(0,'All'),(1,'1'),(2,'2'),(3,'3'),(4,'4'),(5,'5')]

class FilterForm(forms.Form):
    placeprices = forms.ChoiceField( initial = PLACE_CHOICES[0], label = 'Areas',choices = PLACE_CHOICES)
    prices = forms.ChoiceField(initial= PRICE_CHOICES[0], label = 'PriceRanges',choices = PRICE_CHOICES)
    rooms = forms.ChoiceField(initial=ROOM_CHOICES[0],choices = ROOM_CHOICES)

class LocalFilterForm(forms.Form):
    placeprices = forms.ChoiceField( initial = PLACE_CHOICES[0], label = 'Areas',choices = PLACE_CHOICES)
    # prices = forms.ChoiceField(initial= PRICE_CHOICES[0],choices = PRICE_CHOICES) 
    

class AreaFilterForm(forms.Form):
    # placeprices = forms.ChoiceField( initial = PLACE_CHOICES[0],choices = PLACE_CHOICES)
    prices = forms.ChoiceField(initial= PRICE_CHOICES[0], label = 'PriceRanges',choices = PRICE_CHOICES) 