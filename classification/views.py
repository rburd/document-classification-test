# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template import loader
from sklearn.externals import joblib
from forms import ClassifyForm

from django.shortcuts import render

# Create your views here.

def index(request):
    if (request.method == 'POST'):
        form = ClassifyForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            #process submitted text

            #Todo: encode words into redirected url
            return HttpResponseRedirect('/classification')

    else:
        form = ClassifyForm()
        return render(request, 'classification/index.html', {'form': form})

def classify(request):
    #load classifier
    classifier = joblib.load("model.p", "wb+")
    template = loader.get_template('classication/classify.html')

    #get words value and predict
    context = {'here': 'there', }
    return HttpResponse(template.render(context, request))
