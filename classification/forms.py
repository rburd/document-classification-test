from django import forms

class ClassifyForm(forms.Form):
    words = forms.CharField(label='Text', max_length=100000)