from django import forms
from predictor.content import *

style = forms.Select(attrs={'class': 'form-control'})

class DetectionForm(forms.Form):
    applicant_gender = forms.ChoiceField(choices=[('M', 'Male'), ('F', 'Female')], widget=style)
    owned_car = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], widget=style)
    owned_realty = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], widget=style)
    total_children = forms.IntegerField(min_value=0, max_value=20, required=True)
    total_income = forms.IntegerField(min_value=0, max_value=9999999, required=True)
    income_type = forms.ChoiceField(choices=income_type, widget=style)
    education_type = forms.ChoiceField(choices=education_type, widget=style)
    family_status = forms.ChoiceField(choices=family_status, widget=style)
    housing_type = forms.ChoiceField(choices=housing_type, widget=style)
    # owned_work_phone = forms.ChoiceField(choices=[(0, 'Yes'), (1, 'No')], widget=style)
    # owned_phone = forms.ChoiceField(choices=[(0, 'Yes'), (1, 'No')], widget=style)
    # owned_Email = forms.ChoiceField(choices=[(0, 'Yes'), (1, 'No')], widget=style)
    job_title = forms.ChoiceField(choices=job_title, widget=style)
    total_family_members = forms.IntegerField(min_value=0, max_value=20, required=True)
    applicant_age = forms.IntegerField(min_value=0, max_value=120, required=True)
    years_of_working = forms.IntegerField(min_value=0, max_value=60, required=True)


