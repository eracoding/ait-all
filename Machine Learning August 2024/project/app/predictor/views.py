import os
import pickle
from typing import Any

import pandas as pd
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.views.generic import TemplateView
from django.views.generic.edit import FormView

from ECEC.settings import PROJECT_ROOT, GOOGLE_API_KEY
from predictor.content import columns, few_shot_prompt
from predictor.forms import DetectionForm
from google import generativeai


class IndexView(TemplateView):
    template_name = "index.html"

class SuccessView(TemplateView):
    template_name = 'success.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        result = self.request.GET.get("result")
        try:
            context["result"] = result

        except ValueError:
            context["result"] = ''

        return context


class PredictFormView(FormView):
    generativeai.configure(api_key=GOOGLE_API_KEY)
    llm = generativeai.GenerativeModel(
        'gemini-1.5-flash-latest',
        generation_config=generativeai.GenerationConfig(
            temperature=0.1,
            top_p=1,
            max_output_tokens=250,
        )
    )

    form_class = DetectionForm
    template_name = 'predict.html'
    loaded_model = pickle.load(open(os.path.join(PROJECT_ROOT, "best_model_pipeline.pkl"), "rb"))
    # loaded_model = joblib.load(os.path.join(PROJECT_ROOT, "best_model_pipeline.joblib"))

    def form_valid(self, form):
        respone = None
        cleaned_data = form.cleaned_data
        data_list = list(cleaned_data.values())
        result = self.predict(data_list)

        bad_user_data = f"""
        **User's Shared Features**:
        - Gender: {cleaned_data.get('applicant')}
        - Owned Car: {cleaned_data.get('owned_car')}
        - Owned Realty: {cleaned_data.get('owned_realty')}
        - Total Children: {cleaned_data.get('total_children')}
        - Total Income (Annually): {cleaned_data.get('total_income')}
        - Income Type: {cleaned_data.get('income_type')}
        - Education Type: {cleaned_data.get('education_type')}
        - Family Status: {cleaned_data.get('family_status')}
        - Housing Type: {cleaned_data.get('housing_type')}
        - Job Title: {cleaned_data.get('job_title')}
        - Total Family Members: {cleaned_data.get('total_family_members')}
        - Applicant Age: {cleaned_data.get('applicant_age')}
        - Years of Working: {cleaned_data.get('years_of_working')}

        **Feature Importance**:
        - Years of Working: 637
        - Total Income: 600
        - Applicant Age: 581
        - Owned Realty: 204
        - Total Children: 176
        - Total Family Members: 174
        - Owned Car: 169

        **Response**:
        YOUR TURN (DO NOT SUGGEST ABOUT CREDIT HISTORY AND SCORE)
        """

        if result == 0:
            print(type(result))
            response = self.llm.generate_content([few_shot_prompt, bad_user_data]).text
        else:
            response = 'Eligible'

        return redirect(f"{reverse_lazy('predictor:success')}?result={response}")

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["results"] = getattr(self, "result", None)
        return context

    def predict(self, data):
        data_df = pd.DataFrame([data], columns=columns)
        return self.loaded_model.predict(data_df)
