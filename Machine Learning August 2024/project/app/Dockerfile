FROM python:3.12.5

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1

EXPOSE 8000

WORKDIR /django

RUN pip3 install --upgrade pip

RUN pip3 install numpy pandas scikit-learn imblearn xgboost lightgbm cloudpickle joblib google-generativeai django-cors-headers crispy-bootstrap4 gunicorn

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "ECEC.wsgi:application"]
