# Asian Institute of Technology - Assignment 3 - Car Prediction

## Overview
In this assignment, we implemented class based model of LogisticRegression with turning regression problem into classification (splitted training dataset into 4 classes). We also registered our models into MLflow, and implemented ci/cd which automates the integration and deployment part. We used github actions to make the things work. We also implemented the classification report.

The access for the running web platform is [https://fastapi-st125457.ml.brain.cs.ait.ac.th/](https://fastapi-st125457.ml.brain.cs.ait.ac.th/)

## What was implemented
We extended our jupyter notebook developed in the second assignment, added classful implementation of Logistic regression, different features, implemented classification report. The task was to implement classification model, and the ci/cd system for our web platform.

## Server part
While integrating the custom model into fast-api environment, I have faced issues with unpickling the model. It took some time to come up with solution as storing model weights (which I think is real-case scenario), and loading them inside fast-api startup handling. I decided to not duplicate the html document so that each model will have its own page, but rather I prefered to let user choose inside the form. I think this logic is fine. Since we are downloading the model from mlflow, the fastapi has the problem with loading this model via pickle. Therefore, I am only downloading the weights of the model and using it to run the model.

![alt text](.github/assets/img.jpg)
