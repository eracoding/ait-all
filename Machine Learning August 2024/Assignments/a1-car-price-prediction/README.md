# FastAPI ML for Car Price prediction

This repository was developed with regard of the assignment 1, task 3 in the Machine Learning course at Asian Institute of Technology (AIT).
The task was car price prediction (regression) and allow user to input the features for car. Therefore, it was developed via FastAPI + simple form that accepts user input.

## Developed Web API

This web API is a nice template to use in the future related tasks and can be used easily.

There are only two endpoints:
1. '/api/v1' - accepts only GET request - is a form for user inputting necessary features to predict car price.
2. '/api/v1/predict' - accepts only POST request - is predictor endpoint for car price

For ML model definition, see the file:
```
app/services/model.py
```

## Statistics regarding model performance on test set:
```
MSE:  53802638404.5708
RMSE:  231953.95751004294
MAE:  71856.95767761378
Mean Percentage Error: 15.833384145330337
Cosine Similarity: 0.9795473733102381
```
So doing 'square' is bad idea, since the distance over target and predicted values is increased. Therefore, mae is perfect, and seems ok.

## Running Web API

### Public server
I have deployed the web api in amazon EC2 service. Try to check if this running by the following url:
```
http://13.239.96.254/api/v1/
```

FYI, at the time of submission of assignment, it was working. :)
If by the time of assignment check it stopped working, just email me or let me know, I can make it up if it goes down.

### Local

**Notes:**
1. You need to rename .env.example -> .env
2. Running locally without docker requires you to install [poetry](https://python-poetry.org/docs/) (suggested, on how to install you can proceed with [link](https://python-poetry.org/docs/)) or you can proceed with installing requirements to local environment.

**When you want to run the web server, you need to go inside code folder, and execute those commands.**

We can run the shell itself:
```sh
$ sh shell/run.sh
```
Or run the command in terminal:
```sh
$ poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

### Docker
```sh
$ docker build -f Dockerfile -t app .
$ docker run -p 9000:9000 --rm --name app -t -i app
```

### Docker Compose

```sh
$ docker compose up --build
```

## Development
### Run Tests and Linter

```
$ poetry run tox
```

## Reference

- [tiangolo/full\-stack\-fastapi\-postgresql: Full stack, modern web application generator\. Using FastAPI, PostgreSQL as database, Docker, automatic HTTPS and more\.](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [eightBEC/fastapi\-ml\-skeleton: FastAPI Skeleton App to serve machine learning models production\-ready\.](https://github.com/eightBEC/fastapi-ml-skeleton)