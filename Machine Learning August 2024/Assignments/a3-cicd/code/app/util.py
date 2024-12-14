import joblib
import pickle

def save(filename:str, obj:object):
    with open(filename, 'wb') as file:  # Open the file in write-binary mode
        pickle.dump(obj, file)
    # joblib.dump(obj, filename)


def load(filename:str):
    with open(filename, 'rb') as file:  # Open the file in read-binary mode
        return pickle.load(file)
    # joblib.load(filename)

import os


model_name = os.environ['APP_MODEL_NAME']
def load_model(stage='Staging'):
    cached_path = os.path.join('./app', 'ml_models', stage)
    if os.path.exists(cached_path) == False:
        os.makedirs(cached_path)
    
    path = os.path.join(cached_path, model_name)
    print(model_name, os.path.exists(path), path)
    if os.path.exists(path) == False:
        import mlflow
        model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{stage}")
        print("MODEL: ", model)
        save(filename=path, obj=model)
    
    model = load(path)
    return model, True

def register_model_to_production():
    from mlflow.client import MlflowClient
    client = MlflowClient()
    
    for model in client.get_registered_model('st125457-a3-model').latest_versions:
        if model.current_stage == 'Staging':
            version = model.version
            client.transition_model_version_stage(
                name=model_name, version=version, stage='Production', archive_existing_versions = True
            )