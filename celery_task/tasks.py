import json
import random
import requests
import numpy as np
import matplotlib.pyplot as plt
from celery import Task, Celery

class PredTask(Task):
    """Celery task class for tf serving predictions"""
    def __init__(self):
        super().__init__()
    
    def predict(self, image, model):
        prediction_url = f'http://localhost:8501/v1/models/{model}:predict'
        data = json.dumps({"signature_name": "serving_default", 
                    "instances": image})
        headers = {"content-type": "application/json"}
        json_response = requests.post(prediction_url, data=data, headers=headers)
        predictions = json.loads(json_response.text)['predictions']
        return predictions 

    def run(self, data, model):
        preds = self.predict(data, model)

        return preds
