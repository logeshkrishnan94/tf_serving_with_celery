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

# def show(idx, title):
#     plt.figure()
#     plt.imshow(test_images[idx].reshape(28,28))
#     plt.axis('off')
#     plt.title('\n\n{}'.format(title), fontdict={'size': 16})

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# for i in range(0,3):
#   show(i, 'The model thought this was a {} (class {}), and it was actually a {} (class {})'.format(
#     class_names[np.argmax(predictions[i])], np.argmax(predictions[i]), class_names[test_labels[i]], test_labels[i]))