# TF Serving with Celery workers
### Create and activate virtual environment
```python3.8 -m venv venv3```
```source venv3/bin/activate```

### Install requirements
```pip install -r requirements.txt```

### Start redis docker for celery backend and broker
```docker run -d -p 6379:6379 redis```

### Start TF serving docker with model location
```docker run -t --rm -p 8501:8501 -v "$(pwd)/models/:/models/" tensorflow/serving --model_config_file=/models/models.config```

### Start celery workers
```celery -A celery_task worker --loglevel=INFO --pool threads```

### Run infer.py for predictions
```python infer.py```

