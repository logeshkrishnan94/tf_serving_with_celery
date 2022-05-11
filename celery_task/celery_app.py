from __future__ import absolute_import

from celery import Celery
from celery_task.tasks import PredTask

app = Celery('celery_task',
             broker="redis://",
             backend="redis://",
             include=['celery_task.tasks'])


predict_task = app.register_task(PredTask)  

if __name__ == '__main__':
    app.start()