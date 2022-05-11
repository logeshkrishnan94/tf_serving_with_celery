from tensorflow.keras import datasets
from celery_task.celery_app import predict_task

# Loading the mnist dataset
fashion_mnist = datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

predict_result = predict_task.delay(data=test_images[0:3].tolist(), model="simple_model")
res = predict_result.get()
print(res)

