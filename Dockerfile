FROM tensorflow/tensorflow:2.9.2-gpu

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

WORKDIR /app

COPY mnist_cnn.py /app
COPY mnist_cnn_speed.py /app
