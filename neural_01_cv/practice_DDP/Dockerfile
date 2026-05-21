FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir torch torchvision

COPY train.py /app/train.py

CMD ["python", "train.py"]