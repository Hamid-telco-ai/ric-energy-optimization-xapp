FROM python:3.11-slim
WORKDIR /app
COPY services/trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/common/ ./common/
COPY services/trainer/ ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "train.py"]
