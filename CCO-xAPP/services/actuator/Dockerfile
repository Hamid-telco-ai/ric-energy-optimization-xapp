FROM python:3.11-slim
WORKDIR /app

COPY ./services/actuator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./services/actuator /app
COPY ./services/common /app/common

ENV PYTHONUNBUFFERED=1
CMD ["python", "actuator.py"]
