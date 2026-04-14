FROM python:3.11-slim
WORKDIR /app
COPY services/inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/common/ ./common/
COPY services/inference/ ./
ENV PYTHONUNBUFFERED=1
CMD ["python", "infer.py"]
