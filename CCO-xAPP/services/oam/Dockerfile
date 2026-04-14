FROM python:3.11-slim
WORKDIR /app
COPY services/oam/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY services/oam/ ./
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
