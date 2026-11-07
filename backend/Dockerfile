FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "--access-logfile", "-", "--error-logfile", "-", "app:superkart_api"]
