FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE $PORT

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
