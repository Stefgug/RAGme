FROM python:3.11-slim

WORKDIR /app

# Copy shared utilities
COPY shared/ /app/shared/

# Copy service files
COPY services/ingestion/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ingestion/ .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
