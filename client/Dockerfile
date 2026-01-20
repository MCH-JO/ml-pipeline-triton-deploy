FROM python:3.8-slim

WORKDIR /app

# build deps (to compile if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
 && rm -rf /var/lib/apt/lists/*

COPY client/requirements.txt /app/client/requirements.txt
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -r /app/client/requirements.txt

COPY client /app/client
ENV PYTHONPATH=/app

CMD ["python", "-m", "pytest", "-q", "client/tests"]
