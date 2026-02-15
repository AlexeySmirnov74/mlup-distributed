FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only application sources
COPY app.py /app/app.py
COPY workers.py /app/workers.py
COPY common_app.py /app/common_app.py
#COPY metrics.py /app/metrics.py
COPY prometheus.yml /app/prometheus.yml
COPY mlup /app/mlup

EXPOSE 8009

CMD ["python", "app.py"]
