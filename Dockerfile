# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "labs/10_agent_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
