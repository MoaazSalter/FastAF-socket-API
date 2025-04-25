FROM python:3.9-slim 

RUN apt-get update && apt-get install -y \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["uvicorn", "Server:app", "--host", "0.0.0.0", "--port", "3000", "--proxy-headers"]
