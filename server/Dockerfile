FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip \
  && apt-get update \
  && apt-get -y install libgl1-mesa-glx \
  && apt-get -y install libglib2.0-0 \
  && pip install --no-cache-dir -r requirements.txt
COPY . .
COPY ./data ./data
COPY ./dtos ./dtos
COPY ./model ./model
COPY ./service ./service
COPY ./utils ./utils

# Run with uvicorn
CMD ["uvicorn", "main:app", "--reload", "--host=0.0.0.0", "--port=8000"]