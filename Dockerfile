#base image
FROM python:3.12-slim

#working dir
WORKDIR /code

#install tesseract ocr engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

#copy
COPY ./requirements.txt /code/requirements.txt

#run
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#RUN
RUN python -m spacy download en_core_web_sm

#copy
COPY ./app /code/app
COPY ./templates /code/templates

#expose
EXPOSE 80

#command
CMD ["fastapi", "run", "app/main.py", "--port", "80", "--host", "0.0.0.0"]