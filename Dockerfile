#base image
FROM python:3.12-slim

#working dir
WORKDIR /code

#copy
COPY ./requirements.txt /code/requirements.txt

#run
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

#copy
COPY ./app /code/app
COPY ./templates /code/templates
COPY ./process.py/ code/process.py

#expose
EXPOSE 80

#command
CMD ["fastapi", "run", "app/main.py", "--port", "80"]