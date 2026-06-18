#base image
FROM python:3.14

#working dir
WORKDIR /code

#copy
COPY ./requirements.txt /code/requirements.txt

#run
RUN pip install -r requirements.txt

#copy
./app /code/app

#port 
EXPOSE 8000

#command
CMD ["fastapi", "run", "./main.py", "--port", "80"]