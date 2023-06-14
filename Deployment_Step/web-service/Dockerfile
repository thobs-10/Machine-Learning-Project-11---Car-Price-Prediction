FROM python:3.10.8
# above we specify the python that will be in our docker image, the FROM is pulling the docker image that will be contained
# the line below updates the pip version just in case the model we are using requires an updated pipenv
RUN pip install -U pip
# this line installs the pipenv we used for the model development in our localhost to the docker image
RUN pip install pipenv 
# the working directory of the model
WORKDIR /app
# copy the files from the pipenv to the working directory
COPY [ "Pipfile", "Pipfile.lock", "./" ]
# install the pipfiles
RUN pipenv install --system --deploy
# copy the predict script and the model to the working directory
COPY [ "predict.py", "lin_reg.bin", "./" ]
# expose the port so it becomes free and listens to the post requests sent to it.
EXPOSE 9696
# the entry point for  mapping and biinding the localport with the container port and mapping the working directly with the predict
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
