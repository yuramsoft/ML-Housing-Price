FROM python:3.9.12
ENV MLFLASK 1
RUN mkdir mlmflask-env
WORKDIR /mlmflask-env
COPY requirements.txt /mlmflask-env/
RUN pip install -r requirements.txt
COPY . /mlmflask-env 
EXPOSE 5000
CMD ["python3", "app.py"]