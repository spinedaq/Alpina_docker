FROM python:3.8.0

RUN mkdir -p /opt/AlpinaApp
WORKDIR /opt/AlpinaApp
COPY ./requirements.txt /opt/AlpinaApp
RUN pip install -r /opt/AlpinaApp/requirements.txt
COPY . /opt/AlpinaApp
EXPOSE 3000

CMD ["python","main.py"]