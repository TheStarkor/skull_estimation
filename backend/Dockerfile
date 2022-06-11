FROM pytorch/pytorch
WORKDIR '/app'

RUN pip install flask flask_cors

COPY . .

CMD ["python", "app.py"]