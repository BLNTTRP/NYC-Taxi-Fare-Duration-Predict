FROM python:3.11-slim as base

ENV PYTHONPATH=$PYTHONPATH:/api/

COPY ./requirements.txt /api/requirements.txt

WORKDIR /api

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./ /api/

FROM base as test
RUN ["python", "-m", "pytest", "-v", "/api/tests"]

FROM base as build

WORKDIR ..

CMD uvicorn api.main:app --host 0.0.0.0 --port 5000 --workers 4