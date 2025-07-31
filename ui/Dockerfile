FROM python:3.13.3 as base

ENV PYTHONPATH=$PYTHONPATH:/src/ui

COPY ./requirements.txt /src/ui/requirements.txt

WORKDIR /src/ui

COPY ./ /src/ui

RUN pip install --upgrade pip && pip install -r requirements.txt


EXPOSE 8501

FROM base as build

CMD ["streamlit", "run", "--server.enableCORS", "false","--server.enableXsrfProtection", "false", "--server.port", "8501", "./app/app.py"]