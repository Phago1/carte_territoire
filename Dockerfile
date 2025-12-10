# ####### 👇 SIMPLE SOLUTION (x86 and M1) 👇 ########
FROM python:3.12.9-bullseye

COPY requirements.txt requirements.txt
COPY carte_territoire_package carte_territoire_package

RUN pip install -r requirements.txt

CMD uvicorn carte_territoire_package.api.fast_selector:app --host 0.0.0.0 --port $PORT
