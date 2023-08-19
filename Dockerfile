FROM python:3.9.15-slim-bullseye

ARG LC_ALL=C.UTF-8
ARG LANG=C.UTF-8
ARG PYTHONPATH=/opt
ARG PORT=8080
ARG HOST=0.0.0.0

ENV LC_ALL=${LC_ALL}
ENV LANG=${LANG}
ENV PYTHONPATH=${PYTHONPATH}
ENV PORT=${PORT}
ENV HOST=${HOST}

ADD requirements.txt ${PYTHONPATH}/

RUN pip3 install --no-cache-dir --upgrade pip \
 && pip install -r ${PYTHONPATH}/requirements.txt

ADD entrypoint.sh ${PYTHONPATH}/
ADD app/dl_models ${PYTHONPATH}/app/dl_models
ADD app/input ${PYTHONPATH}/app/input
ADD app/output ${PYTHONPATH}/app/output
ADD app/routers ${PYTHONPATH}/app/routers
ADD app/services ${PYTHONPATH}/app/services
ADD app/__init__.py ${PYTHONPATH}/app/__init__.py
ADD app/helper.py ${PYTHONPATH}/app/helper.py
ADD app/main.py ${PYTHONPATH}/app/main.py
ADD app/shared_context.py ${PYTHONPATH}/app/shared_context.py

WORKDIR ${PYTHONPATH}

RUN chmod +x entrypoint.sh

EXPOSE ${PORT}

ENTRYPOINT ["sh", "/opt/entrypoint.sh"]
