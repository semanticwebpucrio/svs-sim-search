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

WORKDIR ${PYTHONPATH}

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE ${PORT}

ENTRYPOINT ["sh", "/opt/entrypoint.sh"]
