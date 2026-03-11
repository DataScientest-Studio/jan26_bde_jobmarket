FROM python:3.11-slim

RUN pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

WORKDIR /home/jovyan/work

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]