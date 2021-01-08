FROM gcr.io/deeplearning-platform-release/xgboost-cpu.1-1

RUN apt-get update &&\
        pip install jupyterlab-nvdashboard &&\
        jupyter labextension install jupyterlab-nvdashboard

EXPOSE 8080

ADD run.sh 
ENTRYPOINT ["bash", "run.sh"]

