# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
# COPY requirements.txt requirements.txt
# COPY setup.py setup.py
# #COPY data/ data/
# COPY .dvcignore .dvcignore
# COPY data.dvc data.dvc
# COPY .dvc/ .dvc/
# COPY src/ src/
# COPY models/ models/
# COPY reports/ reports/
RUN git clone https://github.com/haridimos9/mlops_exercises.git
WORKDIR /mlops_exercises/
RUN pip install -r requirements.txt --no-cache-dir

RUN dvc pull
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]