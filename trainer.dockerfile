# Base image
FROM python:3.9-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
COPY data.dvc data.dvc
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
#RUN export GOOGLE_APPLICATION_CREDENTIALS=dtumlops-374716-e0c1d21736a7.json
#ENV GOOGLE_APPLICATION_CREDENTIALS dtumlops-374716-e0c1d21736a7.json

RUN dvc remote modify myremote gdrive_client_id '600764407405-qpr4rb7sdkbnatbtrii118g11o0kjtv5.apps.googleusercontent.com'
RUN dvc remote modify myremote gdrive_client_secret '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'
RUN dvc pull -v
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]