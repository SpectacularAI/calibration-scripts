FROM python:3
COPY "requirements.txt" /scripts/
COPY "jsonl-to-kalibr.py" /scripts/
COPY "kalibr-to-parameters.py" /scripts/
COPY "kalibr-to-calibration.py" /scripts/
RUN pip3 install -r /scripts/requirements.txt
RUN apt update
RUN apt install ffmpeg -y
