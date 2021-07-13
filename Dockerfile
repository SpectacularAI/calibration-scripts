# Kalibr docker image has ancient 3.5 python which numpy doesn't support
# anymore so use depthai-library as base, its required by OAK-D anyhow
# so will already exist locally when running OAK-D calibration.
FROM luxonis/depthai-library:v2.6.0.0
COPY "requirements.txt" /scripts/
COPY "jsonl-to-kalibr.py" /scripts/
COPY "kalibr-to-parameters.py" /scripts/
COPY "kalibr-to-calibration.py" /scripts/
RUN pip3 install -r /scripts/requirements.txt
