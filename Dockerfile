FROM python:3.11
# CLIMADA has a Python version restriction
# See https://github.com/CLIMADA-project/climada_python/blob/main/setup.py#L62)

#
# Fight OS CVEs and install dependencies
#
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get dist-upgrade -y \
 && apt-get install --assume-yes --no-install-recommends  \
    gdal-bin \
    libgdal-dev \
 && apt-get clean \
 && apt autoremove -y  \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY mcdm ./mcdm

ENTRYPOINT [ "python", "/app/mcdm/mca_roskilde.py"]
