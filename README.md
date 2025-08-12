# pygeoapi-mca-process
Multi-criteria analysis (MCA) process for pygeoapi implemented using CLIMADA

Build the Docker image:

```shell
docker build . -t directed/mca-roskilde:latest
```

Run the Docker image:

```shell
docker run -e CONFIG="/app/test_data.json" -e MODE="ranks" directed/mca-roskilde:latest
```
