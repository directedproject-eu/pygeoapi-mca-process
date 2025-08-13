# pygeoapi-mca-process
Multi-criteria analysis (MCA) process for pygeoapi implemented using CLIMADA

Build the Docker image:

```shell
docker build . -t directed/mca-roskilde:latest
```

Run the Docker image:

```shell
docker run directed/mca-roskilde:latest
```

Trigger async execution

```shell
curl -X 'POST' \
  '<host>/pygeoapi/processes/climada-mca-roskilde/execution' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Prefer: respond-async' \
  -d "{
  \"inputs\": {
    \"token\": \"***\",
    \"weights\": {
      \"measure net cost\": 0.25,
      \"averted risk_aai\": 0.5,
      \"approval\": 0.8,
      \"feasability\": 0.1,
      \"durability\": 0.3,
      \"externalities\": 0.4,
      \"implementation time\": 0.15
    }
  }
}"
```