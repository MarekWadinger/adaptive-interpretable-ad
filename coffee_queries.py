from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

def query_coffee_history():
    # You can generate an API token from the "API Tokens Tab" in the UI
    token = "Y2nr_V73MfQqmotuaeReBnRnuPDihdKm-RCSFRwfyFhlxx4b7YjUkzBokDrhV3q6G2ib-Arknf6e9itL9ROJDg=="
    org = "uiam"
    bucket = "mqtt"

    with InfluxDBClient(url="https://influxdb.cloud.uiam.sk", token=token, org=org) as client:
        query = 'from(bucket: "mqtt") |> range(start: -1y) |> filter(fn: (r) => r._measurement == "shellies" and r._field == "power" and r.device == "Shelly_Kitchen-C_CoffeMachine/relay/0")'
        tables = client.query_api().query(query, org=org)
        client.close()
    
    l = []
    for table in tables:
        for record in table.records:
            l.append({'power': record.get_value()})
        
    return l


def process_coffee_present(x):
    return {'power': float(x.payload.decode())}


def process_coffee(x):
    if isinstance(x, dict):
        return x
    else:
        return process_coffee_present(x)