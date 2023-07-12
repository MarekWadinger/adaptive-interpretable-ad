# RPC Outlier Detection with Dynamic Process Limits

[![Python application](https://github.com/tesla50hz/rpc_online_outlier_detection/actions/workflows/python-app.yml/badge.svg)](https://github.com/tesla50hz/rpc_online_outlier_detection/actions/workflows/python-app.yml)
[![Test Status](/reports/coverage-badge.svg)](https://htmlpreview.github.io/?https://github.com/tesla50hz/rpc_online_outlier_detection/blob/main/reports/coverage/report/index.html)
[![Test Status](/reports/test-badge.svg)](https://htmlpreview.github.io/?https://github.com/tesla50hz/rpc_online_outlier_detection/blob/main/reports/junit/report/index.html)
[![Flake8 Status](/reports/flake8-badge.svg)](https://htmlpreview.github.io/?https://github.com/tesla50hz/rpc_online_outlier_detection/blob/main/reports/flake8/report/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8128206.svg)](https://doi.org/10.5281/zenodo.8128206)

Online outlier detection service for existing real-time infrastructures for low-latency detection and change-point adaptation.
The service provides dynamic process limits based on changing environmental conditions and sensors aging.

The main benefits of the  solution are that it:

* Keeps existing IT infrastructure, saving costs, and does
not require operator retraining
* Automates alerting thresholds setup for a high number of
signals
* Automates alerting for signals with no a priori knowledge
of process limits
* Assesses changing environmental conditions and device
aging
* Uses self-learning approach on streamed data

![Average_Cell_Temperature_sliding_thresh](https://github.com/MarekWadinger/online_outlier_detection/assets/50716630/427586d8-9858-4cf2-9aaa-1ee5407416bd)

## RPC endpoint string

`rpc_online_outlier_detection`

Maintainer: insert your T50Hz email address here

## Usage

### Install/upgrade the `t50hz_rpc` module

```bash
pip install --upgrade git+https://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@github.com/tesla50hz/t50hz_rpc.git
```

### Instantiate the client

```python
from t50hz_rpc import rpc_client

client = rpc_client("rpc_online_outlier_detection")
```

## 🛠 Installation (Piku)

Make sure you have the Piku server set up in your `~/.ssh/config` file:

```config
Host piku.aws
     User piku
     HostName piku.cloud.tesla50hz.eu
     Port 3122
```

Next, add the Piku remote

```bash
git remote add piku piku.aws:rpc_online_outlier_detection # Note: replace "rpc_online_outlier_detection" with the actual RPC endpoing string
```

Finally, to deploy the app to Piku, run:

```bash
git push piku main
```

(Replace `main` by `master` if that is how your main branch is called.)

You can then check the status of your service at Piku by

```bash
ssh piku.aws apps
```

And you can inspect logs of your service by

```bash
ssh piku.aws logs rpc_online_outlier_detection # Note: replace "rpc_online_outlier_detection" with the actual RPC endpoing string
```

## 🛠 Installation (local development)

```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

Then start the server in one terminal, and the client in another:

```bash
python3 rpc_server.py
python3 rpc_client.py
```

## ⚡️ Quickstart

Get your hand on the algorithm using this [notebook](https://github.com/tesla50hz/rpc_online_outlier_detection/blob/main/examples/01_univariate_pc_2023.ipynb) and play around with example data.

## 🏃 Run the services

### Stream MQTT messages

To start the service, run following line of code in your terminal:

```bash
python client.py -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/power"
```

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(NOTE: first valid key value pair is used)
* `[-t | --topic]` to define topic to subscribe to or column in csv file
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

To start consumer, run following command:

```bash
python consumer.py -t "shellies/Shelly3EM-Main-Switchboard-C/emeter/0/dynamic_limits"
```

Note: You can modify the source data stream using attributes:

* `[-f | --config-file]` with path to `config.ini`
(NOTE: first valid key value pair is used)
* `[-t | --topic]` topic of MQTT or column of pd.DataFrame
* `[-k | --key-path]` with path to ssh keys of sender and receiver
(NOTE: if empty, the keys are created)

Query service responds with printed messages as follows:

```bash
Received message: {"time": "1970-01-01 03:17:11", "anomaly": "0", "level_high": "658.396223558289", "level_low": "635.8731097750442"}
```

### Stream file messages

If you want to stream example dataset use

```bash
python client.py -t "Average Cell Temperature"
```

where your `config.ini` shall contain

```ini
[file]
path=examples/data/input/average_temperature.csv
output=examples/data/output/dynamic_limits.json
```

Now, let's query the latest limits from data/output/dynamic_limits.json

```bash
python consumer.py -t "Average Cell Temperature"
```

The response is the latest date in `dynamic_limits.json`

```python
{'time': datetime.datetime(1970, 1, 1, 14, 52, 42), 
 'anomaly': 0, 
 'level_high': 1180.92, 
 'level_low': 1151.15}
```

Note: You can modify the attributes to retrieve thrasholds at any date:

* `[-d | --date]` date as 'Y-m-d H:M:S'

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and approaches.

* Feel welcome to [open an issue](https://github.com/tesla50hz/rpc_online_outlier_detection/issues/new/choose) if you think you've spotted a bug or a performance issue.

<!-- 
## 🤝 Affiliations

<p align="center">
  <img width="70%" src="" alt="affiliations">
</p>
-->

## 💬 Citation

If the service or the algorithm has been useful to you and you would like to cite it in an scientific publication, please refer to the [paper](https://doi.org/10.5281/zenodo.8128206) in Proceedings of the 2023 24th International Conference on Process Control, IEEE:

```bibtex
@inproceedings{wadinger_pc_2023,
  title={Real-Time Outlier Detection with Dynamic Process Limits},
  DOI={10.5281/zenodo.8128206},
  publisher={Zenodo},
  author={Wadinger, Marek and Kvasnica, Michal},
  year={2023},
  month={Jun}
}
```

<!-- 
## 📝 License

This algorithm is free and open-source software licensed under the [3-clause BSD license](https://github.com/online-ml/river/blob/main/LICENSE).
  -->
