# IMPORTS
import argparse
import time
import json
import datetime as dt
import pandas as pd
from scipy.stats import norm
from river import anomaly
from streamz import Stream


# CONSTANTS
THRESHOLD = 0.99735
GRACE_PERIOD=60*24
WINDOW = dt.timedelta(hours=24*7)

# DEFINITIONS
class GaussianScorer(anomaly.GaussianScorer):
    def __init__(self, 
                 threshold=THRESHOLD, 
                 window_size=None, 
                 period=WINDOW, 
                 grace_period=GRACE_PERIOD):
        super().__init__(window_size, period, grace_period)
        self.threshold = threshold
        
    def learn_one(self, x, **kwargs):
        self.gaussian.update(x, **kwargs)
        return self
    
    def score_one(self, x, t=None):
        if self.gaussian.n_samples < self.grace_period:
            return 0
        return 2 * abs(self.gaussian.cdf(x) - 0.5)
    
    def predict_one(self, x, t=None):
        score = self.score_one(x)
        if self.gaussian.obj.n_samples > self.grace_period:
            return 1 if score > self.threshold else 0
        else:
            return 0
        
    def limit_one(self):
        kwargs = {'loc': self.gaussian.mu, 
                'scale': self.gaussian.sigma}
        real_thresh = norm.ppf((self.threshold/2 + 0.5), **kwargs)
        return real_thresh if real_thresh < 1 else 1

                
    def process_one(self, x, t=None):
        if self.gaussian.n_samples == 0:
                self.gaussian._var.mean._mean = x
        t = t.tz_localize(None)
        
        is_anomaly = self.predict_one(x)
        
        real_thresh = self.limit_one()
        
        if not is_anomaly:
            self = self.learn_one(x, **{'t': t})
        
        return is_anomaly, real_thresh

# FUNCTIONS
def preprocess(
        x,
        col):
    col = [col] if not isinstance(col, list) else col
    if isinstance(x, (pd.Series)):    
        return {'time': x.name,
                'data': x[col].to_dict()
        }
    elif isinstance(x, tuple) and isinstance(x[1], (pd.Series)):
        return {'time': x[0],
                'data': x[1][col].to_dict()
        }
    elif isinstance(x, dict):
        return  {k: v for k, v in x.items() if k in col} 
    elif isinstance(x, paho.mqtt.client.MQTTMessage):
        return {'time': x.timestamp,
                'data': {x.topic.split("/")[-1]: float(x.payload)}
        }


def fit_transform(
        x, 
        model, 
        model_inv
        ):
    # TODO: replace x_ for multidimensional implementation
    x_ = next(iter(x['data'].values()))
    is_anomaly, real_thresh = model.process_one(x_, x['time'])
    is_anomaly_, real_thresh_ = model_inv.process_one(-x_, x['time'])
    return {'time': str(x['time']),
            'anomaly':is_anomaly,
            'level_high':real_thresh, 
            'level_low':real_thresh_
            }


def dump_to_file(x, f):
    print(json.dumps(x), file=f)


def process_limits_streaming(
        col: str,
        df: pd.DataFrame):
    model = GaussianScorer()
    model_inv = GaussianScorer()
    
    source = Stream.from_iterable(df.iterrows())
    detector = source.map(preprocess, col).map(fit_transform, model, model_inv)
    
    with open('data.json', 'a') as f:
        detector.sink(dump_to_file, f)
        source.start()
        time.sleep(20)
        source.stop()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="File to read. If none MQTT is used.", default='average_temperature.csv')
    parser.add_argument("-signal", help="Topic of MQTT or Column of pd.DataFrame", default='Average Cell Temperature')
    args = parser.parse_args()
    
    if args.f:
        df = pd.read_csv(args.f, index_col=0)
    df.index = pd.to_datetime(df.index)
    col = args.signal
    process_limits_streaming(col, df)

    # Print summary
    d = pd.read_json('data.json', lines=True)
    text = (
        f"Proportion of anomalous samples: "
        f"{sum(d['anomaly'])/len(d['anomaly'])*100:.02f}%\n"
        f"Total number of anomalous events: "
        f"{sum(pd.Series(d['anomaly']).diff().dropna() == 1)}")
    print(text) 
