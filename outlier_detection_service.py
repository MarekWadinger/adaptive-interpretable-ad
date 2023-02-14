# IMPORTS
import argparse
import json
import datetime as dt
import pandas as pd
from scipy.stats import norm
from river import anomaly


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
def process_limits_streaming(
        col: str,
        df: pd.DataFrame):
    model = GaussianScorer()
    model_inv = GaussianScorer()

    anomaly_samples = []
    anomaly_samples_ = []
    list_thresh_pos = []
    list_thresh_neg = []
    
    df = df[col]
    with open('data.json', 'a') as f:
        for t, x in df.items():
            is_anomaly, real_thresh = model.process_one(x, t)
            is_anomaly_, real_thresh_ = model_inv.process_one(-x, t)
            
            dict_ = {str(t): [is_anomaly, real_thresh, real_thresh_]}
            print(json.dumps(dict_), file=f)
            
            anomaly_samples.append(is_anomaly)
            anomaly_samples_.append(is_anomaly_)
            list_thresh_pos.append(real_thresh)
            list_thresh_neg.append(real_thresh_)
            
    text = (f"Sliding window: {model.period}\n"
        f"Proportion of anomalous samples: "
        f"{sum(anomaly_samples)/len(anomaly_samples)*100:.02f}%\n"
        f"Total number of anomalous events: "
        f"{sum(pd.Series(anomaly_samples).diff().dropna() == 1)}")
    print(text) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-signal", help="Topic of MQTT or Column of pd.DataFrame", default='Average Cell Temperature')
    args = parser.parse_args()
    
    df = pd.read_csv('average_temperature.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    col = args.signal
    process_limits_streaming(col, df)
