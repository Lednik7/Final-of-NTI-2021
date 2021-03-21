import numpy as np
import sys
import pandas as pd
import pickle
import ast

np.set_printoptions(threshold=sys.maxsize)
def got_columns(series, wind_size=5):
    result = {}
    for c in series.columns:
  
        for agg in ["median", "mean", "std", "var"]:
            agg_val = series[-wind_size:][c].fillna(0).aggregate(agg)
            if agg_val !=np.inf:
                result["('"+c+"', '"+agg+"')_"+str(wind_size)] = agg_val
            else:
                result["('"+c+"', '"+agg+"')_"+str(wind_size)] = -1
            
            result[c+"_"+str(wind_size)] = (series[c].iloc[-1] - series[c].iloc[-wind_size-1]).astype(np.float16)
    return result

class Predictor():
    def __init__(self):
        with open("model.pkl","rb") as f:
            self.lg = pickle.load(f)
            
    def forecast(self,series):
        global df_X
        #важные колонки для использования
        val_col = ['Coolness_RHEED',  'R FWHM_RHEED',  'X FWHM_RHEED', 'Y FWHM_RHEED',
            'Filtered Rate', 'Displayed Rate', 'Raw Rate',
            'Source Power',  "Length", "Crystal Position"]

        df_X = got_columns(series[val_col], wind_size=5)
        df_X.update(got_columns(series[val_col], wind_size=50))
        df_X.update(got_columns(series[val_col], wind_size=150))
        df_X.update(got_columns(series[val_col], wind_size=250))
        df_X.update(got_columns(series[val_col], wind_size=350))
        df_X.update(got_columns(series[val_col], wind_size=500))

        for k,v in df_X.items():
            if v is None:
                df_X[k] = 0

        self.df_X = df_X
        result = self.lg.predict_proba(pd.DataFrame([df_X]))
        return result[-1][1]
