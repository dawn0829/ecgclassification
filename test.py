import biosppy
import numpy as np
import pyhrv.tools as tools
import pandas as pd

# Load sample ECG signal & extract R-peaks using BioSppy
df = pd.read_csv("./2022-05-27 03:47:01.523784.csv")
ecg = df.values.reshape(len(df))
signal, rpeaks = biosppy.signals.ecg.ecg(ecg, show=False)[1:3]

# Compute NNI
nni = tools.nn_intervals(rpeaks)
print(nni)