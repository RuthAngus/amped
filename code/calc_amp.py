import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

import kepler_data as kd

df = pd.read_csv("../data/gaia_mc_non_periodic_rvar_append.csv")

# df = pd.read_csv("../data/non_p_cuts.csv")
# Rvar = np.zeros(len(df))
# for i in trange(len(df.kepid.values[:1200])):
    # lcdir = "/Users/rangus/.kplr/data/lightcurves/{}".format(
    #     str(int(df.kepid.values[i])).zfill(9))
    # x, y, yerr = kd.load_and_join(lcdir)
    # Rvar[i] = np.percentile(y, 95) - np.percentile(y, 5) * 1e6
# df["Rvar"] = Rvar
# df.to_csv("../data/gaia_mc_non_periodic_rvar.csv")
# df = pd.read_csv("../data/gaia_mc_non_periodic_rvar.csv")

Rvar = df.Rvar.values
m = Rvar > 0
inds = np.arange(len(Rvar))
print(inds[m][-1])
print(df.kepid.values[inds[m][-1]])
print(df.kepid.values[2399])
print(df.kepid.values[2400])

for i in trange(2400, 2600):
    lcdir = "/Users/rangus/.kplr/data/lightcurves/{}".format(
        str(int(df.kepid.values[i])).zfill(9))
    x, y, yerr = kd.load_and_join(lcdir)
    Rvar[i] = np.percentile(y, 95) - np.percentile(y, 5) * 1e6

df["Rvar"] = Rvar
df.to_csv("../data/gaia_mc_non_periodic_rvar_append.csv")
