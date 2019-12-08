import numpy as np
import pandas as pd
import kinematics_and_rotation as kr
import kplr
client = kplr.API()


def download_periodic():
    df = pd.read_csv("../data/gaia_mc_cuts.csv")

    for i, k in enumerate(df.kepid.values[1770+278+140:]):
        print(k, i, "of", len(df.kepid.values[1770+278+140:]))
        star = client.star(k)
        star.get_light_curves(fetch=True, short_cadence=False)


def download_non_periodic():
    gaia_mc1 = pd.read_csv("gaia_mc_non_periodic.csv")
    print(np.shape(gaia_mc1))

    # Cut out stars with large vb uncertainties.
    m = gaia_mc1.vb_err.values < 1.
    print(np.shape(gaia_mc1.iloc[m]), "no large vb uncertainties")

    m &= gaia_mc1.phot_g_mean_mag.values < 16.
    print(np.shape(gaia_mc1.iloc[m]), "no faint stars")

    # Cut out rapid rotators (synchronized binaries)
    # m &= gaia_mc1.age.values > .5
    # print(np.shape(gaia_mc1.iloc[m]), "no rapid rotators")

    # Cut out very hot and very cold stars. The hot limit is usually 5000 and the cool usually 3500
    mint, maxt = 3500, 5500
    m &= (gaia_mc1.color_teffs.values < maxt) * (mint < gaia_mc1.color_teffs.values)
    print(np.shape(gaia_mc1.iloc[m]), "no hot or cold stars")

    # Try cutting out stars with latitudes greater than bmax degrees
    bmax = 15
    bmin = 10
    m &= (gaia_mc1.b.values < bmax) * (bmin < gaia_mc1.b.values)
    print(np.shape(gaia_mc1.iloc[m]), "no high latitude stars")

    gaia_mc = gaia_mc1.iloc[m]

    # # Restrict to stars with Vz
    # m &= np.isfinite(gaia_mc1.vz.values)
    # print(np.shape(gaia_mc1.iloc[m]), "only stars with Vz")

    # Remove velocity outliers
    v_clipped, clipping_mask = kr.sigma_clip(gaia_mc.vb.values, 3)
    df = gaia_mc.iloc[clipping_mask]

    df.to_csv("../data/non_p_cuts.csv")

    for i, k in enumerate(df.kepid.values[2400:2600]):
        print(str(int(k)), i, "of", len(df.kepid.values[2400:2600]))
        star = client.star(str(int(k)))
        star.get_light_curves(fetch=True, short_cadence=False)

if __name__ == "__main__":
    # download_periodic()
    download_non_periodic()
