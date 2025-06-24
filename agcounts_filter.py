""" This file contains the conversion of raw accelerometer data from a CSV file into activity counts.
It reads the CSV file, processes the accelerometer data, and returns a DataFrame with activity counts.
This code is extracted from github "https://github.com/actigraph/agcounts", and was published with the work of Neishabouri et al. (2022)."""

from agcounts.extract import get_counts
import pandas as pd
import numpy as np

def get_counts_csv(
    file,
    freq: int,
    epoch: int,
    fast: bool = True,
    verbose: bool = False,
    time_column: str = None,
):
    if verbose:
        print("Reading in CSV", flush=True) #ARGUMENT NAMES A ETE RAJOUTES POUR COLLER AUX FICHIER DATA_N_LW.csv ... IL FAUR L'ENLEVER SI LES NOMS DES COLONNES NE CORRESPONDENT PAS
    """dtype_dict = {
        "Timestamp": str,
        "Gyro X": str,
        "Gyro Y": str,
        "Gyro Z": str,
        "Accelerometer X": str,
        "Accelerometer Y": str,
        "Accelerometer Z": str,
        "Event": str,
        "Quat W":str,
        "Quat X":str, 
        "Quat Y":str,
        "Quat Z":str,
    }"""
    
    raw = pd.read_csv(file, skiprows=7,decimal=",", names = ["Timestamp","Gyro X","Gyro Y","Gyro Z","Accelerometer X","Accelerometer Y","Accelerometer Z","Event","Quat W","Quat X","Quat Y","Quat Z"]) #names=list(dtype_dict.keys()), dtype=dtype_dict)
    if time_column is not None:
        ts = raw[time_column]
        ts = pd.to_datetime(ts)
        time_freq = str(epoch) + "s"
        ts = ts.dt.round(time_freq)
        ts = ts.unique()
        ts = pd.DataFrame(ts, columns=[time_column])
    raw = raw[["Accelerometer X", "Accelerometer Y", "Accelerometer Z"]].astype(float)
    if verbose:
        print("Converting to array", flush=True)
    raw = np.array(raw)
    if verbose:
        print("Getting Counts", flush=True)
    counts = get_counts(raw, freq=freq, epoch=epoch, fast=fast)
    del raw
    counts = pd.DataFrame(counts, columns=["Axis1", "Axis2", "Axis3"])
    counts["AC"] = (
        counts["Axis1"] ** 2 + counts["Axis2"] ** 2 + counts["Axis3"] ** 2
    ) ** 0.5
    if time_column is not None:
        ts = ts[0 : counts.shape[0]]
        counts = pd.concat([ts, counts], axis=1)
    return counts


def convert_counts_csv(
    file,
    outfile,
    freq: int=100,
    epoch: int=60,
    verbose: bool = False,
    time_column: str = None,
):
    counts = get_counts_csv(
        file, freq=freq, epoch=epoch, verbose=verbose, time_column=time_column
    )
    counts.to_csv(outfile, index=False)
    return counts


def convert_AC(file_dom):# ///////////////////////// AJOUTER NON DOM_COUNT PLUS TARD
    dom_counts = get_counts_csv(file_dom, freq=128, epoch=1)
    dom_counts = convert_counts_csv(
        file_dom,
        outfile="Activity_counts_files/dom_counts.csv",
        freq=128,
        epoch=1,
        verbose=True,
        time_column= None,
    )


    return dom_counts # ///////////////////////// AJOUTER NON DOM_COUNT PLUS TARD