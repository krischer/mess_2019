# Tutorials for the MESS 2019

## Installation

1. Make sure you have enough free disc space. We recommend at least 20 GB of free disc space to avoid any unforeseen problems.
2. Install *git* (https://git-scm.com/). There are various ways to install it. Make sure this works before your travels.
3. A conda installation with the required packages.

To install conda, navigate to this website: https://conda.io/en/latest/

We recommend the installation of the Miniconda distribution (https://conda.io/en/latest/miniconda.html) but you are free to choose the full Anaconda distribution.

Once you installed the version for your operating system please run the following command (in a terminal on OSX and Linux, `conda` install a special terminal on Windows - use that).

```bash
$ conda create -c conda-forge -n mess_2019 python=3.6 jupyter numpy scipy "matplotlib<2.2" cartopy obspy keras tensorflow scikit-learn seaborn pandas ipywidgets python-graphviz h5py statsmodels
```


#### Special Step on Windows

One more step is required on Windows:

Run: `conda env list` and define where your environment resides:

```bash
$ conda env list
mess_2019      C:\Anaconda3\envs\mess_2019
```

open that path in windows explorer and then navigate to

```
etc\conda\activate.d
```

there should be 3 files, delete them!


## Resetting and Updating

You can always reset and update the repository. **THIS WILL DELETE ALL OF YOUR CHANGES!!!**

```bash
$ git reset --hard HEAD
$ git pull
```