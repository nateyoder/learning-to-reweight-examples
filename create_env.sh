#!/usr/bin/env bash

conda create -y -n reweight python=2 numpy jupyter ipython h5py pillow pandas seaborn cython requests tqdm
source activate reweight

conda install pytorch torchvision>=0.2.1 -c pytorch