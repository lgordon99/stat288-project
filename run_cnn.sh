#!/bin/bash -x

set -x
date
source ~/.bashrc
python cnn.py ${1} ${2}