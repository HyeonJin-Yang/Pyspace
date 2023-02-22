#!/bin/sh
python P1RemoveOutliers.py
python P5Hypertuning.py
python P2LEBReg.py
python P3XGBReg.py
python P4Ensemble.py