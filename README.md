# *pyPSG* toolbox documentation


## Introduction

Dedicated toolboxes have been developed for the analysis of individual physiological signals.
However, for the purpose of fast and efficient data analysis, there was a need to unify these separate modules.
To address this gap, pyPSG was developed — a Python-based toolbox capable of handling and analyzing various types of physiological signals within a single environment.
The toolbox standardizes the signal processing workflow across data from different sources and also enables the preprocessing of physiological signals for machine learning applications.
During development, modularity was a key consideration to allow future expansion with additional biological signals.



## Description
Input data, raw signals are first unified in the EDF (European Data Format) structure. 
The pipeline processes three parallel signal branches: PPG, ECG, and SpO$_2$. 
Each signal undergoes preprocessing using its corresponding module (pyPPG, pecg, or pobm). 
In the case of PPG and ECG, fiducial points are detected to derive beat-to-beat intervals, which are then forwarded to the mhrv module for HRV/BRV analysis. 
Biomarkers are extracted separately from all three signal types, and the results are aggregated and saved for further data analysis.

![](docs/figs/pipeline.png)

## Installation

## Requirements

## Documentation