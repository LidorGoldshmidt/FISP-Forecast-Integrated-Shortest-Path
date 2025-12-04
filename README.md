# FISP: Forecast-Integrated Shortest-Path Framework
MATLAB Implementation for Load Forecasting, Storage Operation, and Penalty Evaluation

This repository contains the full MATLAB implementation used in the paper:
“Measuring Peak Shaving Efficiency of Energy Storage Device Under Load Uncertainty With Machine Learning Based Forecasting Techniques”

The project integrates time-series forecasting models (LSTM, ARMA) with the Shortest-Path (SP) peak-shaving algorithm, and evaluates how forecast error affects operational performance using a weighted penalty metric.

## Repository Structure
FISP-Forecast-Integrated-Shortest-Path/
│
├── code/
│   ├── UpdatedCode.m
│   ├── armaUpdated.m
│   ├── FISP_LSTM_experiment.m
│   ├── Extract_Experiment_Results.m
│   ├── FISP_epochProgress.m
│
└── raw_data/
    ├── Germany_Data/
    │      └── *.csv    (2015–2024 load data)
    └── Italy_Data/
           └── *.csv    (2015–2024 load data)

## Folder Descriptions
📁 code/

This folder contains all MATLAB scripts required to construct datasets, train forecasting models, run the FISP optimization, compute penalties, and generate figures.

UpdatedCode.m

The main script of the project.
It includes:

Data loading and preprocessing (from the raw CSV files)

Construction of seasonal training/validation/testing datasets

Implementation of the Shortest-Path algorithm

Definition of the weighted penalty metric

Running the FISP pipeline on Actual, LSTM, and ARMA forecasts

Plotting functions and result visualization

Invocation of the Experiment Manager for LSTM hyperparameter tuning

This file orchestrates the full pipeline end-to-end.

armaUpdated.m

Trains ARMA models for each season.
Includes:

Grid search over multiple ARMA(p, q) configurations

Parameter estimation for each candidate model

Selection of the best order based on validation MAE

Final retraining and generation of ARMA day-ahead forecasts

FISP_LSTM_experiment.m

Defines the LSTM architecture and its hyperparameter search space.
Used with MATLAB’s Experiment Manager to run automated training trials.

Extract_Experiment_Results.m

Processes the output of the LSTM Experiment Manager:

Sorts trials

Identifies the best model

Extracts hyperparameters

Saves them for the final training stage

FISP_epochProgress.m

A callback function used during training to monitor progress of each epoch and ensure that training remains active (especially useful for long LSTM runs).

📁 raw_data/

Contains the raw load-data used in the experiments.

Germany_Data/

CSV files for years 2015–2024, filtered to summer weekdays.

Italy_Data/

CSV files for years 2015–2024, filtered to winter weekdays.

Each CSV includes:

Time stamps

Actual load (MW)

Metadata columns, depending on the country dataset source

These files are loaded automatically by UpdatedCode.m.

Running the Project

To reproduce results:

Clone the repository

Open MATLAB

Add the code folder to the MATLAB path

Ensure that Deep Learning Toolbox, Econometrics Toolbox, and Signal Processing Toolbox are available

Run:

UpdatedCode


## Optional steps:

Run FISP_LSTM_experiment.m in Experiment Manager to perform hyperparameter search

Run armaUpdated.m to re-train ARMA models

Modify storage capacities, α-weights, or seasonal datasets as needed

## Main Features

✔ Integrated forecasting + optimization pipeline
✔ LSTM forecasting with hyperparameter search
✔ ARMA forecasting with grid-search order selection
✔ Shortest-Path (SP) algorithm for optimal generation scheduling
✔ New weighted penalty metric for operational cost evaluation
✔ Support for multiple seasons and countries
✔ Reproducible figures for the paper

## Citation

If you use this repository, please cite the accompanying paper (details to be added upon publication).
