# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 11:21:29 2023

@author: eliam
"""

import deepsensor.torch
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.train import Trainer

import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load raw data
ds_raw = xr.tutorial.open_dataset("air_temperature")

# Normalise data
data_processor = DataProcessor(x1_name="lat", x2_name="lon")
ds = data_processor(ds_raw)

# Set up task loader
task_loader = TaskLoader(context=ds, target=ds)

# Set up model
model = ConvNP(data_processor, task_loader)

# Generate training tasks with up 100 grid cells as context and all grid cells
#   as targets
train_tasks = []
for date in pd.date_range("2013-01-01", "2014-11-30")[::7]:
    N_context = np.random.randint(0, 100)
    task = task_loader(date, context_sampling=N_context, target_sampling="all")
    train_tasks.append(task)

# Train model
trainer = Trainer(model, lr=5e-5)
for epoch in tqdm(range(10)):
    batch_losses = trainer(train_tasks)

# Predict on new task with 50 context points and a dense grid of target points
test_task = task_loader("2014-12-31", context_sampling=50)
pred = model.predict(test_task, X_t=ds_raw)

