import pickle as pkl
import numpy as np
import pandas as pd
from pycaret.regression import *

dataset = pd.read_csv('Cleaned_Data.csv')
s = setup(data=dataset, target='medv', normalize=True)

'''
Comparing best regression models for dataset and storing top 5 on R2 score
'''
print('\n Model Comaprison')
best = compare_models(sort='R2', n_select = 5)

'''
Hyperparameter tuning over 10 runs on best model
'''
print('\n Hyperparameter tuning results across 10 runs')
tuned = tune_model(best[0])

'''
Plotting regression residuals of tuned model
'''
plot_model(tuned, plot = 'residuals')

'''
Sample Predictions
'''
print('\n Test Data Prediction Metrics')
holdout_pred = predict_model(tuned)

print('\n Sample Predictions')
print(holdout_pred.head())

save_model(tuned, 'tuned_pipeline')
