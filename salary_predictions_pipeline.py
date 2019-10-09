#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Prashant Bansod

@email: prashant.bansod@outlook.com

"""

# importing libraries
import pandas as pd
import numpy as np


# Scikit Learn pre-processing libraries needed
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# sklearn ML libraries
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# load the data into a Pandas dataframe
test_features = pd.read_csv('data/test_features.csv')
train_features = pd.read_csv('data/train_features.csv')
train_salaries = pd.read_csv('data/train_salaries.csv')

training_data = train_features.merge(train_salaries, on="jobId")

# Function to remove records which have less than x% of
# rows in a certain salary range


def remove_extremes(training_data, tune_param):
    '''Function to remove records which have less than x%
    of rows in a certain salary range'''

    rows_removed = 0
    Job_Types = training_data.jobType.unique()

    for j in Job_Types:
        x = training_data[training_data.jobType == j]['salary'].count()
        for i in range(0, 100, 10):
            # identify records in the certain bin for jobType j
            y = training_data[(training_data.jobType == j) &
                (training_data.salary >= np.percentile(training_data.salary,
                                                       i)) &
                (training_data.salary < np.percentile(training_data.salary,
                                                      (i+10)))]['salary'].count()

            if((y/x)*100 < tune_param):
                    # keep count of the number of rows removed
                    rows_removed = rows_removed + y

                    # eliminate the records
                    training_data = training_data.loc[~((training_data.jobType == j)&
                    (training_data.salary >= np.percentile(training_data.salary, i))&
                    (training_data.salary < np.percentile(training_data.salary, (i+10)))), :]
    print(rows_removed)
    return(training_data)


# remove extremes
training_data = remove_extremes(training_data, 1)
train_Salaries = training_data['salary']
training_data.drop('salary', axis=1, inplace=True)
train_Features = training_data

# creating a validation set to help tune to boosting model
train_features,  valid_features, train_salaries, valid_salaries = train_test_split(train_features,
                                                                                  train_salaries,
                                                                                 test_size=0.3)

# Select categorical columns and check their cardinality
categorical_cols = [cols for cols in train_features.columns
                   if train_Features[cols].dtype == 'object']

# only keep columns which have a cardinality < 10
categorical_cols = [col for col in categorical_cols
                   if train_Features[col].nunique() <= 10]

# numerical cols
numeric_cols = [col for col in train_Features.columns
               if train_Features[col].dtype in ["int64", "float64"]]

# removing high cardinal features from training and validation sets
all_cols = numeric_cols + categorical_cols
train_features = train_features[all_cols]
valid_features = valid_features[all_cols]


# There is no missing data, however, we define imputers
# to make the code production ready

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer_jobtype = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='SENIOR')),
    ('encdoer', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


categorical_transformer_industry = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='WEB')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

categorical_transformer_others = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='NONE')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
])


# Bundle the pre-processing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat1', categorical_transformer_jobtype, ['jobType']),
        ('cat2', categorical_transformer_industry, ['industry']),
        ('cat3', categorical_transformer_others, ['degree', 'major'])
    ])

# Model pipeline

model_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', xgb.XGBRegressor(n_jobs=-1, n_estimators=200))
])

param = {
    'model__max_depth': [i for i in range(3, 10, 2)],
    'model__learning_rate': [3e-1, 2e-1, 1e-1],
    'model__min_child_weight': [j for j in range(5, 25, 5)],
    'model__colsample_bytree': [7e-1, 8e-1, 9e-1]
}

fit_params = {
    "model__early_stopping_rounds": 5,
    "model__eval_metric": 'rmse',
    "model__eval_set": [(preprocessor.fit_transform(valid_features),
                         valid_salaries)]
}


grid_model = RandomizedSearchCV(model_pipeline,
                          param_distributions=param, cv=3, verbose=False,
                          fit_params=fit_params).fit(train_features,
                                               train_salaries)
