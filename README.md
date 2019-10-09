## File guide:
Salary predictions- EDA and model build.ipynb : jupyter notebook containing all EDA, preprocessing and model building.

Salary_Predictions_pipeline.py - Pipeline that bundles all of pre-processing, model building and tuning.

Salary_predictions - Exploratory analysis - Notebook with only the data exploration and viz. 

# Problem definition: 

Examine a set of job details along with relevant credentials and make salary predictions. The goal is to predict the salary of a job postings based on the given information.

# 1. Exploratory analysis

Summarize the data in a meaningful way.

Identify patterns and outliers.

Examine the distribution of the variable and relationship between features.

Exploratory analysis of features and their influence on salaries.

# 2. Clean the data and generate new features

Remove rows where salary=0

Remove rows which have less than 2% presence/ jobtype in terms of final salary 

Multiple features including binning and encoding company data, creating a feature for if the company paid above industry avg. salary and binning experience level were tried upon but since none of them provided any significant uplift to the model results, they were all discarded.

# 3. Build a baseline model and compare the lift against a simple average model

A linear regression model was built and it's efficacy was measured against a simple average salary prediction method. 
The linear regression model produced a 82.67 % improvement when the MSE's were compared.

# 4. Choose algorithms

Three models were created and their performance was compared against one another;

Lasso Score: 395.48

RandomForest score (after tuning): 383.61

XGBoost score: 356.649

# 5. Tune the models
As XGBoost showed better results, random search was implemented to tune the hyperparameters of the model.

# 6. Evaluate the results
RMSE was used to evaluate results. RMSE basically measures square root of average of squared error of our predictions. After hyperparameter tuning the results improved MSE = 349.38.

# 7. Build model Pipeline to bundle all processes
A data pipeline was created using scikit-learn PipeLine to bundle all pre-processing, model building and tuning into a pipeline which would help prevent leakage and make the code more scalable and efficient.
