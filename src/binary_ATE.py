import numpy as np
import pandas as pd
import sys
table = np.array([])

df = pd.read_csv('./data/N3C_data_10000_sample.csv')
cols = df.columns.tolist()
drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
            'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
            'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']

tempcols = ['drugs']
for col in cols:
    if col in drugs or col == "Unnamed: 0" or col == 'outcome':
        continue
    tempcols.append(col+'_0')
    tempcols.append(col+'_1')
tempcols.append('total')
tempcols.append('ATE')

tempdf = pd.DataFrame(columns=tempcols)
print(tempdf.columns.tolist())

for drug in drugs:
    row = np.array([])
    for col in cols:
        cell = []
        if col in drugs or col == "Unnamed: 0" or col == 'outcome':
            continue
        # Count of patients who took the drug (drug == 1), have condition 0, and survived (outcome == 1)
        count_condition0 = df[(df[drug] == 1) & (df[col] == 0) & (df['outcome'] == 1)].shape[0]

        # Count of patients who took the drug (drug == 1), have condition 1, and survived (outcome == 1)
        count_condition1 = df[(df[drug] == 1) & (df[col] == 1) & (df['outcome'] == 1)].shape[0]
        #print(col)
        #print(count_condition0, count_condition1)
        
        















from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split




'''
def calculate_ate(data, treatment_col, outcome_col, covariates):
    """
    Calculate the Average Treatment Effect (ATE) for binary treatments.

    Parameters:
        data (pd.DataFrame): The dataset containing treatment, outcome, and covariates.
        treatment_col (str): The name of the treatment column.
        outcome_col (str): The name of the outcome column.
        covariates (list): List of covariate column names.

    Returns:
        float: The estimated ATE.
    """
    # Split data into treatment and control groups
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    # Estimate propensity scores using logistic regression
    X = data[covariates]
    y = data[treatment_col]
    model = LogisticRegression()
    model.fit(X, y)
    data['propensity_score'] = model.predict_proba(X)[:, 1]

    # Calculate weights
    data['weight'] = np.where(data[treatment_col] == 1,
                              1 / data['propensity_score'],
                              1 / (1 - data['propensity_score']))

    # Weighted outcome means
    treated_mean = np.average(treated[outcome_col], weights=treated['weight'])
    control_mean = np.average(control[outcome_col], weights=control['weight'])

    # Calculate ATE
    ate = treated_mean - control_mean
    return ate

# Example usage
if __name__ == "__main__":
    # Example dataset
    data = pd.DataFrame({
        'treatment': [1, 0, 1, 0, 1, 0],
        'outcome': [5, 3, 6, 2, 7, 1],
        'age': [25, 30, 35, 40, 45, 50],
        'income': [50000, 60000, 55000, 62000, 58000, 61000]
    })

    treatment_col = 'treatment'
    outcome_col = 'outcome'
    covariates = ['age', 'income']

    ate = calculate_ate(data, treatment_col, outcome_col, covariates)
    print(f"Estimated ATE: {ate}")'''