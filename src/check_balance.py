import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_mean_difference(df, drug, variables, treatment, control_col=None):
    # calculate standardized mean difference between treatment and control group
    treatment_mask = df[drug] == 1       # create mask for treatment group

    # before matching, control group is patients who didn't take any drugs
    if control_col is None:
       drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine',
             'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine',
             'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
       control_mask = df[drugs[0]] == 0
       for i in range(1, len(drugs)):
           control_mask &= (df[drugs[i]] == 0) 
    else:
        # after matching, use only matched control patients
        match_control = df.loc[treatment_mask, control_col].dropna().values.astype(int)
        control_mask = df.index.isin(match_control)

    mean_diffs = {}
    for var in variables:
        # get values for treatment and control groups
        treat_values = df.loc[treatment_mask, var]
        control_values = df.loc[control_mask, var]

        # calculate mean
        treat_mean = treat_values.mean()    # X_t
        control_mean = control_values.mean()    # X_c

        # calculate standard deviation
        treat_var = treat_values.var()  # S^2t
        control_var = control_values.var()  # S^2 c
        std = np.sqrt((treat_var + control_var) / 2)

        if std > 0:
            std_mean_diff = abs(treat_mean - control_mean) / std    # |X_t - X_c| / √[(S^2_t+S^2_C)/2]
        else:
            std_mean_diff = 0

        mean_diffs[var] = std_mean_diff

    return mean_diffs

def plot(before_diffs, after_diffs, drugs, output_file = None):
    variables = list(before_diffs.keys())
    before_values = [before_diffs[var] for var in variables]
    after_values = [after_diffs[var] for var in variables]
    plt.figure(figsize=(12,7))
    plt.plot(variables, before_values, marker='o', linestyle='-', color='#1f77b4',
             linewidth=2, markersize=8, label='Before Propensity Score Matching')
    plt.plot(variables, after_values, marker='o', linestyle='-', color='#ff7f0e',
             linewidth=2, markersize=8, label='After Propensity Score Matching')
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7)
    # set lables and title
    plt.title(f'Mean Difference Before And After Matching for {drugs}')
    plt.ylabel('Standardized Mean Difference', fontsize=14)
    plt.xlabel('Variables', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # read the data
    if os.path.exists("./data/after_propensity_score_matching.csv"):
        df = pd.read_csv("./data/after_propensity_score_matching.csv")
    else:
        df = pd.read_csv("./data/also_have_propensity_score.csv")

    covars = ['condition1', 'condition2', 'condition3', 'condition4', 'condition5', 'age',
               'ethnicity_concept_id', 'gender_concept_id', 'race_concept_id']
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine',
             'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine',
             'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    # create a directorty for the plots if doesn't exist
    # os.makedirs('./plot', exist_ok=True)
    for drug in drugs:
        # check if propensity score column exist
        propensity_col = f'{drug}_group_vs_control_group'
        matching_col = f'{drug}_group_match_id_in_control_group'
        if propensity_col not in df.columns:
            print(f'Skipping {drug} - propensity score column not found')
            continue

        # calculate differences before matching
        before_diffs = calculate_mean_difference(df, drug, covars, propensity_col)

        # check if matching column exists
        if matching_col in df.columns:
            after_diffs = calculate_mean_difference(df, drug, covars, propensity_col, matching_col)
        else:
            print(f'Matching column for {drug} not found')
            continue 
        plot(before_diffs, after_diffs, drug)

        # print differences
        print(f"Standardized mean differences for {drug}:")
        for var in covars:
            print(f"{var}: Before = {before_diffs[var]:.4f}, After = {after_diffs[var]:.4f}")

