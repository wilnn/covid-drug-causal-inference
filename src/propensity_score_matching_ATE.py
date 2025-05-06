import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle


df = pd.read_csv('./data/after_propensity_score_matching.csv')



def ATE(df):
    print("ATE:")
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    for drug in drugs: 
        condition = df[drug] == 1
        treatment_group = df.loc[condition, ['outcome', f'{drug}_group_match_id_in_control_group']]
        #print(treatment_group.shape)

        #treament_group_indices = treatment_group.index # get the index of the selected row so that can
                                                # assign the propensity score to those row later
        
        treatment_group_outcome = df['outcome'].to_numpy()

        control_match_indices = df[f'{drug}_group_match_id_in_control_group'].to_numpy()

        # Remove NaN entries
        control_match_indices = control_match_indices[~np.isnan(control_match_indices)]

        matched_control_group_outcome = df.loc[control_match_indices]['outcome'].to_numpy()
        print(f"{drug}: ", np.mean(treatment_group_outcome) - np.mean(matched_control_group_outcome))


def linear_regression_treatment_effect(df):
    print("linear_regression_treatment_effect:")
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    for drug in drugs: 
        condition = df[drug] == 1
        treatment_group = df.loc[condition, ['condition1', 'condition2', 'condition3',
                    'condition4', 'condition5', 'age', 'ethnicity_concept_id',
                    'gender_concept_id', 'race_concept_id', f"{drug}"]].to_numpy()

        treatment_group_outcome = df.loc[condition, "outcome"].to_numpy()

        control_match_indices = df[f'{drug}_group_match_id_in_control_group'].to_numpy()

        # Remove NaN entries
        control_match_indices = control_match_indices[~np.isnan(control_match_indices)]

        matched_control_group = df.loc[control_match_indices, ['condition1', 'condition2', 'condition3',
                    'condition4', 'condition5', 'age', 'ethnicity_concept_id',
                    'gender_concept_id', 'race_concept_id', f"{drug}"]].to_numpy()
        
        matched_control_group_outcome = df.loc[control_match_indices, ['outcome']].to_numpy().flatten()

        #print(treatment_group_outcome.shape)

        #break

        #ones = np.ones((treatment_group.shape[0],)) # label for treament_group
        #zeros = np.zeros((treatment_group.shape[0],)) # label for treament_group




        X = np.concatenate((treatment_group, matched_control_group), axis=0)
        y = np.concatenate((treatment_group_outcome, matched_control_group_outcome), axis=0)


        X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

        #model = LogisticRegression(class_weight='balanced')
        model = LinearRegression()

        model.fit(X_shuffled, y_shuffled)

        weights = model.coef_


        bias = model.intercept_
        '''
        print(f"{drug}:")
        print("Feature weights:", weights)
        print("Bias (intercept):", bias)
        print("***************************************************************\n")'''
        
        print(f"{drug}: ", weights[-1])


if __name__ == "__main__":
    ATE(df)
    print("\n*******************************************\n")
    linear_regression_treatment_effect(df)
