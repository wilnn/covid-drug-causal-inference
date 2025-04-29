from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, classification_report



df = pd.read_csv('./data/preprocessed_data.csv')

def propensity_scores(df):
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    for drug in drugs: 
        condition = df[drug] == 1
        treatment_group = df.loc[condition, ['condition1', 'condition2', 'condition3',
                    'condition4', 'condition5', 'age', 'ethnicity_concept_id',
                    'gender_concept_id', 'race_concept_id']]
        ones = np.ones((treatment_group.shape[0],)) # label for treament_group
        #print(treatment_group.shape)

        treament_group_indexes = treatment_group.index # get the index of the selected row so that can
                                                # assign the propensity score to those row later
        #print(selected_indexes)
        #break
        treatment_group = treatment_group.to_numpy()


        condition2 = df[drugs[0]] == 0
        for i in range(1, len(drugs)):
            condition2 &= (df[drugs[i]] == 0)  # Update condition to match rows where drug value is 0
        control_group = df.loc[condition2, ['condition1', 'condition2', 'condition3', 'condition4',
                'condition5', 'age', 'ethnicity_concept_id', 'gender_concept_id',
                'race_concept_id']]

        control_group_indexes = control_group.index # get the index of the selected row so that can
                                                # assign the propensity score to those row later
        control_group = control_group.to_numpy()

        
        zeros = np.zeros((control_group.shape[0],)) # label for control_group
        #print(control_group.shape)
        #zeros = zeros[:treatment_group.shape[0]]
        #control_group = control_group[:treatment_group.shape[0]]
        
        X = np.concatenate((treatment_group, control_group), axis=0)
        y = np.concatenate((ones, zeros), axis=0)


        X_shuffled, y_shuffled = shuffle(X, y, random_state=42)

        #model = LogisticRegression(class_weight='balanced')
        model = LogisticRegression()

        model.fit(X_shuffled, y_shuffled)
        y_pred = model.predict_proba(treatment_group)
        

        df[f'{drug}_group_vs_control_group'] = np.nan
        df.loc[treament_group_indexes, f'{drug}_group_vs_control_group'] = y_pred[:, 1]
        y_pred = model.predict_proba(control_group)
        df.loc[control_group_indexes, f'{drug}_group_vs_control_group'] = y_pred[:, 1]
    return df



if __name__ == "__main__":
    df = propensity_scores(df)
    df.to_csv('./data/also_have_propensity_score.csv', index=False)
