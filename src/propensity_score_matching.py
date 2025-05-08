import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sys

df = pd.read_csv('./data/also_have_propensity_score.csv')



def propensity_score_matching(df):
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    for drug in drugs: 
        condition = df[drug] == 1
        treatment_group = df.loc[condition, [f'{drug}_group_vs_control_group']]
        #print(treatment_group.shape)



        treament_group_indices = treatment_group.index # get the index of the selected row so that can
                                                # assign the propensity score to those row later
        #print(selected_indexes)
        #break
        treatment_group = treatment_group.to_numpy()


        # loop to select the rows(patients) that do not take any drugs
        condition2 = df[drugs[0]] == 0
        for i in range(1, len(drugs)):
            condition2 &= (df[drugs[i]] == 0)  # Update condition to match rows where drug value is 0
        
        control_group = df.loc[condition2, [f'{drug}_group_vs_control_group']]

        control_group_indices = control_group.index # get the index of the selected row so that can
                                                # assign the propensity score to those row later
        control_group = control_group.to_numpy()

        control_group = control_group[~np.isnan(control_group)]
        
        #control_group = control_group[:treatment_group.shape[0]]

        knn = NearestNeighbors(n_neighbors=1)  # 1 nearest neighbor
        #print(control_group)
        knn.fit(control_group.reshape(-1, 1)) # reshape to make control_group becomes a 2D array. Because scikit-learn’s NearestNeighbors expects 2D input
        
        distances, indices = knn.kneighbors(treatment_group.reshape(-1, 1))
        #print(control_group[indices[0][0]])

        #break

        matched_control_indices = control_group_indices[indices.flatten()]


        df[f'{drug}_group_match_id_in_control_group'] = np.nan

        df.loc[treament_group_indices, f'{drug}_group_match_id_in_control_group'] = matched_control_indices

    return df


if __name__ == "__main__":
    df = propensity_score_matching(df)
    df.to_csv('./data/after_propensity_score_matching.csv', index=False)
