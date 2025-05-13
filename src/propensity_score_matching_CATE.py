cd import numpy as np
import scipy.stats
from propensity_score_matching import propensity_score_matching
from propensity_score_matching_ATE import ATE, linear_regression_treatment_effect
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.utils import shuffle
from propensity_scores import propensity_scores
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import precision_score, recall_score, classification_report
import sys


def mean_confidence_interval(data, confidence=0.95):
    #a = np.array(data)
    n = len(data)
    m = np.mean(data)              # average ATE (CATE)
    se = scipy.stats.sem(data)     # standard error of the mean
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)  # margin of error
    return np.array([m, m - h, m + h])      # mean and CI bounds. [mean, lower bound, upper bound]


def boostrap(df, boostrapping_steps=100, ate=True, linear_regression=True):
    out = []
    ATEs = []
    weights = []

    for i in tqdm(range(boostrapping_steps)):
        df = pd.read_csv('./data/preprocessed_data.csv')
        df = propensity_scores(df, sampling=True)
        df = propensity_score_matching(df)

        if ate:
            effect = ATE(df)
            ATEs.append(effect)

        if linear_regression:
            effect = linear_regression_treatment_effect(df)
            weights.append(effect)

    if ate:
        out.append(np.column_stack(ATEs))  # shape: (16, N)
    if linear_regression:
        out.append(np.column_stack(weights))  # shape: (16, N)

    return out


def CATE(data):
    output = []
    for i in data:
        temp = []
        for n in range(i.shape[0]):
            out = mean_confidence_interval(i[n])
            temp.append(out)
        output.append(np.array(temp))
    return output
        


if __name__ == "__main__":
    df = pd.read_csv('./data/preprocessed_data.csv')
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']
    
    out = boostrap(df, boostrapping_steps=500, ate=True, linear_regression=True)
    #print(out)
    #print(type(out))
    #print(type(out[0]))
    #sys.exit()

    out = CATE(out)
    

    df = pd.DataFrame({
    'drugs': drugs,
    'mean': out[0][:, 0],
    'lower bound': out[0][:, 1],
    'upper bound': out[0][:, 2]
    })
    df.to_csv('./results/CATE.csv', index=False)

    if len(out) > 1:
        df = pd.DataFrame({
            'drugs': drugs,
            'mean': out[1][:, 0],
            'lower bound': out[1][:, 1],
            'upper bound': out[1][:, 2]
            })
        df.to_csv('./results/CATE_using_linear_regression_weight.csv', index=False)

    '''
    for i in out:
        for n in range(len(drugs)):
            print(drugs[n]+" :")
            print("- mean:", i[n][0])
            print("- Lower bound:", i[n][1])
            print("- Upper bound:", i[n][2])
            print("***************************************")
    print("########################################")
    print("########################################")
    print("########################################")'''
    


    
