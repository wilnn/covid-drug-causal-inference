import numpy as np
import pandas as pd
import sys
table = []


def causal_table(path, save=False):

    df = pd.read_csv(path)
    cols = df.columns.tolist()
    drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine', 
                'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine',
                'desvenlafaxine', 'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']

    tempcols = ['treatments']
    for col in cols:
        if col in drugs or col == "Unnamed: 0" or col == 'outcome' or col == 'severity_covid_death' or col == 'zip':
            continue
        for i, value in enumerate(df[col].unique()):
            tempcols.append(col+'_'+str(value))
    tempcols.append('totals')
    tempcols.append('ATEs')

    tempdf = pd.DataFrame(columns=tempcols)
    #print(tempdf.columns.tolist())
    first = True
    col_count = 0
    num_unique = []
    for drug in drugs:
        row = []
        row.append(drug)
        # total count of patients who took the drug and survived (outcome == 1)
        total = df[df[drug] == 1].shape[0]

        # count people who took this drug and have condition 0 and survived (outcome == 1)
        # and count people who took this drug and have condition 0 and survived (outcome == 1)
        
        for col in cols:
            if col in drugs or col == "Unnamed: 0" or col == 'outcome' or col == 'severity_covid_death' or col == 'zip':
                continue
            if first:
                col_count += 1
                num_unique.append(df[col].nunique())
            for i, value in enumerate(df[col].unique()):

                if save:
                    cell = []
                # Count of patients who took the drug (drug == 1), have condition 0, and survived (outcome == 1)
                count_condition0 = df[(df[drug] == 1) & (df[col] == value) & (df['outcome'] == 1)].shape[0]
                count_condition0_total = df[(df[drug] == 1) & (df[col] == value)].shape[0]
                
                if save:
                    cell.append(count_condition0)
                    cell.append(count_condition0_total)
                    row.append(str(cell))
                else:
                    cell = np.array([count_condition0, count_condition0_total])
                    row.append(cell)
                    #print(cell)

            
            #row.append(count_condition0)
            #row.append(count_condition1)
        row.append(total)
        row.append(np.NaN)
        table.append(row)
        first = False
    #print(table)
    if save:
        for i, row in enumerate(table):
            tempdf.loc[-1] = row  # adding a row
            tempdf.index = tempdf.index + 1  # shifting index
        #tempdf = tempdf.sort_index()
        tempdf.to_csv('./data/causal_table.csv', index=False)
    
            
    #print(tempdf.columns.tolist())
    #print(tempdf)
    
    if save:
        return table, col_count, num_unique
    return np.array(table, dtype=object), col_count, num_unique


def calculate_ate(table, col_count, num_unique, save=False):
    #print(table[0])
    totals_sum = np.sum(table[:, -2])
    print(totals_sum)
    #sys.exit()
    ATEs = []
    print(table[:, 0])
    for index, row in enumerate(table):

        n = 0
        total_causal_effect_score = 0 # total_causal_effect_score for all confounder columns 
        
        i = 1
        print("each variable", index)
        while i < sum(num_unique):
            
            total_causal_effect_score += calculate_causal_effect_score(table[:, i:i+num_unique[n]], totals_sum, index, num_unique[n])
            
            i+=num_unique[n] # next variables
            n += 1 # index to get the number of unique value for that variables.

            #print("11111111111111111111111111111111111111111")
        #sys.exit()
    
        ATEs.append(total_causal_effect_score/col_count)
    return ATEs



def calculate_causal_effect_score(value_cols, totals_sum, row_index, num_unique):
    """
    Calculate the causal effect score for each drug.
    value_cols: np.array. an array of columns where each column is a value of a column/feature/confounder
    and each cell has 2 number: the first number is the count of patients of have this value for this feature
    and take the durg at that row and recovered (outcome == 1), the second number is the total number
    of patients who have taken this drug and 
    totals_sum: int. is the sum of all the cells in the totals column
    """
    #print(value_cols)
    #sys.exit()
    #print(value_cols)
    #print("******************************")
    value_cols = np.array([[*row] for row in value_cols])
    causal_effect_score = 0
    print("***********")
    for i in range(num_unique):
        
        print(num_unique)
        print(value_cols[row_index, i, 0], value_cols[row_index, i, 1])
        if value_cols[row_index, i, 1] == 0:
            prob = 0
        else:
            prob = value_cols[row_index, i, 0] / value_cols[row_index, i, 1]
        print(prob)

        causal_effect_score += (np.sum(value_cols[:, i, 1])/totals_sum) * prob
        print("Causal Effect Score: ", causal_effect_score)
            
    return causal_effect_score


q = 0
def main():
    table, col_count, num_unique = causal_table('./data/binary.csv')
    #print(col_count)
    #print(num_unique)
    #print(table.shape)
    #print(table[0])
    ATEs = calculate_ate(table, col_count, num_unique)
    df = pd.read_csv('./results/causal_table.csv')
    df['ATEs'] = ATEs
    print(ATEs)
    df.to_csv('./results/causal_table.csv', index=False)

if __name__ == "__main__":
    main()



