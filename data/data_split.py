import pandas as pd
import numpy as  np 

df = pd.read_csv('./data/pooled_word2vec_condition.csv')

def split_condition(condition_str):
    # remove bracket and split by comma
    remove = condition_str.strip("[]").split(',')
    # convert each item to float
    return [float(x.strip()) for x in remove]

split = df['conditions'].apply(split_condition)

# create 5 new columns
for i in range(5):
    df[f'condition{i+1}'] = split.apply(lambda x : x[i])

# convert continuous to binary
for i in range(5):
    columns = f'condition{i+1}'
    median_value = df[columns].median()
    df[columns] = (df[columns] > median_value).astype(int)

median_age = df['age'].median()
df['age'] = (df['age'] > median_age).astype(int)

# convert categorical column to binary
def convert_category(df, column):
    category = df[column].unique()
    for i in category:
        # get indicies for category
        category_indicies = df[df[column] == i].index
        # if there are less than 2 rows then set to 0
        if len(category_indicies) < 2:
            df.loc[category_indicies, column] = 0
            continue
        random = np.random.random(len(category_indicies))
        # sorted indices based on the random
        sorted_index = [index for _, index in sorted(zip(random, category_indicies))]

        # calculate cutoff point 50%
        calculate = len(sorted_index) // 2

        # asign binary (0 to bottom 50%, 1 to top 50%)
        df.loc[sorted_index[:calculate], column] = 0
        df.loc[sorted_index[calculate:], column] = 1

category_cols = ['zip', 'ethnicity_concept_id', 'gender_concept_id', 'race_concept_id']
for i in category_cols:
    convert_category(df, i)


# drop the orginial condition column
df.drop('conditions', axis=1, inplace=True)

condition_cols = [f'condition{i+1}' for i in range(5)]
id_col = ['Unnamed: 0']
other_cols = [col for col in df.columns if col not in condition_cols and col != 'Unnamed: 0']
df = df[id_col + condition_cols + other_cols]
df.to_csv('./data/binary.csv', index=False)