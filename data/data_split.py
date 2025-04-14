import pandas as pd

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

# drop the orginial condition column
df.drop('conditions', axis=1, inplace=True)

condition_cols = [f'condition{i+1}' for i in range(5)]
id_col = ['Unnamed: 0']
other_cols = [col for col in df.columns if col not in condition_cols and col != 'Unnamed: 0']
df = df[id_col + condition_cols + other_cols]
df.to_csv('./data/pooled_word2vec_condition_split.csv', index=False)