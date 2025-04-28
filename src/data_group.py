import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # use min max scaling for age


df = pd.read_csv('./data/split_condition.csv')

# look at the values in each columns
print('Original unique values: ')
print('Ethnicity concepts id: ', df['ethnicity_concept_id'].unique())
print('Gender concept id: ', df['gender_concept_id'].unique())
print('Race concept id: ', df['race_concept_id'].unique())

# convert each column to group numbers, create a mapping start with 0
df['ethnicity_group'] = pd.factorize(df['ethnicity_concept_id'])[0]
df['gender_group'] = pd.factorize(df['gender_concept_id'])[0]
df['race_group'] = pd.factorize(df['race_concept_id'])[0]

# create a mapping dictionary
ethnicity_map = {original: group for group, original in enumerate(pd.unique(df['ethnicity_concept_id']))}
gender_map = {original: group for group, original in enumerate(pd.unique(df['gender_concept_id']))}
race_map = {original: group for group, original in enumerate(pd.unique(df['race_concept_id']))}

# print out
print("\nEthnicity mapping (original->group): ")
for original, group in ethnicity_map.items():
    print(f'{original} -> {group}')

print("\nGender mapping (original->group): ")
for original, group in gender_map.items():
    print(f'{original} -> {group}')

print("\nRace mapping (original->group): ")
for original, group in race_map.items():
    print(f'{original} -> {group}')

# replace the original column
df['ethnicity_concept_id'] = df['ethnicity_group']
df['gender_concept_id'] = df['gender_group']
df['race_concept_id'] = df['race_group']
df.drop(['ethnicity_group', 'gender_group', 'race_group'], axis=1, inplace=True)


# MIN MAX SCALING
# get min max from the age column
print('\nAge before scaling: ')
print('Min: ', df['age'].min())
print('Max: ', df['age'].max())
print('Mean: ', df['age'].mean())
print("First 5 values:", df['age'].head().tolist())

# apply min max scaling
scaler = MinMaxScaler()
df['age_scaled'] = scaler.fit_transform(df[['age']])

# print out
print('\nAge after scaling: ')
print('Min: ', df['age_scaled'].min())
print('Max: ', df['age_scaled'].max())
print('Mean: ', df['age_scaled'].mean())

# replace the original age column
df['age'] = df['age_scaled']
df.drop('age_scaled', axis=1, inplace=True)

# save back to the file
df.to_csv('./data/sprocess_data.csv', index=False)
