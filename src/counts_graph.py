import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/N3C_data_10000_sample.csv")

# list of drug columns 
drugs = ['trazodone', 'amitriptyline', 'fluoxetine', 'citalopram', 'paroxetine', 'venlafaxine',
             'vilazodone', 'vortioxetine', 'sertraline', 'bupropion', 'mirtazapine', 'desvenlafaxine',
             'doxepin', 'duloxetine', 'escitalopram', 'nortriptyline']

took_drug_count = []
# no_drug_count = []
percentages = []

for drug in drugs:
    # convert numeric to ensure proper counting (just in case of string values)
    df[drug] = pd.to_numeric(df[drug], errors='coerce').fillna(0)

# patients who took no drugs at all
no_drugs = (df[drugs] == 0).all(axis=1)
no_drugs_count = no_drugs.sum()
total_patients = df.shape[0]

for drug in drugs:
    # count patients who took drug
    took_drug = df[df[drug] == 1].shape[0]
    percentage = 100 * took_drug / total_patients
    took_drug_count.append(took_drug)
    percentages.append(percentage)

results = pd.DataFrame({
    'Drug' : drugs,
    'Took Drug' : took_drug_count,
    'Percentage' : percentages
})

# sort by number of people who took the drug (descending)
results = results.sort_values('Took Drug', ascending=False)
fig, ax = plt.subplots(figsize=(14, 10))
x = np.arange(len(drugs))
bar_width = 0.7
# create the drug bar
bars = ax.bar(x, results['Took Drug'], bar_width, label='Took Drug', color='#ff7f0e')
# no drug bar will be after the last drug bar
no_drug_x = len(drugs) + 1
no_drug_bar = ax.bar(no_drug_x, no_drugs_count, bar_width, label='Took No Drugs', color='#1f77b4')
ax.set_xlabel('Drug', fontsize=14)
ax.set_ylabel('Number of Patients', fontsize=14)
ax.set_title('Number of Patients Taking Each Drug vs. Not Taking Any Drug', fontsize=16)

all_x = np.append(x, no_drug_x)
all_labels = list(results['Drug']) + ['No Drugs']
ax.set_xticks(all_x)
ax.set_xticklabels(all_labels, rotation=45, ha='right')
ax.legend()

for i, bar in enumerate(bars):
    height = bar.get_height()
    percentage = results['Percentage'].iloc[i]
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=9)

# Add text label to the "No Drugs" bar
no_drug_percentage = 100 * no_drugs_count / total_patients
ax.text(no_drug_x, no_drugs_count + 50,
        f'{int(no_drugs_count)}\n({no_drug_percentage:.1f}%)',
        ha='center', va='bottom', fontsize=9)

# add a line showing total number of patients
plt.axhline(y=total_patients, color='red', linestyle='--', alpha=0.7)
plt.text(0, total_patients + 100, f'Total Patients: {total_patients}', color='red', fontsize=12)

plt.tight_layout()
plt.show()