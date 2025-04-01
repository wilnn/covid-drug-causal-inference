import pandas as pd
import numpy as np
data = pd.read_csv("./data/N3C_data_10000_sample.csv")  # Skip header


data.to_excel("./data/N3C_data_10000_sample.xlsx", index=False)


