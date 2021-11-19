
#%%
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
# %%
# drop nan
df = df.dropna() # na in shot_made_flag
# drop columns that don't give information
df = df.drop(["team_id", "team_name"], axis=1)