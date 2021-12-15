
#%%
import pandas as pd
import numpy as np

df = pd.read_csv("data/data.csv")
# %%
# drop nan
df = df.dropna() # na in shot_made_flag
# drop columns that don't give information
df = df.drop(["team_id", "team_name"], axis=1)

#%%
df['secondsFromGameStart'] = (df['period'] <= 4).astype(int)*(df['period']-1)*12*60 + (df['period'] > 4).astype(int)*((df['period']-4)*5*60 + 3*12*60) + 60*(11-df['minutes_remaining'])+(60-df['seconds_remaining'])
# %%
df = df.reset_index(drop=True)
df.to_csv('data/data_cleaned.csv', index = False)
