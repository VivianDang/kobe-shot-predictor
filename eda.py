#%%
from numpy import average
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data_cleaned.csv")


#%%
sns.set_theme(style = 'darkgrid')
#%%[markdown]
## Hot handedness

#%%
madeLastShotList   = [None]
timeDifferenceFromLastShotList   = [None]

for shot in range(1,len(df)):
    # make sure the current shot and last shot were all in the same game
    if df.loc[shot,'game_id'] == df.loc[shot-1,'game_id'] and df.loc[shot,'period'] == df.loc[shot-1,'period']:
        madeLastShotList.append(df.loc[shot-1,'shot_made_flag'] == 1)        
        timeDifferenceFromLastShotList.append(df.loc[shot,'secondsFromGameStart'] - df.loc[shot-1,'secondsFromGameStart'])
    else:
        madeLastShotList.append(None)        
        timeDifferenceFromLastShotList.append(None)

 # %%
df['made_last_shot'] = madeLastShotList
# %%

sns.barplot(y='shot_made_flag', x = 'made_last_shot', data = df, estimator = np.mean)
# %%
sns.countplot(x='made_last_shot', hue = 'shot_made_flag', data = df)

# %%
from scipy.stats import ttest_ind
ttest_ind(df[df['made_last_shot']==1]['shot_made_flag'], df[df['made_last_shot']==0]['shot_made_flag'])

# No significant difference == no hot hand effect
