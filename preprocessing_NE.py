#%% [markdown]

### KAGGLE PROMPT: 

# Using 20 years of data on Kobe's swishes and misses, can you predict which shots will find the bottom of the net?
# This competition is well suited for practicing classification basics, feature engineering, and time series analysis.

# Practice got Kobe an eight-figure contract and 5 championship rings. What will it get you?

#%%
# LIBRARY IMPORTS
import os

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats as stats
import statistics
import datetime as dt

print("\nIMPORT SUCCESS.")

#%%
# DATA IMPORTS
kobe = 'data.csv'
kobe = pd.read_csv(kobe, header = 0, index_col = 'shot_id')
kobe.info()
#kobe.head()
#kobe.columns

#%% [markdown]
## DATA DICTIONARY

# * shot_made_flag = [0, 1]
# * playoffs = [0, 1]
# * game_event_id = [2~659]
# * lat = 
# * lon = 
# * loc_x = 
# * loc_y = 
# * minutes_remaining = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# * seconds_remaining = [27, 22, 45, 52, 19, 32,  5, 12, 36, 56,  0,  9, 44, 16, 48,  1, 50, 29, 46,  8,  4, 57, 47, 11, 30, 20, 26, 58, 33, 13, 59, 21, 55, 38, 6, 40, 10,  2, 37, 17, 53, 15, 24, 49, 41, 54, 25, 39, 14, 43, 23, 18, 34, 51, 28,  3,  7, 42, 35, 31]
# * period = [1, 2, 3, 4]
# * shot_distance = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69, 70, 71, 74, 77, 79]
# * season = ['1996-97', '1997-98', '1998-99', '1999-00', '2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16']
# * shot_type = ['2PT Field Goal', '3PT Field Goal']
# * combined_shot_type = ['Jump Shot', 'Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Bank Shot']
# * shot_zone_range = ['Right Side(R)', 'Left Side(L)', 'Left Side Center(LC)', 'Right Side Center(RC)', 'Center(C)', 'Back Court(BC)']
# * shot_zone_basic = ['Mid-Range', 'Restricted Area', 'In The Paint (Non-RA)', 'Above the Break 3', 'Right Corner 3', 'Backcourt', 'Left Corner 3']
# * matchup = ['LAL @ POR', 'LAL vs. UTA', 'LAL @ VAN', 'LAL vs. LAC', 'LAL @ HOU', 'LAL @ SAS', 'LAL vs. HOU', 'LAL vs. DEN', 'LAL @ SAC', 'LAL @ DEN', 'LAL vs. CHI', 'LAL vs. GSW', 'LAL vs. MIN', 'LAL @ LAC', 'LAL vs. IND', 'LAL @ SEA', 'LAL vs. SAS', 'LAL vs. DAL', 'LAL vs. PHI', 'LAL @ GSW', 'LAL vs. SEA', 'LAL vs. DET', 'LAL vs. MIL', 'LAL vs. VAN', 'LAL @ TOR', 'LAL @ MIA', 'LAL @ DAL', 'LAL vs. POR', 'LAL @ PHX', 'LAL vs. CLE', 'LAL @ UTA', 'LAL vs. MIA', 'LAL vs. NJN', 'LAL @ NYK', 'LAL @ CLE', 'LAL @ MIN', 'LAL vs. CHH', 'LAL vs. SAC', 'LAL vs. PHX', 'LAL @ NJN', 'LAL @ PHI', 'LAL @ CHH', 'LAL @ IND', 'LAL vs. TOR', 'LAL @ DET', 'LAL @ WAS', 'LAL @ ORL', 'LAL @ ATL', 'LAL @ MIL', 'LAL vs. NYK', 'LAL vs. MEM', 'LAL vs. ORL', 'LAL @ MEM', 'LAL @ CHI', 'LAL vs. WAS', 'LAL vs. ATL', 'LAL vs. BOS', 'LAL @ BOS', 'LAL vs. NOH', 'LAL @ NOH', 'LAL @ UTH', 'LAL vs. SAN', 'LAL @ NOK', 'LAL @ PHO', 'LAL vs. NOK', 'LAL vs. PHO', 'LAL @ CHA', 'LAL vs. CHA', 'LAL vs. OKC', 'LAL @ OKC', 'LAL vs. BKN', 'LAL @ BKN', 'LAL @ NOP', 'LAL vs. NOP']
# * opponent = ['POR', 'UTA', 'VAN', 'LAC', 'HOU', 'SAS', 'DEN', 'SAC', 'CHI', 'GSW', 'MIN', 'IND', 'SEA', 'DAL', 'PHI', 'DET', 'MIL', 'TOR', 'MIA', 'PHX', 'CLE', 'NJN', 'NYK', 'CHA', 'WAS', 'ORL', 'ATL', 'MEM', 'BOS', 'NOH', 'NOP', 'OKC', 'BKN']

#%%
# EXPLORATION
#kobe['game_date'].sort_values(ascending = True).unique()[0:200]

#%%
# VARIABLE ASSIGNMENT
    # clutchtime = 
    # overtime = 

#%%
# DROP NA VALUES
kobe = kobe.dropna() # na in shot_made_flag

# DROP IRRELEVANT / REDUNDANT COLUMNS
kobe_clean = kobe.drop(['team_id', 'team_name', 'game_id', 'game_event_id', 'game_date', 'matchup', 'season'], axis = 1)
kobe_clean.info()

#%%
# CLEANING / RE-FORMATTING
    # 'season'
    # 'game_date'
    # 'shot_distance'

#%%
# MAPPING / DUMMIFYING
    # 'action_type'
    # 'combined_shot_type'
    # 'shot_type'
    # 'shot_zone_area'
    # 'shot_zone_basic'
    # 'shot_zone_range'

#%%
# ONE-HOT ENCODING

#%%
# FEATURE ENGINEERING

#%%
# TEAM / OPPONENT ANALYSIS

#%%

# HEATMAP - correlation generated to visualize target / variable relationships
kobe_corr = kobe.corr()[['shot_made_flag']].sort_values('shot_made_flag', ascending=False)
plt.figure(figsize=(12,15))
sns.heatmap(kobe_corr, annot = True, cmap = 'mako', vmin=-1, vmax=1, linecolor = 'white', linewidth = .005);

#%%
# PAIRPLOT - ALL COLUMNS - **TAKES A MINUTE TO RUN**
plt.figure(figsize=(12,12))
sns.pairplot(kobe_clean, palette = 'mako');

#%%
# PAIRPLOT - SHOT COLUMNS - **TAKES A MINUTE TO RUN**
kobe_shots = kobe[kobe['']]
plt.figure(figsize=(12,12))
sns.pairplot(kobe_clean, palette = 'mako');

'Jump Shot', 'Dunk', 'Layup', 'Tip Shot', 'Hook Shot', 'Bank Shot'

#%%
print("\nREADY TO CONTINUE.")