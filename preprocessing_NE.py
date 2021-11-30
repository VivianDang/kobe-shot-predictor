#%% [markdown]

### KAGGLE PROMPT: 

# * Using 20 years of data on Kobe's swishes and misses, can you predict which shots will find the bottom of the net?
# * This competition is well suited for practicing classification basics, feature engineering, and time series analysis.

# * Practice got Kobe an eight-figure contract and 5 championship rings. What will it get you?

#%% [markdown]

### SMART QUESTIONS: 
##### (1) Does the game situation affect accuracy?
    # * Regular Season vs Playoffs
    # * Periods / Minutes
    # * Clutch Time
##### (2) Does the spatial location of shots affect accuracy?
##### (3) Do his shots indicate the hot hand effect?

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

print("\nIMPORT SUCCESS")

#%%
# DATA IMPORTS
kobe = '/Users/nehat312/kobe-shot-predictor/data/data_NE.csv'
kobe = pd.read_csv(kobe, header = 0, index_col = 'shot_id')
kobe.info()

opp_stats = '/Users/nehat312/kobe-shot-predictor/images/image_scratch.xlsx'
opp_stats = pd.read_excel(opp_stats, sheet_name = 'OPP STATS', header = 0, index_col = 'OPP')
#opp_stats.info()

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
# DROP NA VALUES
kobe = kobe.dropna() # na in shot_made_flag

# DROP IRRELEVANT / REDUNDANT COLUMNS
kobe_clean = kobe.drop(['team_id', 'team_name', 'game_id', 'game_event_id', 'game_date', 'matchup', 'season'], axis = 1)

# EJECT OUTLIERS
#kobe_clean = kobe_clean[(kobe_clean['shot_distance'] <= 50)]
kobe_clean = kobe_clean[(kobe_clean['shot_distance'] <= 30)]
kobe_clean.info()

#%%
kobe_clean['opponent'].unique()

#%%
numerical_features = ['loc_x', 'loc_y', 'minutes_remaining', 'period', 'shot_distance']
categorical_features = ['combined_shot_type', 'playoffs','shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent']

kobe_categorical_one_hot = pd.get_dummies(kobe_clean[categorical_features])
kobe_cleaned = pd.concat([kobe_clean[numerical_features], kobe_categorical_one_hot], axis=1)
kobe_cleaned.head()

#%% [markdown]
### SITUATIONAL STATISTICS
#%%
# VARIABLE ASSIGNMENT

qtr1 = kobe_clean[(kobe_clean['period'] == 1)]
qtr2 = kobe_clean[(kobe_clean['period'] == 2)]
qtr3 = kobe_clean[(kobe_clean['period'] == 3)]
qtr4 = kobe_clean[(kobe_clean['period'] == 4)]
half1 = kobe_clean[(kobe_clean['period'] >= 1) & (kobe_clean['period'] <= 2)]
half2 = kobe_clean[(kobe_clean['period'] >= 3) & (kobe_clean['period'] <= 4)]
ot1 = kobe_clean[(kobe_clean['period'] == 5)]
ot2 = kobe_clean[(kobe_clean['period'] == 6)]
ot3 = kobe_clean[(kobe_clean['period'] == 7)]
overtime = kobe_clean[(kobe_clean['period'] >= 5)]

basetime = kobe_clean[(kobe_clean['period'] <= 4)]
clutchtime_1min = kobe_clean[(kobe_clean['period'] >= 4) & (kobe_clean['minutes_remaining'] <= 1)]
clutchtime_2min = kobe_clean[(kobe_clean['period'] >= 4) & (kobe_clean['minutes_remaining'] <= 2)]
clutchtime_5min = kobe_clean[(kobe_clean['period'] >= 4) & (kobe_clean['minutes_remaining'] <= 5)]

playoffs = kobe_clean[(kobe_clean['playoffs'] == 1)]
regular = kobe_clean[(kobe_clean['playoffs'] == 0)]

home = kobe_clean[(kobe_clean['home'] == 1)]
away = kobe_clean[(kobe_clean['away'] == 1)]

print("\nVARIABLES ASSIGNED")

#%%
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(12,9))
sns.histplot(data=qtr1, x="shot_distance", kde=True, color="gold", ax=axs[0, 0])
sns.histplot(data=qtr2, x="shot_distance", kde=True, color="darkviolet", ax=axs[0, 1])
sns.histplot(data=qtr3, x="shot_distance", kde=True, color="deepskyblue", ax=axs[1, 0])
sns.histplot(data=qtr4, x="shot_distance", kde=True, color="dimgrey", ax=axs[1, 1])

plt.show()

#%%
# SHOOTING SPLITS - BY PERIOD [1-7]
plt.figure(figsize=(12,9))
sns.boxplot(data=kobe_clean, x='period', y='shot_distance', hue='shot_made_flag', color = 'gold')
plt.title("FIELD GOAL ATTEMPTS BY PERIOD / SHOT DISTANCE", fontsize = 20)
plt.xlabel("PERIOD", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("SHOT DISTANCE", fontsize = 16)
plt.yticks(range(0,40,5));

#%%
# SHOOTING SPLITS - BY PERIOD [1-7]
qtr1_pct = pd.DataFrame(qtr1.groupby("shot_distance")["shot_made_flag"].mean())
qtr2_pct = pd.DataFrame(qtr2.groupby("shot_distance")["shot_made_flag"].mean())
qtr3_pct = pd.DataFrame(qtr3.groupby("shot_distance")["shot_made_flag"].mean())
qtr4_pct = pd.DataFrame(qtr4.groupby("shot_distance")["shot_made_flag"].mean())
overtime_pct = pd.DataFrame(overtime.groupby("shot_distance")["shot_made_flag"].mean())
qtrs_pct = pd.concat([qtr1_pct, qtr2_pct, qtr3_pct, qtr4_pct, overtime_pct], axis=1)
qtrs_pct.describe()

#%%
# SHOOTING SPLITS - BY MINUTES REMAINING [0-11]
plt.figure(figsize=(12,9))
sns.boxplot(data=kobe_clean, x='minutes_remaining', y='shot_distance', hue='shot_made_flag', color='gold')
plt.title("FG% BY MINUTE / SHOT DISTANCE", fontsize = 20)
plt.xlabel("MINUTES REMAINING", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("SHOT DISTANCE", fontsize = 16)
plt.yticks(range(0,40,5));

#%%
# SHOOTING SPLITS - BY HALF [1-2] / MINUTES REMAINING [0-11]
half1_pct = pd.DataFrame(half1.groupby("shot_distance")["shot_made_flag"].mean())
half2_pct = pd.DataFrame(half2.groupby("shot_distance")["shot_made_flag"].mean())
halves_pct = pd.concat([half1_pct, half2_pct], axis=1)
halves_pct.describe()

#%%
# SHOOTING SPLITS - CLUTCHTIME [PERIODS 4-7] / <5 MINUTES
plt.figure(figsize=(12,9))
sns.boxplot(data=clutchtime_5min, x='minutes_remaining', y='shot_distance', hue='shot_made_flag', color='gold')
plt.title("CLUTCH-TIME FG%", fontsize = 20)
plt.xlabel("MINUTES REMAINING", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("SHOT DISTANCE (FT.)", fontsize = 16)
plt.yticks(range(0,40,5));

#%%
# SHOOTING SPLITS - CLUTCHTIME [PERIODS 4-7] / <5 MINUTES
baseline_pct = pd.DataFrame(basetime.groupby("shot_distance")["shot_made_flag"].mean())
clutch_5min_pct = pd.DataFrame(clutchtime_5min.groupby("shot_distance")["shot_made_flag"].mean())
clutch_base_pct = pd.concat([baseline_pct, clutch_5min_pct], axis=1)
clutch_base_pct.describe()

#%%
# SHOOTING SPLITS - HOME / AWAY
pal1 = {0:'goldenrod', 1:'purple'}

plt.figure(figsize=(12,9))
sns.lineplot(data=kobe_clean, x='shot_distance', y='shot_made_flag', hue='home', style='home', legend=True, markers = True, palette=pal1)
plt.title("HOME/AWAY FG%", fontsize = 20)
plt.xlabel("SHOT DISTANCE (FT.)", fontsize = 16)
plt.xticks(range(0,35,5))
plt.ylabel("FIELD GOAL %", fontsize = 16);
#plt.yticks(range(0,1,2));

#%%
# SHOOTING SPLITS - HOME / AWAY
home_pct = pd.DataFrame(home.groupby("shot_distance")["shot_made_flag"].mean())
away_pct = pd.DataFrame(away.groupby("shot_distance")["shot_made_flag"].mean())
home_away_pct = pd.concat([home_pct, away_pct], axis=1)
home_away_pct.describe()

#%%
# SHOOTING SPLITS - PLAYOFFS / REGULAR SEASON
pal1 = {0:'goldenrod', 1:'purple'}

plt.figure(figsize=(12,9))
sns.lineplot(data=kobe_clean, x='shot_distance', y='shot_made_flag', hue='playoffs', style='playoffs', legend=True, markers = True, palette=pal1)
plt.title("PLAYOFFS / REGULAR SEASON FG%", fontsize = 20)
plt.xlabel("SHOT DISTANCE (FT.)", fontsize = 16)
plt.xticks(range(0,35,5))
plt.ylabel("FIELD GOAL %", fontsize = 16);
#plt.yticks(range(0,1.2,.2));

#%%
# SHOOTING SPLITS - PLAYOFFS / REGULAR SEASON
regular_splits = pd.DataFrame(regular.groupby("shot_distance")["shot_made_flag"].mean())
playoffs_splits = pd.DataFrame(playoffs.groupby("shot_distance")["shot_made_flag"].mean())
playoff_reg_splits = pd.concat([regular_splits, playoffs_splits], axis=1)
playoff_reg_splits.describe()

#%%
# SHOOTING SPLITS - OPPONENT
plt.figure(figsize=(12,12))
sns.lineplot(data=kobe_clean, x='game_year', y='opponent', size='shot_made_flag', palette='mako')
plt.title("PLAYOFF FG% BY MINUTE / SHOT DISTANCE", fontsize = 20)
plt.xlabel("PERIOD", fontsize = 16)
plt.xticks(range(1996,2017,1))
plt.ylabel("SHOT DISTANCE", fontsize = 16)
#plt.yticks(range(0,40,5));

#%%
# FEATURE ENGINEERING
#kobe_szn_splits1 = kobe_clean.groupby(["game_year", "opponent"])[["shot_made_flag", "shot_distance"]].mean()
kobe_opp_splits = pd.DataFrame(kobe_clean.groupby("opponent")[["shot_made_flag", "shot_distance"]].mean()).sort_values(by="shot_made_flag", ascending=False)
kobe_szn_splits3 = pd.DataFrame(kobe_clean.groupby("game_year")[["opponent", "shot_made_flag", "shot_distance"]].mean()).sort_values(by="shot_made_flag", ascending=False)

print(kobe_szn_splits3)
#kobe_szn_splits2
#kobe_szn_splits3

#%%
# FG% BY OPPONENT / SEASON
plt.figure(figsize=(24,12))
sns.barplot(data=kobe_clean, x='opponent', y='shot_made_flag', hue='game_year', palette='mako')
plt.title("FG% BY SEASON / SHOT DISTANCE / OPPONENT", fontsize = 20)
plt.xlabel("OPPONENT", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("FG%", fontsize = 16);

#%%
# FG% BY OPPONENT / SEASON
plt.figure(figsize=(24,12))
sns.barplot(data=kobe_opp_splits, x=kobe_opp_splits.index, y='shot_made_flag', palette='mako')
plt.title("FG% BY SEASON / SHOT DISTANCE / OPPONENT", fontsize = 20)
plt.xlabel("OPPONENT", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("FG%", fontsize = 16);

#%%
# FIELD GOAL % - BY SEASON / OPPONENT
plt.figure(figsize=(12,12))
sns.scatterplot(data=kobe_szn_splits1, x='game_year', y='shot_made_flag', hue='opponent', palette='mako')
plt.title("FG% BY SEASON / OPPONENT", fontsize = 20)
plt.xlabel("SEASON (YR)", fontsize = 16)
plt.xticks(range(1996,2017,1))
plt.ylabel("FIELD GOAL %", fontsize = 16);
#plt.yticks()

#%%
# HEATMAP - correlation generated to visualize target / variable relationships
kobe_corr = kobe.corr()[['shot_made_flag']].sort_values('shot_made_flag', ascending=False)
plt.figure(figsize=(12,15))
sns.heatmap(kobe_corr, annot = True, cmap = 'mako', vmin=-1, vmax=1, linecolor = 'white', linewidth = .005);

#%%
# SHOTS BY MONTH
# TEAM / OPPONENT ANALYSIS
# BIRTHDAY
# CHILD BDAY





#%%
# TRAIN-TEST/SPLIT
X = pd.concat([kobe_clean[numerical_features], kobe_categorical_one_hot], axis=1)
Y = kobe_clean['shot_made_flag']


#%%
print("\nANALYSIS CONCLUSION")