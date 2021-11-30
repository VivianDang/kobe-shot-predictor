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
numerical_features = ['loc_x', 'loc_y', 'minutes_remaining', 'period', 'shot_distance']
categorical_features = ['combined_shot_type', 'playoffs','shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent']

kobe_categorical_one_hot = pd.get_dummies(kobe_clean[categorical_features])
kobe_cleaned = pd.concat([kobe_clean[numerical_features], kobe_categorical_one_hot], axis=1)
kobe_cleaned.head()

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

szn_1996 = kobe_clean[(kobe_clean['game_year'] == 1996)]
szn_1997 = kobe_clean[(kobe_clean['game_year'] == 1997)]
szn_1998 = kobe_clean[(kobe_clean['game_year'] == 1998)]
szn_1999 = kobe_clean[(kobe_clean['game_year'] == 1999)]
szn_2000 = kobe_clean[(kobe_clean['game_year'] == 2000)]
szn_2001 = kobe_clean[(kobe_clean['game_year'] == 2001)]
szn_2002 = kobe_clean[(kobe_clean['game_year'] == 2002)]
szn_2003 = kobe_clean[(kobe_clean['game_year'] == 2003)]
szn_2004 = kobe_clean[(kobe_clean['game_year'] == 2004)]
szn_2005 = kobe_clean[(kobe_clean['game_year'] == 2005)]
szn_2006 = kobe_clean[(kobe_clean['game_year'] == 2006)]
szn_2007 = kobe_clean[(kobe_clean['game_year'] == 2007)]
szn_2008 = kobe_clean[(kobe_clean['game_year'] == 2008)]
szn_2009 = kobe_clean[(kobe_clean['game_year'] == 2009)]
szn_2010 = kobe_clean[(kobe_clean['game_year'] == 2010)]
szn_2011 = kobe_clean[(kobe_clean['game_year'] == 2011)]
szn_2012 = kobe_clean[(kobe_clean['game_year'] == 2012)]
szn_2013 = kobe_clean[(kobe_clean['game_year'] == 2013)]
szn_2014 = kobe_clean[(kobe_clean['game_year'] == 2014)]
szn_2015 = kobe_clean[(kobe_clean['game_year'] == 2015)]
szn_2016 = kobe_clean[(kobe_clean['game_year'] == 2016)]

print("\nVARIABLES ASSIGNED")

#%% [markdown]
### SITUATIONAL STATISTICS

#%%
# SHOOTING SPLITS - QUARTERS
sns.set(style="darkgrid")
fig, axs = plt.subplots(2, 2, figsize=(12,9))
ax1 = sns.histplot(data=qtr1, x="shot_distance", kde=True, color="#fdb927", ax=axs[0, 0])
ax2 = sns.histplot(data=qtr2, x="shot_distance", kde=True, color="#552583", ax=axs[0, 1])
ax3 = sns.histplot(data=qtr3, x="shot_distance", kde=True, color="#fdb927", ax=axs[1, 0])
ax4 = sns.histplot(data=qtr4, x="shot_distance", kde=True, color="#552583", ax=axs[1, 1])
plt.yticks(range(0,1600,200));

#%%
# SHOOTING SPLITS - BY PERIOD [1-7]
pal1 = {0:'#fdb927', 1:'#552583'}
plt.figure(figsize=(12,9))
sns.boxplot(data=kobe_clean, x='period', y='shot_distance', hue='shot_made_flag', palette=pal1)
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
pal1 = {0:'#fdb927', 1:'#552583'}
plt.figure(figsize=(12,9))
sns.boxplot(data=kobe_clean, x='minutes_remaining', y='shot_distance', hue='shot_made_flag', palette=pal1)
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
pal1 = {0:'#fdb927', 1:'#552583'}
plt.figure(figsize=(12,9))
sns.boxplot(data=clutchtime_5min, x='minutes_remaining', y='shot_distance', hue='shot_made_flag', palette=pal1)
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
pal1 = {0:'#fdb927', 1:'#552583'}
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
pal1 = {0:'#fdb927', 1:'#552583'}
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
# SHOOTING SPLITS - SEASON / YEAR
year_cols = ['1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
splits_1996 = pd.DataFrame(szn_1996.groupby("opponent")["shot_made_flag"].mean())
splits_1997 = pd.DataFrame(szn_1997.groupby("opponent")["shot_made_flag"].mean())
splits_1998 = pd.DataFrame(szn_1998.groupby("opponent")["shot_made_flag"].mean())
splits_1999 = pd.DataFrame(szn_1999.groupby("opponent")["shot_made_flag"].mean())
splits_2000 = pd.DataFrame(szn_2000.groupby("opponent")["shot_made_flag"].mean())
splits_2001 = pd.DataFrame(szn_2001.groupby("opponent")["shot_made_flag"].mean())
splits_2002 = pd.DataFrame(szn_2002.groupby("opponent")["shot_made_flag"].mean())
splits_2003 = pd.DataFrame(szn_2003.groupby("opponent")["shot_made_flag"].mean())
splits_2004 = pd.DataFrame(szn_2004.groupby("opponent")["shot_made_flag"].mean())
splits_2005 = pd.DataFrame(szn_2005.groupby("opponent")["shot_made_flag"].mean())
splits_2006 = pd.DataFrame(szn_2006.groupby("opponent")["shot_made_flag"].mean())
splits_2007 = pd.DataFrame(szn_2007.groupby("opponent")["shot_made_flag"].mean())
splits_2008 = pd.DataFrame(szn_2008.groupby("opponent")["shot_made_flag"].mean())
splits_2009 = pd.DataFrame(szn_2009.groupby("opponent")["shot_made_flag"].mean())
splits_2010 = pd.DataFrame(szn_2010.groupby("opponent")["shot_made_flag"].mean())
splits_2011 = pd.DataFrame(szn_2011.groupby("opponent")["shot_made_flag"].mean())
splits_2012 = pd.DataFrame(szn_2012.groupby("opponent")["shot_made_flag"].mean())
splits_2013 = pd.DataFrame(szn_2013.groupby("opponent")["shot_made_flag"].mean())
splits_2014 = pd.DataFrame(szn_2014.groupby("opponent")["shot_made_flag"].mean())
splits_2015 = pd.DataFrame(szn_2015.groupby("opponent")["shot_made_flag"].mean())
splits_2016 = pd.DataFrame(szn_2016.groupby("opponent")["shot_made_flag"].mean())
yearly_splits = pd.concat([splits_1996, splits_1997, splits_1998, splits_1999, splits_2000, splits_2001, splits_2002, splits_2003, splits_2004, splits_2005, splits_2006, splits_2007, splits_2008, splits_2009, splits_2010, splits_2011, splits_2012, splits_2013, splits_2014, splits_2015, splits_2016], axis=1)
yearly_splits.columns = year_cols
yearly_splits

#%%
# OPPONENT FILTERING / SORTING
kobe_opp_splits = pd.DataFrame(kobe_clean.groupby("opponent")[["shot_made_flag", "shot_distance"]].mean()).sort_values(by="shot_made_flag", ascending=False)
#kobe_opp_splits2 = pd.DataFrame(kobe_clean.groupby("opponent", "game_year")[["shot_made_flag", "shot_distance"]].mean()).sort_values(by="shot_made_flag", ascending=False)
#kobe_opp_splits3 = pd.DataFrame(kobe_clean.groupby("opponent")[["shot_made_flag", "shot_distance", "game_year"]])

#%%
# FG% BY OPPONENT
pal_opps = {'#fdb927', '#552583'}
plt.figure(figsize=(18,9))
sns.barplot(data=kobe_opp_splits, x='shot_made_flag', y=kobe_opp_splits.index, palette=pal_opps)
plt.title("FG% BY OPPONENT", fontsize = 20)
plt.xlabel("FG%", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("OPPONENT", fontsize = 16);

#%%
# FG% BY OPPONENT / H/A
pal_opps = {'#fdb927', '#552583'}
plt.figure(figsize=(18,9))
sns.violinplot(data = yearly_splits, palette=pal_opps, legend=True)
plt.title("FG% BY OPPONENT", fontsize = 20)
plt.xlabel("FG%", fontsize = 16)
#plt.xticks(range(1996,2017,1))
plt.ylabel("OPPONENT", fontsize = 16)
plt.legend(loc='best');

#%%
# TEAMS BY SEASON
pal1 = {0:'#fdb927', 1:'#552583'}
fig, axs = plt.subplots(2, 2, figsize=(24,18))
ax1 = sns.scatterplot(data=szn_1998, x="shot_distance", y="opponent", hue='shot_made_flag', palette=pal1, ax=axs[0, 0])
ax2 = sns.barplot(data=szn_1999, x="shot_made_flag", y="opponent", color="#552583", ax=axs[0, 1])
ax3 = sns.barplot(data=szn_2000, x="shot_made_flag", y="opponent", color="#fdb927", ax=axs[1, 0])
ax4 = sns.barplot(data=szn_2001, x="shot_made_flag", y="opponent", color="#552583", ax=axs[1, 1])
#plt.xticks(range(1996,2017,1))
#plt.yticks(range(0,1600,200))
plt.show();
#%%
# TEAMS BY SEASON
pal1 = {0:'#fdb927', 1:'#552583'}
fig, axs = plt.subplots(2, 2, figsize=(24,18))
ax1 = sns.scatterplot(data=szn_1998, x="shot_distance", y="opponent", hue='shot_made_flag', palette=pal1, ax=axs[0, 0])
ax2 = sns.barplot(data=szn_1999, x="shot_made_flag", y="opponent", color="#552583", ax=axs[0, 1])
ax3 = sns.barplot(data=szn_2000, x="shot_made_flag", y="opponent", color="#fdb927", ax=axs[1, 0])
ax4 = sns.barplot(data=szn_2001, x="shot_made_flag", y="opponent", color="#552583", ax=axs[1, 1])
#plt.xticks(range(1996,2017,1))
#plt.yticks(range(0,1600,200))
plt.show();



#%%
# SHOOTING SPLITS - MONTHLY
pal1 = {0:'#fdb927', 1:'#552583'}

plt.figure(figsize=(12,9))
sns.barplot(data=home, x='game_month', y='shot_made_flag', hue='home', style='home', legend=True, markers = True, palette=pal1)
plt.title("PLAYOFFS / REGULAR SEASON FG%", fontsize = 20)
plt.xlabel("SHOT DISTANCE (FT.)", fontsize = 16)
plt.xticks(range(0,13,1))
plt.ylabel("FIELD GOAL %", fontsize = 16);
#plt.yticks(range(0,1.2,.2));


#%%
# HEATMAP - correlation generated to visualize target / variable relationships
kobe_corr = kobe_clean.corr()[['shot_made_flag']].sort_values('shot_made_flag', ascending=False)
plt.figure(figsize=(12,15))
sns.heatmap(kobe_corr, annot = True, cmap = 'mako', vmin=-1, vmax=1, linecolor = 'white', linewidth = .005);

#%%
# SHOTS BY MONTH
# TEAM / OPPONENT ANALYSIS
# BIRTHDAY
# CHILD BDAY



#%%
print("\nEDA CONCLUSION")