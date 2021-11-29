#%%
# LIBRARY IMPORTS
import os
from matplotlib import colors

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

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

#%%
# DROP NA VALUES
kobe = kobe.dropna() # na in shot_made_flag

# DROP IRRELEVANT / REDUNDANT COLUMNS
kobe_clean = kobe.drop(['team_id', 'team_name', 'game_id', 'game_event_id', 'game_date', 'matchup', 'season'], axis = 1)
kobe_clean.info()
#%%

#%% 
# draw_court() function from MichaelKrueger's excelent script
def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


#%%
# define subsets
kobe_spatial = kobe_clean[["loc_x", "loc_y", "action_type", "combined_shot_type", "shot_zone_area", "shot_type", "shot_zone_basic", "shot_zone_range", "shot_distance", "shot_made_flag"]].copy()
# kobe_spatial = kobe_spatial[kobe_spatial["action_type"]!=kobe_spatial["combined_shot_type"]]
kobe_jump = kobe_spatial[kobe_spatial.combined_shot_type=="Jump Shot"]
kobe_dunk = kobe_spatial[kobe_spatial.combined_shot_type=="Dunk"]
kobe_layup = kobe_spatial[kobe_spatial.combined_shot_type=="Layup"]
kobe_tip = kobe_spatial[kobe_spatial.combined_shot_type=="Tip Shot"]
kobe_hook = kobe_spatial[kobe_spatial.combined_shot_type=="Hook Shot"]
kobe_bank = kobe_spatial[kobe_spatial.combined_shot_type=="Bank Shot"]

kobe_action = kobe_clean[["action_type", "combined_shot_type", "shot_made_flag"]].copy()

# Short Alias for Action type
kobe_action.action_type = kobe_action.action_type.apply(lambda x:x.lower().replace(" shot", ""))


#%%
############### Action Type ##################
####### Action Type Success Rate #######
action_accuracy = kobe_action.groupby("action_type")["shot_made_flag"].agg(["mean", "count"]).sort_values("count", ascending=False)
action_accuracy.columns = ["success rate", "count"]
action_accuracy = action_accuracy.reset_index()
action_accuracy
#%%
#### plot accuracy ####
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x=action_accuracy.index, y="mean", size="count", sizes=(5, 1583), size_norm=(1, 15836),data=action_accuracy, alpha=0.7, color="skyblue")

# add text label for action type with high frequency and high accuracy
for i in [0,1,3,6,8]:
        plt.text(i, action_accuracy["mean"][i], action_accuracy.action_type[i], fontdict=dict(color="blue", alpha=0.5, size=16))
plt.text(2, action_accuracy["mean"][2], "layup", fontdict=dict(color="blue", alpha=0.5, size=16), ha="center")
plt.text(5-0.4, action_accuracy["mean"][5]+0.01, action_accuracy.action_type[5], fontdict=dict(color="blue", alpha=0.5, size=16))

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Shot Accuracy for Each Action Type")
plt.xticks(range(1, 56, 1), rotation=-90)
plt.xlabel("Action Types")
plt.ylabel("Success Rate")
plt.legend().remove()
plt.tight_layout()
plt.show()
#%%

# fig, ax = plt.subplots(figsize=(9, 6))
# draw_court(outer_lines=True)
# sns.scatterplot(x="loc_x", y="loc_y", hue="action_type", style="shot_made_flag", markers={0:"X", 1:"o"}, data=kobe_spatial, alpha=0.7)

# plt.legend(bbox_to_anchor=(1, 1))
# plt.title("Jump Shot")
# plt.ylim(-100,500)
# plt.xlim(300,-300)
# plt.show()
# %%
############# Combined Shot Type ################
####### Combined Shot Type Success Rate #######
combined_accuracy = kobe_action.groupby("combined_shot_type")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
combined_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
combined_accuracy
#%%
#### jump shot #####
jump_accuracy = kobe_jump.groupby("shot_zone_range")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
jump_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
jump_accuracy
#%%
plt.subplots(figsize=(13,11))
draw_court(outer_lines=True)
ax1 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_jump, alpha=0.6)
# sns.scatterplot(x="loc_x", y="loc_y", hue="shot_zone_basic", style="shot_made_flag", markers={0:"X", 1:"o"}, data=kobe_jump, alpha=0.7)
plt.legend(bbox_to_anchor=(1, 1))
plt.title("Jump Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
#### bank shot #####
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax2 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_bank, alpha=0.7)

# plt.legend(bbox_to_anchor=(1, 1))
plt.title("Bank Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
#### hook shot #####
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax3 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_hook, alpha=0.7)

# plt.legend(bbox_to_anchor=(1, 1))
plt.title("Hook Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
#### layup #####
layup_accuracy = kobe_layup.groupby("shot_made_flag").agg(["count"])
# layup_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
layup_accuracy
#%%
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax4 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_layup, alpha=0.5)

# plt.legend(bbox_to_anchor=(1, 1))
plt.title("Layup")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
#### tip ##### 
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_tip, alpha=0.7)

# plt.legend(bbox_to_anchor=(1, 1))
plt.title("Tip Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
#### dunk shot #####
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax5 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_dunk, alpha=0.7)

# plt.legend(bbox_to_anchor=(1, 1))
plt.title("Dunk Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
#%%[markdown]
## Action Type Accuracy
# Dunk shot is the most likely type to succeed.
#
# Bank shot and hook shot are very likely to succeed.

# %%
############# shot_zone_area ################
####### Success Rate #######
area_accuracy = kobe_spatial.groupby("shot_zone_area")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
area_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
area_accuracy
#%%
####### Plot #######
plt.subplots(figsize=(8,8))
draw_court(outer_lines=True)
sns.scatterplot(x="loc_x", y="loc_y", hue="shot_zone_area",style="shot_made_flag", markers={0:"X", 1:"o"}, data=kobe_spatial, alpha=0.7)

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Shot Zone Area")
plt.ylim(-100,600)
plt.xlim(300,-300)
plt.show()
# %%
############# shot_zone_basic ################
####### Success Rate #######
basic_accuracy = kobe_spatial.groupby("shot_zone_basic")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
basic_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
basic_accuracy
#%%
####### Plot #######
plt.subplots(figsize=(8,8))
draw_court(outer_lines=True)
sns.scatterplot(x="loc_x", y="loc_y", hue="shot_zone_basic",style="shot_made_flag", markers={0:"X", 1:"o"}, data=kobe_spatial, alpha=0.7)

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Shot Zone Basic")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
############# shot_zone_range and distance ################
####### Success Rate #######
range_accuracy = kobe_spatial.groupby("shot_zone_range")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
range_accuracy.columns = ["success rate", "count"]
# combined_accuracy = combined_accuracy.reset_index()
range_accuracy
#%%
####### Plot #######
plt.subplots(figsize=(8,8))
draw_court(outer_lines=True)
sns.scatterplot(x="loc_x", y="loc_y", hue="shot_zone_range",style="shot_made_flag", markers={0:"X", 1:"o"}, data=kobe_spatial, alpha=0.7)

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Shot Zone Range")
plt.ylim(-100,900)
plt.xlim(300,-300)
plt.show()
# %%
####### Accuracy vs Distance #######
plt.subplots(figsize=(8,8))
sns.lmplot(x="shot_distance", y="shot_made_flag",data=kobe_spatial[kobe_spatial.shot_distance < 50],y_jitter=0.02, logistic=True, palette="Blue_r")

plt.title("Accuracy vs Distance")
plt.xlim(0,55)
plt.ylabel("Succeed")
plt.xlabel("Distance(ft)")
plt.show()
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import glm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics
from matplotlib.lines import Line2D
#%%
kobe_spa_cp = kobe_spatial[["combined_shot_type", "shot_zone_area", "shot_distance", "shot_made_flag"]].copy()
area_dummy = pd.get_dummies(kobe_spa_cp.shot_zone_area, prefix="area")
type_dummy = pd.get_dummies(kobe_spa_cp.combined_shot_type, prefix="type")
kobe_spa_cp = kobe_spa_cp.join([area_dummy, type_dummy])
kobe_spa_cp.drop(["shot_zone_area", "combined_shot_type"], axis=1, inplace=True)
#%%
###### model ######
model = glm("shot_made_flag~C(shot_zone_area)+shot_distance+C(combined_shot_type)", data=kobe_clean, family=sm.families.Binomial())
modelfit = model.fit()
print(modelfit.summary())
#%%
y = kobe_spa_cp["shot_made_flag"]
x = kobe_spa_cp.drop(["shot_made_flag"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
#%%
model1 = LogisticRegression()
model1.fit(x_train, y_train)
print('Logit model accuracy (with the test set):', model1.score(x_test, y_test))
#%%
# predict
y_pred = pd.Series(model.predict(X_test))
y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
z.head()
#%%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
# %%
