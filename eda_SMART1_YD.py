#%%
# LIBRARY IMPORTS
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc

from scipy import stats as stats

from scipy.stats import ttest_ind
from scipy.stats import f_oneway

print("\nIMPORT SUCCESS.")
#%%
# DATA IMPORTS
kobe_clean = pd.read_csv('data_cleaned.csv')#, header = 0, index_col = 'shot_id')
kobe_clean.info()

print("\nIMPORT SUCCESS.")
#%% 
# draw_court() function from MichaelKrueger's excellent script
# https://www.kaggle.com/bbx396/kobechart
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

print("\nReady to continue.")
#%%
############# How does spacial location affect Kobe's shot accuracy? ##############
# define some useful subsets

# all spatial factors
kobe_spatial = kobe_clean[["loc_x", "loc_y", "action_type", "combined_shot_type", "shot_zone_area", "shot_type", "shot_zone_basic", "shot_zone_range", "shot_distance", "shot_made_flag"]].copy()
# all action types
kobe_action = kobe_clean[["action_type", "combined_shot_type", "shot_made_flag"]].copy()
# Short Alias for Action type
kobe_action.action_type = kobe_action.action_type.apply(lambda x:x.lower().replace(" shot", ""))

# jump shots
kobe_jump = kobe_spatial[kobe_spatial.combined_shot_type=="Jump Shot"].copy()
# dunk shot
kobe_dunk = kobe_spatial[kobe_spatial.combined_shot_type=="Dunk"].copy()
# layup
kobe_layup = kobe_spatial[kobe_spatial.combined_shot_type=="Layup"].copy()
# tip-in
kobe_tip = kobe_spatial[kobe_spatial.combined_shot_type=="Tip Shot"].copy()
# hook shot
kobe_hook = kobe_spatial[kobe_spatial.combined_shot_type=="Hook Shot"].copy()
# bank shot
kobe_bank = kobe_spatial[kobe_spatial.combined_shot_type=="Bank Shot"].copy()

print("\nReady to continue.")
#%%
############### Action Type ##################
####### Action Type Success Rate #######
action_accuracy = kobe_action.groupby("action_type")["shot_made_flag"].agg(["mean", "count"]).sort_values("count", ascending=False)
action_accuracy.columns = ["success rate", "count"]
action_accuracy = action_accuracy.reset_index()
action_accuracy
#%%
####### plot accuracy #######
fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(x=action_accuracy.index, y="success rate", size="count", sizes=(5, 1583), size_norm=(1, 15836),data=action_accuracy, alpha=0.7, color="skyblue")

# add text label for action type with high frequency and high accuracy
for i in [0,1,3,6,8]:
        plt.text(i, action_accuracy["success rate"][i], action_accuracy.action_type[i], fontdict=dict(color="blue", alpha=0.5, size=16))
plt.text(2, action_accuracy["success rate"][2], "layup", fontdict=dict(color="blue", alpha=0.5, size=16), ha="center")
plt.text(5-0.4, action_accuracy["success rate"][5]+0.01, action_accuracy.action_type[5], fontdict=dict(color="blue", alpha=0.5, size=16))

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Shot Accuracy for Each Action Type")
plt.xticks(range(1, 56, 1), rotation=-90)
plt.xlabel("Action Types")
plt.ylabel("Success Rate")
plt.legend().remove()
plt.tight_layout()
plt.show()
# %%
############# Combined Shot Type ################
####### Combined Shot Type Success Rate #######
combined_accuracy = kobe_action.groupby("combined_shot_type")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
combined_accuracy.columns = ["success rate", "count"]
combined_accuracy
#%%
# define the shot sides of the basket
def shot_side(x):
    """Return which side of the basket did kobe shot.
        @param x (pd.Series): row in df
        @return string: {"right side","left side","middle"}
    """
    if x < 0:
        return "leftt side"
    elif x > 0:
        return "right side"
    else: return "middle"
#%%
####### jump shot #######
## shot range accuracy ##
jump_accuracy = kobe_jump.groupby("shot_zone_range")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
jump_accuracy.columns = ["success rate", "count"]
jump_accuracy
#%%
## shot side accuracy ##
kobe_jump["side"] = kobe_jump["loc_x"].apply(lambda x: shot_side(x))
jump_accuracy2 = kobe_jump.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
jump_accuracy2.columns = ["success rate", "count"]
jump_accuracy2
#%%
## F test on shot sides ##
f_oneway(kobe_jump[kobe_jump['shot_zone_area']=="Left Side(L)"]['shot_made_flag'], kobe_jump[kobe_jump['shot_zone_area']=="Left Side Center(LC)"]['shot_made_flag'],kobe_jump[kobe_jump['shot_zone_area']=="Right Side Center(RC)"]['shot_made_flag'],kobe_jump[kobe_jump['shot_zone_area']=="Right Side(R)"]['shot_made_flag'],kobe_jump[kobe_jump['shot_zone_area']=="Center(C)"]['shot_made_flag'],kobe_jump[kobe_jump['shot_zone_area']=="Back Court(BC)"]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(13,11))
draw_court(outer_lines=True)
ax1 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_jump, alpha=0.6)

plt.legend(bbox_to_anchor=(1, 1))
plt.title("Jump Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
####### bank shot #######
## shot side accuracy ##
kobe_bank["side"] = kobe_bank["loc_x"].apply(lambda x: shot_side(x))
bank_accuracy = kobe_bank.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
bank_accuracy.columns = ["success rate", "count"]
bank_accuracy
#%%
## T test on sides ##
ttest_ind(kobe_bank[kobe_bank['loc_x']>0]['shot_made_flag'], kobe_bank[kobe_bank['loc_x']<0]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax2 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_bank, alpha=0.7)

plt.title("Bank Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
####### hook shot #######
## shot side accuracy ##
kobe_hook["side"] = kobe_hook["loc_x"].apply(lambda x: shot_side(x))
hook_accuracy = kobe_hook.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
hook_accuracy.columns = ["success rate", "count"]
hook_accuracy
#%%
## T test on sides ##
ttest_ind(kobe_hook[kobe_hook['loc_x']>0]['shot_made_flag'], kobe_hook[kobe_hook['loc_x']<0]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax3 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_hook, alpha=0.7)

plt.title("Hook Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
####### layup #######
## shot side accuracy ##
kobe_layup["side"] = kobe_layup["loc_x"].apply(lambda x: shot_side(x))
layup_accuracy = kobe_layup.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
layup_accuracy.columns = ["success rate", "count"]
layup_accuracy
#%%
## T test on sides ##
ttest_ind(kobe_layup[kobe_layup['loc_x']>0]['shot_made_flag'], kobe_layup[kobe_layup['loc_x']<0]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax4 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_layup, alpha=0.5)

plt.title("Layup")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
####### tip-in #######
## shot side accuracy ##   
kobe_tip["side"] = kobe_tip["loc_x"].apply(lambda x: shot_side(x))
tip_accuracy = kobe_tip.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
tip_accuracy.columns = ["success rate", "count"]
tip_accuracy
#%%
## T test on sides ##
ttest_ind(kobe_tip[kobe_tip['loc_x']>0]['shot_made_flag'], kobe_tip[kobe_tip['loc_x']<0]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_tip, alpha=0.7)

plt.title("Tip In")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
# %%
####### dunk shot #######
## shot side accuracy ##
kobe_dunk["side"] = kobe_dunk["loc_x"].apply(lambda x: shot_side(x))
dunk_accuracy = kobe_dunk.groupby("side")["shot_made_flag"].agg(["mean","count"]).sort_values("mean", ascending=False)
dunk_accuracy.columns = ["success rate", "count"]
dunk_accuracy
#%%
## F test on sides ##
f_oneway(kobe_dunk[kobe_dunk['loc_x']==0]['shot_made_flag'], kobe_dunk[kobe_dunk['loc_x']>0]['shot_made_flag'], kobe_dunk[kobe_dunk['loc_x']<0]['shot_made_flag'])
#%%
## scatterplot ##
plt.subplots(figsize=(8,6))
draw_court(outer_lines=True)
ax5 = sns.scatterplot(x="loc_x", y="loc_y", hue="shot_made_flag", palette={0:"red",1:"green"}, data=kobe_dunk, alpha=0.7)

plt.title("Dunk Shot")
plt.ylim(-100,500)
plt.xlim(300,-300)
plt.show()
#%%[markdown]
## Shot Type Summary
# Shot side has no significant effect on shot accuracy.  
#
# It is very likely to score when Kobe attempts a dunk shot or a bank shot.  
# %%
############# shot_zone_area ################
####### Success Rate #######
area_accuracy = kobe_spatial.groupby("shot_zone_area")["shot_made_flag"].agg(["mean", "count"]).sort_values("mean", ascending=False)
area_accuracy.columns = ["success rate", "count"]
area_accuracy
#%%
# t test on accuracy in RC and LC
ttest_ind(kobe_spatial[kobe_spatial['shot_zone_area']=="Right Side Center(RC)"]['shot_made_flag'],kobe_spatial[kobe_spatial['shot_zone_area']=="Left Side Center(LC)"]['shot_made_flag'])
#%%
# f test on all areas
f_oneway(kobe_spatial[kobe_spatial['shot_zone_area']=="Left Side(L)"]['shot_made_flag'], kobe_spatial[kobe_spatial['shot_zone_area']=="Left Side Center(LC)"]['shot_made_flag'],kobe_spatial[kobe_spatial['shot_zone_area']=="Right Side Center(RC)"]['shot_made_flag'],kobe_spatial[kobe_spatial['shot_zone_area']=="Right Side(R)"]['shot_made_flag'],kobe_spatial[kobe_spatial['shot_zone_area']=="Center(C)"]['shot_made_flag'],kobe_spatial[kobe_spatial['shot_zone_area']=="Back Court(BC)"]['shot_made_flag']) # no significant result
#%%
####### scatterplot #######
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
basic_accuracy
#%%
# f test on all areas
f_oneway(kobe_spatial[kobe_spatial["shot_zone_basic"]=="Restricted Area"]["shot_made_flag"], kobe_spatial[kobe_spatial["shot_zone_basic"]=="In The Paint (Non-RA)"]["shot_made_flag"],kobe_spatial[kobe_spatial["shot_zone_basic"]=="Mid-Range"]["shot_made_flag"],kobe_spatial[kobe_spatial["shot_zone_basic"]=="Left Corner 3"]["shot_made_flag"],kobe_spatial[kobe_spatial["shot_zone_basic"]=="Right Corner 3"]["shot_made_flag"],kobe_spatial[kobe_spatial["shot_zone_basic"]=="Above the Break 3"]["shot_made_flag"]) # no significant result
#%%
####### scatterplot #######
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
range_accuracy
#%%
####### scatterplot #######
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
sns.lmplot(x="shot_distance", y="shot_made_flag",data=kobe_spatial[kobe_spatial.shot_distance < 50],y_jitter=0.02, logistic=True, palette="Blue_r")

plt.title("Accuracy vs Distance")
plt.xlim(0,55)
plt.ylabel("Succeed")
plt.xlabel("Distance(ft)")
plt.show()
#%%[markdown]
## Shot Zone Areas Summary
# It is clear that the accuracy of Kobe's shots is negatively related with his distance from the hoop. The closer he is to the basket, the more accurate he is. Not surprisingly, every shots shoot from the back court area failed unless some accident happens. Kobe is most accurate when he shoots from the center area.
# %%
###### model ######
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
y_pred = pd.Series(model.predict(x_test))
y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
z.head()
#%%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
# %%