#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%
df = pd.read_csv("data_cleaned.csv")

# %%

numerical_features = ['loc_x', 'loc_y', 'minutes_remaining', 'period', 'shot_distance']
categorical_features = ['combined_shot_type','shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent']
target = 'shot_made_flag'

df = df[numerical_features + categorical_features + [target]]

num_train = 20000
num_test = len(df) - num_train

df_train = df[:num_train]
df_test = df[num_train:]

# %%
import statsmodels.api as sm
from statsmodels.formula.api import glm

formula = target + ' ~ ' + ' + '.join(numerical_features) + ' + ' + ' + '.join(['C('+f+')' for f in categorical_features])
lr = glm(formula=formula, data=df_train, family=sm.families.Binomial())
lr_fit = lr.fit()
lr_fit.summary()

#%%
# get importance
feature_importance = pd.DataFrame({'feature': lr_fit.params.index, 'coefficient': lr_fit.params.values})
top_features = feature_importance.sort_values(by=['coefficient'], ascending=False).head(10)
sns.barplot(x='coefficient', y = 'feature', data = top_features)
# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score
Y_pred = np.where(lr_fit.predict(df_test) > 0.5, 1, 0)
Y_test = df_test[target]
print('Accuracy:', accuracy_score(Y_test, Y_pred))
print('Precision:', precision_score(Y_test, Y_pred))
print('Recall:', recall_score(Y_test, Y_pred))


# %%
from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, Y_pred)
# %%
