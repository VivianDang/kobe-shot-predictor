#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#%%
df = pd.read_csv("data_cleaned.csv")

# %%

numerical_features = ['lat', 'lon','loc_x', 'loc_y', 'minutes_remaining','shot_distance', 'secondsFromGameStart']
categorical_features = ['combined_shot_type', 'shot_zone_area', 'shot_zone_basic','playoffs', 'period', 'shot_zone_range','game_month','home', 'opponent']
target = 'shot_made_flag'
X = df[categorical_features + numerical_features]
Y = df[target]
#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_categorical_one_hot = pd.get_dummies(df[categorical_features])
X = pd.concat([df[numerical_features], X_categorical_one_hot], axis=1)
Y = df[target]



#%%
# Feature selection
from sklearn.feature_selection import SelectFdr,SelectKBest, f_classif, chi2
selector = SelectKBest(f_classif, k=20)
selector.fit(X,Y)
selected_features = list(X.columns[selector.get_support()])
X = X[selected_features]

#%%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# %%
from sklearn.linear_model import LogisticRegression

# fit model
lr = LogisticRegression()
lr.fit(X_train, Y_train)
# get importance
feature_importance = pd.DataFrame({'feature': X.columns, 'coefficient': lr.coef_[0], 'abs_coef': abs(lr.coef_[0])})
top_features = feature_importance.sort_values(by=['abs_coef'], ascending=False).head(10)
sns.barplot(x='coefficient', y = 'feature', data = top_features)
# get scores
from sklearn.metrics import accuracy_score, precision_score, recall_score
Y_pred = lr.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, Y_pred))
print('Precision:', precision_score(Y_test, Y_pred))
print('Recall:', recall_score(Y_test, Y_pred))
# get confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

# %%
# %%
from sklearn.svm import LinearSVC

# fit model
svc = LinearSVC()
svc.fit(X_train, Y_train)
# get importance
feature_importance = pd.DataFrame({'feature': X.columns, 'coefficient': svc.coef_[0], 'abs_coef': abs(svc.coef_[0])})
top_features = feature_importance.sort_values(by=['abs_coef'], ascending=False).head(10)
sns.barplot(x='coefficient', y = 'feature', data = top_features)
# get scores
from sklearn.metrics import accuracy_score, precision_score, recall_score
Y_pred = svc.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, Y_pred))
print('Precision:', precision_score(Y_test, Y_pred))
print('Recall:', recall_score(Y_test, Y_pred))
# get confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

# %%
from sklearn.tree import DecisionTreeClassifier

# fit model
tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train, Y_train)
# get importance
feature_importance = pd.DataFrame({'feature': X.columns, 'gini index': tree.feature_importances_})
top_features = feature_importance.sort_values(by=['gini index'], ascending=False).head(10)
sns.barplot(x='gini index', y = 'feature', data = top_features)
# get scores
from sklearn.metrics import accuracy_score, precision_score, recall_score
Y_pred = tree.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, Y_pred))
print('Precision:', precision_score(Y_test, Y_pred))
print('Recall:', recall_score(Y_test, Y_pred))
# get confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

# %%
from catboost import CatBoostClassifier
categorical_features_indices = np.arange(len(categorical_features), len(X.columns))
# fit model
catboost = CatBoostClassifier(iterations = 300, depth = 9, learning_rate = 0.001, cat_features = categorical_features_indices)
catboost.fit(X_train, Y_train)
# get importance
feature_importance = pd.DataFrame({'feature': X.columns, 'prediction value change': catboost.get_feature_importance()})
top_features = feature_importance.sort_values(by=['prediction value change'], ascending=False).head(10)
sns.barplot(x='prediction value change', y = 'feature', data = top_features)
# get scores
from sklearn.metrics import accuracy_score, precision_score, recall_score
Y_pred = catboost.predict(X_test)
print('Accuracy:', accuracy_score(Y_test, Y_pred))
print('Precision:', precision_score(Y_test, Y_pred))
print('Recall:', recall_score(Y_test, Y_pred))
# get confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, Y_pred))

# %%

# %%

# %%

# %%
