import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from matplotlib import style
import matplotlib.pyplot as plt
from scipy.stats import uniform
from random import uniform as rndflt
from scipy.stats import randint
from matplotlib.artist import setp
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_predict,StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix,recall_score,f1_score,accuracy_score,precision_score
from sklearn.metrics import make_scorer, average_precision_score

df = pd.read_csv('creditcard.csv')
#print(df.isnull().sum())  #no missing values
#print(df['Class'].value_counts())

####################3##################identify categorical columns, if any############################
# for attribute in df.columns.tolist():
#     print('Unique values for {0} : {1}'.format(attribute, df[attribute].nunique()))

####################################### Sample a test set, put it aside, and never look at it##########
''' IMPORTANT!
We do not want to separate the X(inputs) from the y(label/target) before splitting into
test and train data if we are planning on having a stratfied split. This is becuase the 
attribute we're planning on stratify splitting on, will be missing from either X or y,
in this case y, becuase we want a balance of all wine qualities in both train and test 
data '''
train_set, test_set = train_test_split(df, test_size=0.3, shuffle=True, stratify=df['Class'])

######################################Correlation Heatmap Visualization####################################
# sns.set(font_scale=0.5)
# correlation = train_set.corr()
# fig, ax = plt.subplots(1, 1)
# ax.set_facecolor('#F0F0F0')
# fig = plt.gcf()
# fig.set_size_inches(14, 10, forward=True)
# fig.patch.set_facecolor('#F0F0F0')
# # cmap = sns.palplot(sns.diverging_palette(220, 20, n=7))
# heatmap = sns.heatmap(correlation, annot=True, linewidths=1, linecolor='#F0F0F0', vmin=-0.9, vmax=1, cmap='BrBG')
# plt.show()

#######################################FEATURE SELECTION####################################################
'''We will use one of 2 methods, or both  For classifcation: Method 1, Correlation Method. Method 2. 
Univariate method using SelectKBest. '''
#Method1 was already carried out
#drop features based on mutual_info_class algo below
train_set = train_set.drop(['V19','V20','V23','V24','V26','V22','V13','V25','V15'],1)
test_set = test_set.drop(['V19','V20','V23','V24','V26','V22','V13','V25','V15'],1)

#Good idea to split into X and y now as we'll only want study features vs target/feature here
X_train_set = train_set.drop(['Class'],1)
y_train_set = train_set['Class']
X_test_set = test_set.drop(['Class'],1)
y_test_set = test_set['Class']

#for scorng, we have the option between,chi2, f_classif, mutual_info_classif
#ANOVA will be used(explaned in detail on website, why)
'''mutual_info_classif'''
# bestfeatures = SelectKBest(score_func=mutual_info_classif, k=30)
# fit = bestfeatures.fit(X_train_set,y_train_set)
# df_scores = pd.DataFrame(fit.scores_)
# df_columns = pd.DataFrame(X_train_set.columns)
# # concatenate dataframes
# feature_scores = pd.concat([df_columns, df_scores],axis=1)
# feature_scores.columns = ['Feature_Name','Score']  # name output columns
# print(feature_scores.nlargest(30,'Score'))  # print 30 best features


########################################FEATURE SCALING##############################################
#because of the imbalance, the best scaling method to use is either RobustScaler or QuantileTransformer
scaler = RobustScaler()
X_train_set = scaler.fit_transform(X_train_set)
X_test_set = scaler.fit_transform(X_test_set)

#############################SHORTLISITNG AND COMPARING METHODS#####################################

'''for this project, we are exploring LogisticRegression only'''

#################################################SCORING METHOD#######################################
'''THIS IS A COST-SENSITIVE CLASSIFICATION PROBLEM'''
#LogisticRegression class provides a weighing option to help with:
'''1.there is a large class imbalance'''
'''2.we care more about False Negatives, it's best to use the PR curve over the ROC curve'''
#this is how we would make our own:
# lgr_clf = LogisticRegression()
# y_scores = cross_val_predict(lgr_clf, X_train_set, y_train_set, cv=10,method="decision_function")
# score_func = average_precision_score(y_train_set, y_score=y_scores, average='weighted')
# custom_scorer = make_scorer(score_func)
'''HOWEVER, we have average precision, equal to taking the area under the curve'''

##########################################3#############PLOTS#####################################################3


######################################FINE TUNING LOGISTIC REGRESSION HYPERPARAMETERS#########################################
'''A DECENT UNDERSTANDING OF ALL PARAMETERS IS REQUIRED HERE'''
#penalty : l2 . why?page203

lgr_clf = LogisticRegression(penalty='l2', dual=False, solver='sag', n_jobs=-1)
param_distributions = {'tol':[5e-5, 1e-4, 5e-4], 'C':[0.8, 1, 2], 'fit_intercept':[True, False],
'class_weight':[{0:0.05, 1:0.95}, {0:0.1, 1:0.9}, {0:0.2, 1:0.8}, {0:0.25, 1:0.75}, {0:0.3, 1:0.7}, 
{0:0.4, 1:0.6},{0:0.5, 1:0.5},{0:0.6, 1:0.4},{0:0.7, 1:0.3}], 'max_iter': [100, 200, 300]}

'''RANDOMSEARCH'''
# def hypertuning_rscv(clf, p_distr, nbr_iter,X,y):
#     rdmsearch = RandomizedSearchCV(clf, param_distributions=p_distr,
#                                   n_jobs=-1, n_iter=nbr_iter, scoring='average_precision', cv=StratifiedShuffleSplit(n_splits=10))
#     #CV = Cross-Validation ( here using Stratified KFold CV)
#     rdmsearch.fit(X,y)
#     ht_params = rdmsearch.best_params_
#     ht_score = rdmsearch.best_score_
#     return ht_params, ht_score

# rf_parameters, rf_ht_score = hypertuning_rscv(lgr_clf, param_distributions, 40, X_train_set, y_train_set)

# print(rf_parameters)
# print(rf_ht_score)
'''RESULTS'''
# {'tol': 0.0001, 'max_iter': 300, 'fit_intercept': True, 'class_weight': {0: 0.4, 1: 0.6}, 'C': 2}
# 0.936267821601993

'''GRIDSEARCH'''
# def hypertuning_rscv(clf, p_distr, nbr_iter,X,y):
#     rdmsearch = GridSearchCV(clf, param_grid=p_distr,
#                                   n_jobs=-1, scoring='average_precision', cv=StratifiedShuffleSplit(n_splits=10))
#     #CV = Cross-Validation ( here using Stratified KFold CV)
#     rdmsearch.fit(X,y)
#     ht_params = rdmsearch.best_params_
#     ht_score = rdmsearch.best_score_
#     return ht_params, ht_score

# rf_parameters, rf_ht_score = hypertuning_rscv(lgr_clf, param_distributions, 40, X_train_set, y_train_set)

# print(rf_parameters)
# print(rf_ht_score)

'''RESULTS'''
# {'C': 0.8, 'class_weight': {0: 0.2, 1: 0.8}, 'fit_intercept': True, 'max_iter': 300, 'tol': 5e-05}
# 0.9448304076749737


#########################################BEST CLASSIFIER##############################################
lgr_clf = LogisticRegression(penalty='l2', dual=False, tol=5e-05, C=0.8, fit_intercept=True,
class_weight={0: 0.2, 1: 0.8}, solver='sag', max_iter=300, n_jobs=-1)

lgr_clf.fit(X_train_set,y_train_set)
y_pred = lgr_clf.predict(X_test_set)
print(classification_report(y_test_set, y_pred))

###########################################PR CURVE#############################################
##y_scores = cross_val_predict(lgr_clf, X_test_set, y_test_set, cv=10,method="decision_function")
# y_scores = lgr_clf.decision_function(X_test_set)
# average_precision = average_precision_score(y_test_set, y_score=y_scores, average='weighted')
# disp = plot_precision_recall_curve(lgr_clf, X_test_set, y_test_set)
# disp.ax_.set_title('2-class Precision-Recall curve: '
#                    'AP={0:0.2f}'.format(average_precision))
# plt.show()

########################################CONFUSSION MATRIX#######################################

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(lgr_clf, X_test_set, y_test_set,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()
###########################################################################################################