
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 11, 3

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('train.csv')
print("Train Data")
print(train_data.head(3))
print("Test Data")
print(test_data.head(3))

print(train_data.shape)
features = [columns for columns in train_data.columns if columns not in ["Class", "Time"] ]
print(features)


def hackathon_GBC_model(clf, train, features):
    clf.fit(train[features], train["Class"])
    probab_of_predict = clf.predict_proba(train[features])[:,1]
    predict_train = clf.predict(train[features])
    cv_score = cross_val_score(clf, train[features], train["Class"], cv=5, scoring="roc_auc")
    print("----------------------Model performance-----------------------")
    print("Accuracy score: ", accuracy_score(train["Class"].values, predict_train))
    print("AUC: ", roc_auc_score(train["Class"],probab_of_predict) )
    print("CV score: Mean - {}, Max - {}, Min - {}, Std - {}".format(np.mean(cv_score), np.max(cv_score),
                                                                     np.min(cv_score), np.std(cv_score)))

    Relative_Feature_importance = pd.Series(clf.feature_importances_, features).sort_values(ascending=False)
    Relative_Feature_importance.plot(kind='bar', title='Order of Feature Importance')
    plt.ylabel('Feature Importance')
    plt.show()

clf = GradientBoostingClassifier(random_state=15)
print(hackathon_GBC_model(clf, train_data, features))

estimators = [x for x in range(10, 131, 10)]
first_tune = {'n_estimators': estimators}
first_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05,min_samples_split=700,
                                                                 min_samples_leaf=70,
                                                                 max_depth=8,max_features='sqrt',subsample=0.8,
                                                                 random_state=15,
                                                                 ),
                            param_grid=first_tune,scoring='roc_auc', n_jobs=6, iid=False, cv=5)

first_search.fit(train_data[features], train_data["Class"])

print(first_search.grid_scores_ , first_search.best_params_, first_search.best_score_)


min_split = [x for x in range(300,1101,100)]
depth = [x for x in range(5, 15, 1)]
second_tune = {'max_depth':depth, 'min_samples_split':min_split}
second_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05, n_estimators=40,
                                                                  min_samples_split=450,
                                                                  min_samples_leaf=70,
                                                                  max_depth=8,max_features='sqrt',subsample=0.8,
                                                                  random_state=15
                                                                  ),
                             param_grid=second_tune, scoring='roc_auc', n_jobs=6, iid=False, cv=5)

second_search.fit(train_data[features], train_data["Class"])

print(second_search.grid_scores_, second_search.best_params_, second_search.best_score_)


min_sample_leaf = [x for x in range(20, 200, 10)]
third_tune = {'min_samples_leaf':min_sample_leaf}
third_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05,n_estimators=40,
                                                                 min_samples_split=800,
                                                                 min_samples_leaf=50,
                                                                 max_depth=11,max_features='sqrt', subsample=0.8,
                                                                 random_state=15,
                                                                 ),
                            param_grid=third_tune, scoring='roc_auc', n_jobs=6, iid=False, cv=5)

third_search.fit(train_data[features], train_data["Class"])

print(third_search.grid_scores_, third_search.best_params_, third_search.best_score_)

clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=40, min_samples_split=800, min_samples_leaf=170,
                                 max_depth=11, random_state=15, max_features='sqrt', subsample=0.8)
print(hackathon_GBC_model(clf, train_data, features))

max_feat = [x for x in range(10, 29, 2)]
fourth_tune = {'max_features':max_feat}
fourth_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05,n_estimators=40,
                                                                  min_samples_split=800,
                                                                  min_samples_leaf=170,
                                                                  max_depth=11,max_features='sqrt', subsample=0.8,
                                                                  random_state=15,
                                                                 ),
                             param_grid=fourth_tune, scoring='roc_auc', n_jobs=6, iid=False, cv=5)

fourth_search.fit(train_data[features], train_data["Class"])

print(fourth_search.grid_scores_, fourth_search.best_params_, fourth_search.best_score_)

sub_sample = [0.5, 0.55, 0.6, 0.65, 0.7]
fifth_tune = {'subsample': sub_sample}
fifth_search = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05,n_estimators=40,
                                                                 min_samples_split=800,
                                                                 min_samples_leaf=170,
                                                                 max_depth=11,max_features=12, subsample=0.8,
                                                                 random_state=15,
                                                                 ),
                            param_grid=fifth_tune, scoring='roc_auc', n_jobs=6, iid=False, cv=5)

fifth_search.fit(train_data[features], train_data["Class"])

print(fifth_search.grid_scores_, fifth_search.best_params_, fifth_search.best_score_)

clf = GradientBoostingClassifier(learning_rate=0.05, n_estimators=40, min_samples_split=800, min_samples_leaf=170,
                                 max_depth=11, random_state=15, max_features=12, subsample=0.55)
print(hackathon_GBC_model(clf, train_data, features))
