import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier




train= pd.read_csv("/users/krutarthmajithia/downloads/train_driver.csv")
test= pd.read_csv("/users/krutarthmajithia/downloads/test_driver.csv")



# Preprocessing 
id_test = test['id'].values
target_train = train['target'].values

train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(col_to_drop, axis=1)  
test = test.drop(col_to_drop, axis=1)  


train = train.replace(-1, np.nan)
test = test.replace(-1, np.nan)


cat_features = [a for a in train.columns if a.endswith('cat')]

for column in cat_features:
	temp = pd.get_dummies(pd.Series(train[column]))
	train = pd.concat([train,temp],axis=1)
	train = train.drop([column],axis=1)
    
for column in cat_features:
	temp = pd.get_dummies(pd.Series(test[column]))
	test = pd.concat([test,temp],axis=1)
	test = test.drop([column],axis=1)


print(train.values.shape, test.values.shape)



class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):

            S_test_i = np.zeros((T.shape[0], self.n_splits))

            for j, (train_idx, test_idx) in enumerate(folds):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
#                y_holdout = y[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
#                cross_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
#                print("    cross_score: %.5f" % (cross_score.mean()))
                y_pred = clf.predict_proba(X_holdout)[:,1]                

                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict_proba(T)[:,1]
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict_proba(S_test)[:,1]
        return res
    
xgb_params = {}
xgb_params['objective'] = 'binary:logistic'
xgb_params['learning_rate'] = 0.04
xgb_params['n_estimators'] = 490
xgb_params['max_depth'] = 4
xgb_params['subsample'] = 0.9
xgb_params['colsample_bytree'] = 0.9  
xgb_params['min_child_weight'] = 10


# RandomForest params
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30


xgb_model = XGBClassifier(**xgb_params)
rf_model = RandomForestClassifier(**rf_params)
gb_model = GradientBoostingClassifier(max_depth=5)
log_model = LogisticRegression()

stack = Ensemble(n_splits=3,
        stacker = log_model,
        base_models = (xgb_model, gb_model, rf_model))        
        
y_pred = stack.fit_predict(train, target_train, test)        
