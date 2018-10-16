# -*- coding: utf-8 -*-

# automated parameter tuning on lightGBM
def auto_lgbclassifier(train,target,max_iter):
    from hyperopt import hp
    from hyperopt.pyll.stochastic import sample
    from sklearn.model_selection import cross_val_score
    import pandas as pd
    import numpy as np
    import pickle
    import os
    from hyperopt import tpe
    import lightgbm as lgb
    import warnings
    import copy
    train_set = lgb.Dataset(train, label = target)
    # set space
    space = {"objective":"binary",
             "boosting_type":"gbdt",
             'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
             'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.2)),
             'max_depth':hp.quniform('max_depth',1,10,1),
             'bagging_fraction':hp.uniform('bagging_fraction', 0.6, 1.0),
             'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
             'feature_fraction':hp.uniform('feature_fraction', 0.6, 1.0),
             'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
             'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0)}
    space_ini={'boosting_type': 'gbdt','objective': 'binary','learning_rate': 0.05,'num_leaves': 50,'max_depth': 6,'subsample': 0.8,'colsample_bytree': 0.8}
    best_score=0
    iteration=0
    cv_results = lgb.cv(space_ini, train_set,num_boost_round=10000,early_stopping_rounds=20,nfold=10,metrics="l1")
    n_estimators=len(cv_results["l1-mean"]) 
    space["n_estimators"]=n_estimators
    best_score=0
    while iteration<max_iter:
        space0=sample(space)
        space0["max_depth"]=int(space0["max_depth"])
        space0["num_leaves"]=int(space0["num_leaves"])
        space0["min_child_samples"]=int(space0["min_child_samples"])
        model=lgb.LGBMClassifier()
        model.learning_rate=0.05
        model.max_depth=space0["max_depth"]
        model.num_leaves=space0["num_leaves"]
        model.subsample=space0["bagging_fraction"]
        model.colsample_bytree=space0["feature_fraction"]
        model.reg_alpha=space0["reg_alpha"]
        model.reg_lambda=space0["reg_lambda"]
        new_score=np.mean(cross_val_score(model,train,target,scoring="accuracy",cv=10))
        if new_score>best_score:
            best_score=new_score
            final_model=copy.copy(model)
            print("The current iteration is "+ str(iteration)+'  '+'with the score  '+str(best_score))
        iteration=iteration+1
    return final_model

