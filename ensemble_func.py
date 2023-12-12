# module used
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

# ensemble functions
## ps
# PS function
def ps_train_ensemble(data,Train,Test,features,outcomes,
                      include_lg=False, include_el=True, include_xgb=True, include_mlp=False,
                      include_l1=False, include_svm=False, include_nb=True,include_rf=False):
    import warnings
    warnings.filterwarnings('ignore')
    # Train,Test = train_test_split(data,test_size=0.4,random_state=42)
    np.random.seed(233)
    models = {}
    # base learner 1
    if include_lg:
        lg = linear_model.LogisticRegression()
        lg.fit(data[features],data[outcomes])
        models['logit'] = lg

    # base learner 2
    if include_el:
        el_class = linear_model.SGDClassifier(penalty='elasticnet',loss='log')
        param_grid_el = {'alpha': [0.0001, 0.001, 0.01, 0.1]}
        gs_el = GridSearchCV(estimator=el_class,param_grid=param_grid_el,cv=5,scoring='neg_log_loss')
        gs_el.fit(Train[features],Train[outcomes])
        el_class.set_params(**gs_el.best_params_)
        el_class.fit(Test[features],Test[outcomes])
        models['enet'] = el_class

    # base learner 3
    if include_xgb:
        param_grid_G = {
        'learning_rate': [0.01, 0.1, 1],
         'max_depth': [55,65,80],
        # 'n_estimators': [100,200,300],
        # 'alpha':[0.05,0.1,0.18],
        # 'gamma': [1,1.5,1.8]
        }
        xgb_cls = xgb.XGBClassifier(objective='binary:logistic')
        gs_xgb = GridSearchCV(
            estimator=xgb_cls,
            param_grid=param_grid_G,
            cv=5,
            scoring='neg_log_loss'
        )
        gs_xgb.fit(Train[features], Train[outcomes])
        xgb_cls.set_params(**gs_xgb.best_params_)
        xgb_cls.fit(Test[features], Test[outcomes])
        models['xgb'] = xgb_cls

    # base learner 4
    if include_mlp:
        mlp_cls = MLPClassifier()
        param_grid_mlpc = {
            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
        }
        gs_mlp = GridSearchCV(estimator=mlp_cls,param_grid=param_grid_mlpc,cv=5,scoring="neg_log_loss")
        gs_mlp.fit(Train[features],Train[outcomes])
        mlp_cls.set_params(**gs_mlp.best_params_)
        mlp_cls.fit(Test[features],Test[outcomes])
        models['mlp'] = mlp_cls

    if include_l1:
        l1_cls = linear_model.LogisticRegression(penalty='l1',solver='saga')
        param_grid_l1 = {
        'C':[0.1,1,5],
        #'max_features': [None, 'sqrt', 'log2']
        }
        gs_l1 = GridSearchCV(estimator=l1_cls,param_grid=param_grid_l1,cv=5,scoring="neg_log_loss")
        gs_l1.fit(Train[features],Train[outcomes])
        l1_cls.set_params(**gs_l1.best_params_)
        l1_cls.fit(Test[features],Test[outcomes])
        models['gbr'] = l1_cls

    if include_svm:
        svm_c = SVC(kernel='linear',probability=True)
        gs_svm = GridSearchCV(estimator=svm_c,param_grid={'C': [0.1,1,10,100]},cv=5,scoring='neg_log_loss')
        gs_svm.fit(Train[features],Train[outcomes])
        svm_c.set_params(**gs_svm.best_params_)
        svm_c.fit(Test[features],Test[outcomes])
        models['svm'] = svm_c
        
    if include_nb:
        nb_c = GaussianNB()
        param_grid_nb = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
            }
        gs_nb = GridSearchCV(nb_c, param_grid=param_grid_nb, cv=5)
        gs_nb.fit(Train[features],Train[outcomes])
        nb_c.set_params(**gs_nb.best_params_)
        nb_c.fit(Test[features],Test[outcomes])
        models['nb'] = nb_c

    if include_rf:
        rf_c = RandomForestClassifier(random_state = 42)
        param_grid_rf = {
            'n_estimators':[100,200],
            'max_depth':[40,50],
            'min_samples_split':[2,5],
            }
        gs_rf = GridSearchCV(rf_c, param_grid=param_grid_rf, cv=5, scoring='neg_log_loss')
        gs_rf.fit(Train[features],Train[outcomes])
        rf_c.set_params(**gs_rf.best_params_)
        rf_c.fit(Test[features],Test[outcomes])
        models['rf'] = rf_c

    # meta learner
    Q1fit = []
    if include_lg:
        Q1fit.append(lg.predict_proba(data[features])[:,1].ravel())
    if include_el:
        Q1fit.append(el_class.predict_proba(data[features])[:,1].ravel())
    if include_xgb:
        Q1fit.append(xgb_cls.predict_proba(data[features])[:,1].ravel())
    if include_mlp:
        Q1fit.append(mlp_cls.predict_proba(data[features])[:,1].ravel())
    if include_l1:
        Q1fit.append(l1_cls.predict_proba(data[features])[:,1].ravel())
    if include_svm:
        Q1fit.append(svm_c.predict_proba(data[features])[:,1].ravel())
    if include_nb:
        Q1fit.append(nb_c.predict_proba(data[features])[:,1].ravel())
    if include_rf:
        Q1fit.append(rf_c.predict_proba(data[features])[:,1].ravel())

    cls_stack = linear_model.LogisticRegression()
    cls_stack.fit(np.transpose(Q1fit),data[outcomes])

    return models, cls_stack

def ps_function(data,features,outcomes, lg=False, el=True, xgb=True, mlp=False,
                l1=False, svm=False, nb=True,rf=False):
    import warnings
    warnings.filterwarnings('ignore')
    Train,Test = train_test_split(data,test_size=0.4,random_state=42)
    model_tr,stack_tr = ps_train_ensemble(data,Train,Test,features,outcomes, include_lg=lg, include_el=el, include_xgb=xgb, include_mlp=mlp,
                                          include_l1=l1, include_svm=svm, include_nb=nb,include_rf=rf)
    model_ts,stack_ts = ps_train_ensemble(data,Test,Train,features,outcomes, include_lg=lg, include_el=el, include_xgb=xgb, include_mlp=mlp,
                                          include_l1=l1, include_svm=svm, include_nb=nb,include_rf=rf)
    return model_tr,model_ts,stack_tr,stack_ts

def ps_predict(preddat,features,model_tr,model_ts,stack_tr,stack_ts):
    y_pred_tr = {}
    for name, model in model_tr.items():
        y_pred_tr[name] = model.predict_proba(preddat[features])[:,1].ravel()
    Q_base_tr = np.transpose(np.array(list(y_pred_tr.values())))
    eta_tr = stack_tr.predict_proba(Q_base_tr)[:,1]

    y_pred_ts = {}
    for name, model in model_ts.items():
        y_pred_ts[name] = model.predict_proba(preddat[features])[:,1].ravel()
    Q_base_ts = np.transpose(np.array(list(y_pred_ts.values())))
    eta_ts = stack_ts.predict_proba(Q_base_ts)[:,1]

    return (eta_tr+eta_ts)/2
    


## Gcomputation
# Gcomputation function
def gcompu_train_ensemble(data,Train,Test,features,outcomes,
                          include_lm = True, include_rg = False, 
                          include_xgb = True, include_mlp = False,
                          include_gbr = True):
    import warnings
    warnings.filterwarnings('ignore')
    # Train,Test = train_test_split(data,test_size=0.4,random_state=42)
    np.random.seed(233)
    models = {}
    # base learner 1
    if include_lm:
        lm_1 = linear_model.LinearRegression()
        lm_1.fit(data[features],data[outcomes])
        models['lm'] = lm_1

    # base learner 2
    if include_rg:
        rg_lm = linear_model.Ridge()
        param_grid_rg = {'alpha': [0.1, 1, 10]}
        gs_rg = GridSearchCV(estimator=rg_lm,param_grid=param_grid_rg,cv=5,scoring="neg_mean_absolute_error")
        gs_rg.fit(Train[features],Train[outcomes])
        rg_lm.set_params(**gs_rg.best_params_)
        rg_lm.fit(Test[features],Test[outcomes])
        models['rg'] = rg_lm

    # base learner 3
    if include_xgb:
        param_grid_G = {
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [15,20,25,35],
        # 'n_estimators': [100, 150,200],
        #'alpha':[0.05,0.1,0.18],
        #'gamma': [1,1.5,1.8]
        }
        xgb_lm1_tr = xgb.XGBRegressor()
        gs_gcomp1_tr = GridSearchCV(
            estimator=xgb_lm1_tr,
            param_grid=param_grid_G,
            cv=5,
            scoring='neg_mean_absolute_error'
        )
        gs_gcomp1_tr.fit(Train[features], Train[outcomes])
        xgb_lm1_tr.set_params(**gs_gcomp1_tr.best_params_)
        xgb_lm1_tr.fit(Test[features], Test[outcomes])
        models['xgb'] = xgb_lm1_tr

    # base learner 4
    if include_mlp:
        mlp_lm = MLPRegressor()
        param_grid_mlpr = {
            'hidden_layer_sizes': [ (100,), (150,), (100, 100), (150,150)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
        }
        gs_mlp_lm = GridSearchCV(estimator=mlp_lm,param_grid=param_grid_mlpr,cv=5,scoring="neg_mean_absolute_error")
        gs_mlp_lm.fit(Train[features],Train[outcomes])
        mlp_lm.set_params(**gs_mlp_lm.best_params_)
        mlp_lm.fit(Test[features],Test[outcomes])
        models['mlp'] = mlp_lm

    if include_gbr:
        gbr_lm = GradientBoostingRegressor()
        param_grid_gbr = {
        # 'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [15,20,25,35],
        #'min_samples_split': [2, 3, 4],
        #'min_samples_leaf': [1, 2, 3],
        #'max_features': [None, 'sqrt', 'log2']
        }
        gs_gbr_lm = GridSearchCV(estimator=gbr_lm,param_grid=param_grid_gbr,cv=5,scoring="neg_mean_absolute_error")
        gs_gbr_lm.fit(Train[features],Train[outcomes])
        gbr_lm.set_params(**gs_gbr_lm.best_params_)
        gbr_lm.fit(Test[features],Test[outcomes])
        models['gbr'] = gbr_lm

    # meta learner
    Q1fit = []
    if include_lm:
        Q1fit.append(lm_1.predict(data[features]).ravel())
    if include_rg:
        Q1fit.append(rg_lm.predict(data[features]).ravel())
    if include_xgb:
        Q1fit.append(xgb_lm1_tr.predict(data[features]).ravel())
    if include_mlp:
        Q1fit.append(mlp_lm.predict(data[features]).ravel())
    if include_gbr:
        Q1fit.append(gbr_lm.predict(data[features]).ravel())

    lm_stack = linear_model.LinearRegression()
    lm_stack.fit(np.transpose(Q1fit),data[outcomes])

    return models, lm_stack

def gcompu_function(data,features,outcomes,lm = True, rg = False, 
                    xgb = True, mlp = False,gbr = True):
    import warnings
    warnings.filterwarnings('ignore')
    Train,Test = train_test_split(data,test_size=0.4,random_state=42)
    model_tr,stack_tr = gcompu_train_ensemble(data,Train,Test,features,outcomes,
                                              include_lm = lm, include_rg = rg, 
                                              include_xgb = xgb, include_mlp = mlp,include_gbr = gbr)
    model_ts,stack_ts = gcompu_train_ensemble(data,Test,Train,features,outcomes,
                                              include_lm = lm, include_rg = rg, 
                                              include_xgb = xgb, include_mlp = mlp,include_gbr = gbr)
    return model_tr,model_ts,stack_tr,stack_ts


def gcompu_predict(preddat,features,model_tr,model_ts,stack_tr,stack_ts):
    y_pred_tr = {}
    for name, model in model_tr.items():
        y_pred_tr[name] = model.predict(preddat[features]).ravel()
    Q_base_tr = np.transpose(np.array(list(y_pred_tr.values())))
    eta_tr = stack_tr.predict(Q_base_tr)
    y_pred_ts = {}
    for name, model in model_ts.items():
        y_pred_ts[name] = model.predict(preddat[features]).ravel()
    Q_base_ts = np.transpose(np.array(list(y_pred_ts.values())))
    eta_ts = stack_ts.predict(Q_base_ts)

    return (eta_tr+eta_ts)/2



