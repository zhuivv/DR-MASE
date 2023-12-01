# module used
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import linear_model
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
import sklearn.metrics as metrics
from sklearn.ensemble import StackingRegressor, StackingClassifier
import xgboost as xgb
import pandas as pd
import multiprocess
from scipy import linalg, special
import statsmodels.formula.api as smf
import statistics as stat
import time
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.naive_bayes import GaussianNB

# ensemble functions
## ps
def ps_train_ensemble(data,features,outcomes,stseed,seed,
                          include_lg = False, include_el = True, 
                          include_xgb = True, include_mlp = False,
                          include_l1 = True, include_svm = False,
                          include_nb = True):
    import warnings
    warnings.filterwarnings('ignore')
    Train,Test = train_test_split(data,test_size=0.4,random_state=stseed)
    np.random.seed(seed)
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
         'max_depth': [15,5,10],
        # 'n_estimators': [100, 200, 300],
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

    cls_stack = linear_model.LogisticRegression()
    cls_stack.fit(np.transpose(Q1fit),data[outcomes])

    return models, cls_stack

    


## Gcomputation
def gcompu_train_ensemble(data,features,outcomes,
                          include_lm = True, include_rg = False, 
                          include_xgb = True, include_mlp = False,
                          include_gbr = True,include_l1= True):
    import warnings
    warnings.filterwarnings('ignore')
    Train,Test = train_test_split(data,test_size=0.4,random_state=42)
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
        'max_depth': [5,15,10],
        #'n_estimators': [100, 200, 300],
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

    # base learner 5
    if include_gbr:
        gbr_lm = GradientBoostingRegressor()
        param_grid_gbr = {
        #'n_estimators': [100, 200, 300],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [5,10,15],
        #'min_samples_split': [2, 3, 4],
        #'min_samples_leaf': [1, 2, 3],
        #'max_features': [None, 'sqrt', 'log2']
        }
        gs_gbr_lm = GridSearchCV(estimator=gbr_lm,param_grid=param_grid_gbr,cv=5,scoring="neg_mean_absolute_error")
        gs_gbr_lm.fit(Train[features],Train[outcomes])
        gbr_lm.set_params(**gs_gbr_lm.best_params_)
        gbr_lm.fit(Test[features],Test[outcomes])
        models['gbr'] = gbr_lm

    # base learner 6
    if include_l1:
        l1_lm = linear_model.Lasso()
        param_grid_l1 = {'alpha': [0.1, 0.5, 1.0, 2.0],}
        gs_l1_lm = GridSearchCV(estimator=l1_lm,param_grid=param_grid_l1,cv=5,scoring="neg_mean_absolute_error")
        gs_l1_lm.fit(Train[features],Train[outcomes])
        l1_lm.set_params(**gs_l1_lm.best_params_)
        l1_lm.fit(Test[features],Test[outcomes])
        models['l1'] = l1_lm

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
    if include_l1:
        Q1fit.append(l1_lm.predict(data[features]).ravel())

    lm_stack = linear_model.LinearRegression()
    lm_stack.fit(np.transpose(Q1fit),data[outcomes])

    return models, lm_stack

# MSM with ensemble
ps0feature = np.concatenate([covbase,covtp1])
baseps0, cls_stack0 = ps_train_ensemble(abcd_encoded,ps0feature,trttp1,30,233,include_nb = False,include_mlp = True)
ps_pred0 = {}
for name, model in baseps0.items():
    ps_pred0[name] = model.predict_proba(abcd_encoded[ps0feature])[:,1].ravel()
ps_base0 = np.transpose(np.array(list(ps_pred0.values())))
enps0 = cls_stack0.predict_proba(ps_base0)[:,1]
ps1feature = np.concatenate([covbase,covtp1,covtp2, output1,trttp1])
baseps1, cls_stack1 = ps_train_ensemble(abcd_encoded,ps1feature,trttp2,32,233,include_l1=True,include_mlp=True)
ps_pred1 = {}
for name, model in baseps1.items():
    ps_pred1[name] = model.predict_proba(abcd_encoded[ps1feature])[:,1].ravel()
ps_base1 = np.transpose(np.array(list(ps_pred1.values())))
enps1 = cls_stack1.predict_proba(ps_base1)[:,1]

# estimation funciton
def dr_abcd_boot(out1,out2,trt1,trt2,enps0, enps1,data,
            covbase,covtp1,covtp2,  ps0feature = ps0feature, ps1feature = ps1feature,B = 500,
            baseps0 = baseps0, baseps1 = baseps1, cls_stack0 = cls_stack0, cls_stack1 = cls_stack1):
    # out1, out2, 
    out1 = [out1]; out2 = [out2]
    feature1 = np.concatenate([covbase,covtp1,trt1])
    feature2 = np.concatenate([covbase,covtp1,trt1,trt2,covtp2,out1])

    preddat11 = data.copy(); preddat11[trt1]=1;preddat11[trt2]=1
    preddat00 = data.copy(); preddat00[trt1]=0;preddat00[trt2]=0
    basemodel1, lm_stack1 = gcompu_train_ensemble(data,feature2,out2,include_mlp=False,include_rg=False,include_gbr=True,include_xgb = False,include_l1 = True,include_lm = True)
    preddat10 = data.copy(); preddat10[trt1]=1;preddat10[trt2]=0
    preddat01 = data.copy(); preddat01[trt1]=0;preddat01[trt2]=1

    # MSM with ensembel
    # IPW = (data[trt1].values.ravel()/enps0+(1-data[trt1].values.ravel())/(1-enps0))*(data[trt2].values.ravel()/enps1+(1-data[trt2].values.ravel())/(1-enps1))
    # msm_lm = linear_model.LinearRegression()
    # msm_lm.fit(data[[trt1[0],trt2[0]]],data[out2],sample_weight=IPW)
    # phi_msm_sum = (np.sum(msm_lm.coef_)); phi_msm1 = msm_lm.coef_[0]; phi_msm2 = msm_lm.coef_[1]
    # phi_msm_ratio = phi_msm_sum/(phi_msm_sum+msm_lm.intercept_)

    # Gcomputation with ensembel
    gcomp_data = data.copy()
    y_predQ11 = {}
    for name, model in basemodel1.items():
        y_predQ11[name] = model.predict(preddat11[feature2]).ravel()
    Q_baseQ11 = np.transpose(np.array(list(y_predQ11.values())))
    gcomp_data['Q11'] = lm_stack1.predict(Q_baseQ11)

    y_predQ00 = {}
    for name, model in basemodel1.items():
        y_predQ00[name] = model.predict(preddat00[feature2]).ravel()
    Q_baseQ00 = np.transpose(np.array(list(y_predQ00.values())))
    gcomp_data['Q00'] = lm_stack1.predict(Q_baseQ00)

    basemodel0_11, lm_stack_11 = gcompu_train_ensemble(gcomp_data,feature1,'Q11')
    basemodel0_00, lm_stack_00 = gcompu_train_ensemble(gcomp_data,feature1,'Q00')

    y_pred_Q00 = {}
    for name, model in basemodel0_00.items():
        y_pred_Q00[name] = model.predict(preddat00[feature1]).ravel()
    Q_base_Q00 = np.transpose(np.array(list(y_pred_Q00.values())))
    Q_00= lm_stack_00.predict(Q_base_Q00)

    y_pred_Q11 = {}
    for name, model in basemodel0_11.items():
        y_pred_Q11[name] = model.predict(preddat11[feature1]).ravel()
    Q_base_Q11 = np.transpose(np.array(list(y_pred_Q11.values())))
    Q_11= lm_stack_11.predict(Q_base_Q11)

    phi_gcomp = (np.mean(Q_11-Q_00))


    # DR 
    y_pred2 = {}
    for name, model in basemodel1.items():
        y_pred2[name] = model.predict(data[feature2]).ravel()
    Q_base2 = np.transpose(np.array(list(y_pred2.values())))
    eta2 = lm_stack1.predict(Q_base2)

    preddat1 = data.copy(); preddat1[trt2]=1
    preddat0 = data.copy(); preddat0[trt2]=0

    y_pred20 = {}
    for name, model in basemodel1.items():
        y_pred20[name] = model.predict(preddat0[feature2]).ravel()
    Q_base20 = np.transpose(np.array(list(y_pred20.values())))
    eta20 = lm_stack1.predict(Q_base20)
    dat20 = data.copy(); dat20['eta20'] = eta20
    basemodel20, lm_stack20 = gcompu_train_ensemble(dat20,feature1,'eta20',include_rg=False,include_mlp=False,include_gbr=True,include_xgb = False,include_l1 = True,include_lm = True)

    y_pred21 = {}
    for name, model in basemodel1.items():
        y_pred21[name] = model.predict(preddat1[feature2]).ravel()
    Q_base21 = np.transpose(np.array(list(y_pred21.values())))
    eta21 = lm_stack1.predict(Q_base21)
    dat21 = data.copy(); dat21['eta21'] = eta21
    basemodel21, lm_stack21 = gcompu_train_ensemble(dat21,feature1,'eta21',include_rg=False,include_mlp=False,include_gbr=True,include_xgb = False,include_l1 = True,include_lm = True)
    
    y_pred0 = {}
    for name, model in basemodel20.items():
        y_pred0[name] = model.predict(preddat0[feature1]).ravel()
    Q_base0 = np.transpose(np.array(list(y_pred0.values())))
    eta10 = lm_stack20.predict(Q_base0)

    y_pred1 = {}
    for name, model in basemodel21.items():
        y_pred1[name] = model.predict(preddat1[feature1]).ravel()
    Q_base1 = np.transpose(np.array(list(y_pred1.values())))
    eta11 = lm_stack21.predict(Q_base1)

    y_pred_00 = {}
    for name, model in basemodel20.items():
        y_pred_00[name] = model.predict(preddat00[feature1]).ravel()
    Q_base_00 = np.transpose(np.array(list(y_pred_00.values())))
    eta0_00 = lm_stack20.predict(Q_base_00)

    y_pred_11 = {}
    for name, model in basemodel21.items():
        y_pred_11[name] = model.predict(preddat11[feature1]).ravel()
    Q_base_11 = np.transpose(np.array(list(y_pred_11.values())))
    eta0_11= lm_stack21.predict(Q_base_11)

    y_pred_10 = {}
    for name, model in basemodel20.items():
        y_pred_10[name] = model.predict(preddat10[feature1]).ravel()
    Q_base_10 = np.transpose(np.array(list(y_pred_10.values())))
    eta0_10 = lm_stack20.predict(Q_base_10)

    y_pred_01 = {}
    for name, model in basemodel21.items():
        y_pred_01[name] = model.predict(preddat01[feature1]).ravel()
    Q_base_01 = np.transpose(np.array(list(y_pred_01.values())))
    eta0_01 = lm_stack21.predict(Q_base_01)

    # DR step
    t0 = data[trt1].values.ravel(); t1= data[trt2].values.ravel(); y1 = data[out2].values.ravel(); y0 = data[out1].values.ravel()
    f01 = ((t0/enps0+(1-t0)/(1-enps0))*(t1/enps1+(1-t1)/(1-enps1))*(y1-eta2.ravel()))
    f02 = 1/((enps0**t0)*(1-enps0)**(1-t0))*(eta20.ravel()-eta10.ravel())+1/((enps0**t0)*(1-enps0)**(1-t0))*(eta21.ravel()-eta11.ravel())
    f03 = (eta0_00+eta0_11+eta0_10.ravel()+eta0_01.ravel())
    c0 = np.mean(f01+f02+f03)
    b0 = [4,2,2]

    f11 = (t0/enps0+(1-t0)/(1-enps0))*(t1/enps1+(1-t1)/(1-enps1))*(y1-eta2.ravel())*t0
    f12 = t0/((enps0**t0)*(1-enps0)**(1-t0))*(eta20.ravel()-eta10.ravel())+t0/((enps0**t0)*(1-enps0)**(1-t0))*(eta21.ravel()-eta11.ravel())
    f13 = (eta0_11+eta0_10.ravel())
    c1 = np.mean(f11+f12+f13)
    b1 = [2,2,1]

    f21 = (t0/enps0+(1-t0)/(1-enps0))*(t1/enps1+(1-t1)/(1-enps1))*(y1-eta2.ravel())*t1
    f22 = 1/((enps0**t0)*(1-enps0)**(1-t0))*(eta21.ravel()-eta11.ravel())
    f23 = (eta0_11+eta0_01.ravel())
    c2 = np.mean(f21+f22+f23)
    b2 = [2,1,2]

    coef = np.linalg.solve(np.array([b0,b1,b2]),np.array([c0,c1,c2]))

    effratio = (coef[1]+coef[2])/(coef[1]+coef[2]+coef[0])
    beta1 = coef[1]; beta2 = coef[2]; sumeff = beta1+beta2

    dr_boot = []; n = data.shape[0]; beta1_boot = []; beta2_boot = []; dr_ratio = []
    ps_boot = []; msm_boot = []; msm_b1boot = []; msm_b2boot = []
    gcomp_boot = []; gcomp_b1boot = []; gcomp_b2boot = []
    for b in range(B):
        np.random.seed(233+b)
        sampleid = np.random.choice(range(1, n+1), size=n, replace=True)
        dat_boot = data.iloc[sampleid-1, :]
        bootpreddat11 = dat_boot.copy(); bootpreddat11[trt1]=1;bootpreddat11[trt2]=1
        bootpreddat00 = dat_boot.copy(); bootpreddat00[trt1]=0;bootpreddat00[trt2]=0
        bootpreddat10 = dat_boot.copy(); bootpreddat10[trt1]=1;bootpreddat10[trt2]=0
        bootpreddat01 = dat_boot.copy(); bootpreddat01[trt1]=0;bootpreddat01[trt2]=1

        pred_boot1 = dat_boot.copy(); pred_boot1[trt2]=1
        pred_boot0 = dat_boot.copy(); pred_boot1[trt2]=0

        # PS
        bootwo_pred0 = {}
        for name, model in baseps0.items():
            bootwo_pred0[name] = model.predict_proba(dat_boot[ps0feature])[:,1].ravel()
        bootwo_base0 = np.transpose(np.array(list(bootwo_pred0.values())))
        enps0_btwo = cls_stack0.predict_proba(bootwo_base0)[:,1]

        bootwo_pred1 = {}
        for name, model in baseps1.items():
            bootwo_pred1[name] = model.predict_proba(dat_boot[ps1feature])[:,1].ravel()
        bootwo_base1 = np.transpose(np.array(list(bootwo_pred1.values())))
        enps1_btwo = cls_stack1.predict_proba(bootwo_base1)[:,1]

        ## MSM
        IPWboot = (dat_boot[trt1].values.ravel()/enps0_btwo+(1-dat_boot[trt1].values.ravel())/(1-enps0_btwo))*(dat_boot[trt2].values.ravel()/enps1_btwo+(1-dat_boot[trt2].values.ravel())/(1-enps1_btwo))
        msmboot_lm = linear_model.LinearRegression()
        msmboot_lm.fit(dat_boot[[trt1[0],trt2[0]]],dat_boot[out2],sample_weight=IPWboot)
        msm_boot.append((np.sum(msmboot_lm.coef_))); msm_b1boot.append(msmboot_lm.coef_[0][0]); 
        # print(msmboot_lm.coef_,out2,b,msm_b1boot)
        msm_b2boot.append(msmboot_lm.coef_[0][1])
    

        # ICE

        booty_pred2 = {}
        for name, model in basemodel1.items():
            booty_pred2[name] = model.predict(dat_boot[feature2]).ravel()
        bootQ_base2 = np.transpose(np.array(list(booty_pred2.values())))
        booteta2 = lm_stack1.predict(bootQ_base2)

        y_pred20_boot = {}
        for name, model in basemodel1.items():
            y_pred20_boot[name] = model.predict(pred_boot0[feature2]).ravel()
        Q_base20_boot = np.transpose(np.array(list(y_pred20_boot.values())))
        booteta20 = lm_stack1.predict(Q_base20_boot)

        y_pred21_boot = {}
        for name, model in basemodel1.items():
            y_pred21_boot[name] = model.predict(pred_boot1[feature2]).ravel()
        Q_base21_boot = np.transpose(np.array(list(y_pred21_boot.values())))
        booteta21 = lm_stack1.predict(Q_base21_boot)

        booty_pred0 = {}
        for name, model in basemodel20.items():
            booty_pred0[name] = model.predict(pred_boot0[feature1]).ravel()
        bootQ_base0 = np.transpose(np.array(list(booty_pred0.values())))
        booteta10 = lm_stack20.predict(bootQ_base0)

        booty_pred1 = {}
        for name, model in basemodel21.items():
            booty_pred1[name] = model.predict(pred_boot1[feature1]).ravel()
        bootQ_base1 = np.transpose(np.array(list(booty_pred1.values())))
        booteta11 = lm_stack21.predict(bootQ_base1)


        booty_pred_00 = {}
        for name, model in basemodel20.items():
            booty_pred_00[name] = model.predict(bootpreddat00[feature1]).ravel()
        bootQ_base_00 = np.transpose(np.array(list(booty_pred_00.values())))
        booteta0_00 = lm_stack20.predict(bootQ_base_00)

        booty_pred_11 = {}
        for name, model in basemodel21.items():
            booty_pred_11[name] = model.predict(bootpreddat11[feature1]).ravel()
        bootQ_base_11 = np.transpose(np.array(list(booty_pred_11.values())))
        booteta0_11= lm_stack21.predict(bootQ_base_11)

        booty_pred_10 = {}
        for name, model in basemodel20.items():
            booty_pred_10[name] = model.predict(bootpreddat10[feature1]).ravel()
        bootQ_base_10 = np.transpose(np.array(list(booty_pred_10.values())))
        booteta0_10 = lm_stack20.predict(bootQ_base_10)

        booty_pred_01 = {}
        for name, model in basemodel21.items():
            booty_pred_01[name] = model.predict(bootpreddat01[feature1]).ravel()
        bootQ_base_01 = np.transpose(np.array(list(booty_pred_01.values())))
        booteta0_01 = lm_stack21.predict(bootQ_base_01)


        # DR step
        boott0 = dat_boot[trt1].values.ravel(); boott1= dat_boot[trt2].values.ravel(); booty1 = dat_boot[out2].values.ravel(); 
        bootf01 = ((boott0/enps0_btwo+(1-boott0)/(1-enps0_btwo))*(boott1/enps1_btwo+(1-boott1)/(1-enps1_btwo))*(booty1-booteta2.ravel()))
        bootf02 = 1/((enps0_btwo**boott0)*(1-enps0_btwo)**(1-boott0))*(booteta20.ravel()-booteta10.ravel())+1/((enps0_btwo**boott0)*(1-enps0_btwo)**(1-boott0))*(booteta21.ravel()-booteta11.ravel())
        bootf03 = (booteta0_00+booteta0_11+booteta0_10.ravel()+booteta0_01.ravel())
        bootc0 = np.mean(bootf01+bootf02+bootf03)
        bootb0 = [4,2,2]

        bootf11 = (boott0/enps0_btwo+(1-boott0)/(1-enps0_btwo))*(boott1/enps1_btwo+(1-boott1)/(1-enps1_btwo))*(booty1-booteta2.ravel())*boott0
        bootf12 = boott0/((enps0_btwo**boott0)*(1-enps0_btwo)**(1-boott0))*(booteta20.ravel()-booteta10.ravel())+boott0/((enps0_btwo**boott0)*(1-enps0_btwo)**(1-boott0))*(booteta21.ravel()-booteta11.ravel())
        bootf13 = (booteta0_11+booteta0_10.ravel())
        bootc1 = np.mean(bootf11+bootf12+bootf13)
        bootb1 = [2,2,1]

        bootf21 = (boott0/enps0_btwo+(1-boott0)/(1-enps0_btwo))*(boott1/enps1_btwo+(1-boott1)/(1-enps1_btwo))*(booty1-booteta2.ravel())*boott1
        bootf22 = 1/((enps0_btwo**boott0)*(1-enps0_btwo)**(1-boott0))*(booteta21.ravel()-booteta11.ravel())
        bootf23 = (booteta0_11+booteta0_01.ravel())
        bootc2 = np.mean(bootf21+bootf22+bootf23)
        bootb2 = [2,1,2]

        bootcoef = np.linalg.solve(np.array([bootb0,bootb1,bootb2]),np.array([bootc0,bootc1,bootc2]))

        dr_boot.append(bootcoef[1]+bootcoef[2]); beta2_boot.append(bootcoef[1]); beta1_boot.append(bootcoef[2])
        dr_ratio.append((bootcoef[1]+bootcoef[2])/(bootcoef[0]+bootcoef[1]+bootcoef[2]))


    return  effratio, beta1, beta2, sumeff, phi_gcomp, dr_boot, beta2_boot, beta1_boot, dr_ratio, msm_boot,msm_b1boot,msm_b2boot, gcomp_boot, out2

#dr_abcd(output1[0],output2[0])

start_time = time.time()
#np.random.seed(233)
with multiprocess.Pool(processes=30) as pool:
    #np.random.seed(233)
    result_abcd_boot = pool.starmap(dr_abcd_boot, zip(output1,output2))
    pool.close(); pool.join()

end_time = time.time()
(end_time-start_time)