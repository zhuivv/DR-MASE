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
from ensemble_func import ps_function,ps_predict,gcompu_function,gcompu_predict


# estimation funciton
def dr_abcd_boot(out1,out2,trt1,trt2,enps0, enps1,data,
            covbase,covtp1,covtp2,  ps0feature, ps1feature,
            baseps0_tr, baseps0_ts, baseps1_tr,baseps1_ts, 
            cls_stack0_tr,cls_stack0_ts, cls_stack1_tr,cls_stack1_ts,B = 500,):
    # out1, out2, 
    out1 = [out1]; out2 = [out2]
    feature1 = np.concatenate([covbase,covtp1,trt1])
    feature2 = np.concatenate([covbase,covtp1,trt1,trt2,covtp2,out1])

    preddat11 = data.copy(); preddat11[trt1]=1;preddat11[trt2]=1
    preddat00 = data.copy(); preddat00[trt1]=0;preddat00[trt2]=0
    preddat10 = data.copy(); preddat10[trt1]=1;preddat10[trt2]=0
    preddat01 = data.copy(); preddat01[trt1]=0;preddat01[trt2]=1

    basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts = gcompu_function(data,feature2,out2,xgb=False)

    # DR 
    eta2 = gcompu_predict(data,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)

    preddat1 = data.copy(); preddat1[trt2]=1
    preddat0 = data.copy(); preddat0[trt2]=0

    eta20 = gcompu_predict(preddat0,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)
    
    dat20 = data.copy(); dat20['eta20'] = eta20
    basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts = gcompu_function(dat20,feature1,'eta20',xgb=False)

    eta21 = gcompu_predict(preddat1,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)
    
    dat21 = data.copy(); dat21['eta21'] = eta21
    basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts = gcompu_function(dat21,feature1,'eta21',xgb=False)

    eta10 = gcompu_predict(preddat0,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)

    eta11 = gcompu_predict(preddat1,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)

    eta0_00 = gcompu_predict(preddat00,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)

    eta0_11= gcompu_predict(preddat11,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)

    eta0_10 = gcompu_predict(preddat10,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)

    eta0_01 = gcompu_predict(preddat01,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)

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
        enps0_btwo = ps_predict(dat_boot,ps0feature,baseps0_tr,baseps0_ts,cls_stack0_tr,cls_stack0_ts)
        enps1_btwo = ps_predict(dat_boot,ps1feature,baseps1_tr,baseps1_ts,cls_stack1_tr,cls_stack1_ts)    

        # ICE
        booteta2 = gcompu_predict(dat_boot,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)
        booteta20 = gcompu_predict(pred_boot0,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)
        booteta21 = gcompu_predict(pred_boot1,feature2,basemodel1_tr,basemodel1_ts,lm_stack1_tr,lm_stack1_ts)
        
        booteta10 = gcompu_predict(pred_boot0,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)
        booteta11 = gcompu_predict(pred_boot1,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)

        booteta0_00 = gcompu_predict(bootpreddat00,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)
        booteta0_11= gcompu_predict(bootpreddat11,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)
        booteta0_10 = gcompu_predict(bootpreddat10,feature1,basemodel20_tr,basemodel20_ts,lm_stack20_tr,lm_stack20_ts)
        booteta0_01 = gcompu_predict(bootpreddat01,feature1,basemodel21_tr,basemodel21_ts,lm_stack21_tr,lm_stack21_ts)

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


    return  sumeff, beta1, beta2, dr_boot, beta2_boot, beta1_boot

#dr_abcd(output1[0],output2[0])

