library(dplyr);library(ltmle); library(tidyverse);library(parallel)
D = list(function(row) c(0,0),function(row) c(1,1))
measure = array(dim = c(2,1,2))
dimnames(measure)[[2]] <- c("A")
measure[,,1] <- cbind(0:1); measure[,,2] <- cbind(0:1)
tmledata = abcd_wide_final_tp2 %>% 
  rename(y_2 = nihtbx_picvocab_agecorrected_1,
         y_3 = nihtbx_picvocab_agecorrected_2)
tmlecov = c(covtp1,covtp2)
tmledata = tmledata %>% select(c(covtp1,trttp1,'y_2',covtp2,trttp2,'y_3'))
start_time = Sys.time()
tlmemsm.est <- ltmleMSM(tmledata,Anodes = c(trttp1,trttp2),
                    Lnodes = c(covtp1,covtp2),regimes = D,
                    Ynodes = c("y_2","y_3"),final.Ynodes = c("y_2","y_3"),
                    survivalOutcome = F,summary.measures = measure,
                    working.msm = "Y~A",
                    SL.library = list(Q = c("SL.glm", "SL.gam", "SL.xgboost","SL.glm.interaction"),
                                      g = c("SL.glm", "SL.gam", "SL.xgboost","SL.glm.interaction")),
                    estimate.time = FALSE)
Sys.time()-start_time
summary(tlmemsm.est)$cmat['A','Estimate']
summary(tlmemsm.est)$cmat['A','Std. Error']
print(summary(tlmemsm.est))

tmledata2 = abcd_wide_final_tp2 %>% select(c(covtp1,trttp1,"nihtbx_picvocab_agecorrected_1",covtp2,trttp2,"nihtbx_picvocab_agecorrected_2")) #%>%
  # filter((!is.na(nihtbx_picvocab_agecorrected_1))&(!is.na(nihtbx_picvocab_agecorrected_2)))
tmleest <- ltmle(tmledata2,Anodes = c(trttp1,trttp2),Lnodes = c(covtp1,covtp2),
                 Ynodes = c("nihtbx_picvocab_agecorrected_1",
                            "nihtbx_picvocab_agecorrected_2"),
                 SL.library = list(Q = c("SL.glm", "SL.gam", "SL.xgboost","SL.glm.interaction"),
                                   g = c("SL.glm", "SL.gam", "SL.xgboost","SL.glm.interaction")),
                 abar = list(treatment = c(1,1),control = c(0,0)),
                 estimate.time = F)
print(summary(tmleest))


## adjust tmle_msm on simulation
test_tmle = read.csv('~/Dropbox/MSM/simulation/test_tmle.csv')

start_time = Sys.time()
ttmle_est <- ltmleMSM(test_tmle,Anodes = c('t0','t1'),
                        Lnodes = colnames(test_tmle)[grepl('z',colnames(test_tmle))],regimes = D,
                        Ynodes = c("y0","y1"),final.Ynodes = c("y0","y1"),
                        survivalOutcome = F,summary.measures = measure,
                        working.msm = "Y~A",
                        SL.library = list(Q = c("SL.glm","SL.gam","SL.xgboost"),
                                          g = c("SL.glm","SL.gam","SL.xgboost")),
                        estimate.time = FALSE)
Sys.time()-start_time
print(summary(ttmle_est))


start_time = Sys.time()
tmleest <- ltmle(test_tmle,Anodes = c('t0','t1'),
                 Lnodes = colnames(test_tmle)[grepl('z',colnames(test_tmle))],
                 Ynodes = c("y0","y1"),
                 SL.library = list(Q = c("SL.glm","SL.gam","SL.xgboost"),
                                   g = c("SL.glm","SL.gam","SL.xgboost")),
                 abar = list(treatment = c(1,1),control = c(0,0)),
                 estimate.time = F)
Sys.time()-start_time
print(summary(tmleest))



##################################
#### real data application for paper
##################################
## for complete data only
run_tmle <- function(data,covtp1,covtp2,trttp1,trttp2, node_names) {
  # Renaming variables as per specific node names for clarity in TMLE function
  names(data)[names(data) == node_names$y1] <- "y1"
  names(data)[names(data) == node_names$y2] <- "y2"
  
  # Select relevant columns
  tmledata <- data %>%
    select(all_of(c(covtp1, trttp1, "y1", covtp2, trttp2, "y2"))) %>%
    filter(!is.na(y1) & !is.na(y2))  # Ensure there are no NA values in the key outcome variables
  
  # Run TMLE
  tmle_est <- ltmle(
    tmledata,
    Anodes = c(trttp1, trttp2),
    Lnodes = c(covtp1, covtp2),
    Ynodes = c("y1", "y2"),
    SL.library = list(Q = c("SL.glm", "SL.gam", "SL.xgboost","SL.glmnet"),
                      g = c("SL.glm", "SL.gam", "SL.xgboost","SL.glmnet")),
    abar = list(treatment = c(1, 1), control = c(0, 0)),
    estimate.time = FALSE
  )
  
  # Extract estimates and standard errors
  est <- summary(tmle_est)$effect.measures$ATE$estimate
  se <- summary(tmle_est)$effect.measures$ATE$std.dev
  
  return(c(tmle_est = est, tmle_se = se))
}

var_pairs <-  lapply(seq_along(outtp12_1), function(i) {
  list(y1 = outtp12_1[i], y2 = outtp12_2[i])
})

## parallel computing
start_time = Sys.time()
cl <- makeCluster(6)
# Export necessary objects and functions to the cluster
clusterExport(cl, varlist = c("run_tmle", "abcd_wide_final_tp2","covbase",
                              "covtp1","covtp2","trttp1","trttp2", "var_pairs"))
clusterEvalQ(cl, {
  library(dplyr)
  library(ltmle)
  library(tidyr)
})
# Use parLapply to run in parallel
results <- parLapply(cl, var_pairs, function(pair) {
  run_tmle(abcd_wide_final_tp2,c(covbase,covtp1),covtp2,trttp1,trttp2, pair)
})
# Stop the cluster
stopCluster(cl)
Sys.time()-start_time
result_tmle = do.call(rbind,results)


###################
# for median imputed data
start_time = Sys.time()
cl <- makeCluster(6)
# Export necessary objects and functions to the cluster
clusterExport(cl, varlist = c("run_tmle", "abcd_wide_cl1","covbase",
                              "covtp1","covtp2","trttp1","trttp2", "var_pairs"))
clusterEvalQ(cl, {
  library(dplyr)
  library(ltmle)
  library(tidyr)
})
# Use parLapply to run in parallel
results_m <- parLapply(cl, var_pairs, function(pair) {
  run_tmle(abcd_wide_cl1,c(covbase,covtp1),covtp2,trttp1,trttp2, pair)
})
# Stop the cluster
stopCluster(cl)
Sys.time()-start_time
result_tmle_m = do.call(rbind,results_m)



###########################
# for three time points


run_tmle3 <- function(data,covtp1,covtp2,covtp3,trttp1,trttp2,trttp3, node_names) {
  # Renaming variables as per specific node names for clarity in TMLE function
  names(data)[names(data) == node_names$y1] <- "y1"
  names(data)[names(data) == node_names$y2] <- "y2"
  names(data)[names(data) == node_names$y3] <- "y3"
  
  # Select relevant columns
  tmledata <- data %>%
    select(all_of(c(covtp1, trttp1, "y1", covtp2, trttp2, "y2",covtp3,trttp3,'y3'))) %>%
    filter(!is.na(y1) & !is.na(y2)&!is.na(y3))  # Ensure there are no NA values in the key outcome variables
  
  # Run TMLE
  tmle_est <- ltmle(
    tmledata,
    Anodes = c(trttp1, trttp2,trttp3),
    Lnodes = c(covtp1, covtp2,covtp3),
    Ynodes = c("y1", "y2","y3"),
    SL.library = list(Q = c("SL.glm", "SL.gam", "SL.xgboost","SL.glmnet"),
                      g = c("SL.glm", "SL.gam", "SL.xgboost","SL.glmnet")),
    abar = list(treatment = c(1, 1,1), control = c(0, 0,0)),
    estimate.time = FALSE
  )
  
  # Extract estimates and standard errors
  est <- summary(tmle_est)$effect.measures$ATE$estimate
  se <- summary(tmle_est)$effect.measures$ATE$std.dev
  
  return(c(tmle_est = est, tmle_se = se))
}

var_pairs <-  lapply(seq_along(outtp123_1), function(i) {
  list(y1 = outtp123_1[i], y2 = outtp123_2[i],y3 = outtp123_3[i])
})




## parallel computing
start_time = Sys.time()
cl <- makeCluster(6)
# Export necessary objects and functions to the cluster
clusterExport(cl, varlist = c("run_tmle3", "abcd_wide_cl3","covbase",
                              "covtp1","covtp2","covtp3","trttp1",
                              "trttp2", "trttp3","var_pairs"))
clusterEvalQ(cl, {
  library(dplyr)
  library(ltmle)
  library(tidyr)
})
# Use parLapply to run in parallel
results <- parLapply(cl, var_pairs, function(pair) {
    # Attempt to run your original function
    run_tmle3(abcd_wide_cl3, c(covbase, covtp1), covtp2, covtp3, trttp1, trttp2, trttp3, pair)
})
# Stop the cluster
stopCluster(cl)
Sys.time()-start_time
result_tmle_three = do.call(rbind,results)


# Check results for errors



