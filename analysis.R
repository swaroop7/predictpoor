setwd("/Users/krothaps/Downloads/kaggle/drivendata/")
library(ModelMetrics)
# Compute logloss for the validation dataset and use it to benchmark various approaches
# Sampling up and down
# Model C has lot of improvement to be done

source("Jan20.R")

set.seed(123)
# import files
A_hh_te = read.csv('A_hhold_test.csv', header=TRUE) 
A_hh_tr = read.csv('A_hhold_train.csv', header=TRUE)
A_ind_te = read.csv('A_indiv_test.csv', header=TRUE)
A_ind_tr = read.csv('A_indiv_train.csv', header=TRUE)

B_hh_te = read.csv('B_hhold_test.csv', header=TRUE) 
B_hh_tr = read.csv('B_hhold_train.csv', header=TRUE)
B_ind_te = read.csv('B_indiv_test.csv', header=TRUE)
B_ind_tr = read.csv('B_indiv_train.csv', header=TRUE)

C_hh_te = read.csv('C_hhold_test.csv', header=TRUE) 
C_hh_tr = read.csv('C_hhold_train.csv', header=TRUE)
C_ind_te = read.csv('C_indiv_test.csv', header=TRUE)
C_ind_tr = read.csv('C_indiv_train.csv', header=TRUE)


tc <- trainControl(method = 'repeatedcv', 
                   number = 5, 
                   repeats = 1, #set repeats=5 later
                   verboseIter = TRUE,
                   classProbs = TRUE, # TRUE if using AUC, FALSE if using Kappa
                   summaryFunction = twoClassSummary , # mnLogLoss ; twoClassSummary if AUC, defaultSummary if Kappa
                   selectionFunction = "oneSE", # "best", "oneSE", to be more conservative,
                   returnData = F,
                   trim  = FALSE)

## set xgb parameters
xgbGrid = expand.grid(
  nrounds = c(150, 100, 75, 50, 25, 10),
  max_depth = c(1),
  eta = c(0.1, 0.3),
  colsample_bytree = c(0.8,1),
  gamma = c(0),
  min_child_weight = c(2, 5),
  subsample = c(0.75,1)
)

xgbGrid_B = expand.grid(
  nrounds = c(150, 100, 75, 50, 25, 10),
  max_depth = c(1),
  eta = c(0.1, 0.3),
  colsample_bytree = c(0.8,1),
  gamma = c(0),
  min_child_weight = c(2),
  subsample = c(0.75,1)
)

# train model
formula_str <- paste0("poor", " ~.")
train_formula <- as.formula(formula_str)

# A_train <- A_hh_tr[, !(colnames(A_hh_tr) %in% c("id","country"))]
e = use_individual(A_hh_tr, A_hh_te, A_ind_tr, A_ind_te, 70)

A_train <- merge(A_hh_tr,e, by = c("id"), sort = FALSE)
A_test <- merge(A_hh_te, e, by = c("id"), sort = FALSE)

A_train <- A_train[, !(colnames(A_train) %in% c("id","country"))]

train.index <- createDataPartition(A_train$poor, p = .7, list = FALSE)
A_train_1 <- A_train[ train.index,]
A_valid  <- A_train[-train.index,]

tc_A <- trainControl(method = 'repeatedcv', 
                   number = 5, 
                   repeats = 5, #set repeats=5 later
                   verboseIter = TRUE,
                   classProbs = TRUE, # TRUE if using AUC, FALSE if using Kappa
                   summaryFunction = defaultSummary , # mnLogLoss ; twoClassSummary if AUC, defaultSummary if Kappa
                   selectionFunction = "oneSE", # "best", "oneSE", to be more conservative,
                   returnData = F,
                   trim  = FALSE)

xgbfit_A <- caret::train(train_formula,
                       A_train_1,
                       method = "xgbTree",
                       tuneGrid = xgbGrid,
                       metric = "Kappa", # PRAUC # GainAUC
                       # preProc = c("center", "scale"),
                       trControl = tc_A,
                       nthread = 7)

# Kappa - 0.548
# Logloss - 0.856
# ROC - 0.80

confusionMatrix(A_valid$poor, stats::predict(xgbfit_A, newdata = A_valid, type="raw" ))

A_valid$Y <- stats::predict(xgbfit_A, newdata = A_valid , type = "prob")$True
A_train_1$Y <- stats::predict(xgbfit_A, newdata = A_train_1 , type = "prob")$True

A_valid_Logloss <- logLoss(A_valid$poor, A_valid$Y)
A_tr_Logloss <- logLoss(A_train_1$poor, A_train_1$Y)

AUC_train_1_A <- ROCR::performance(prediction(A_train_1$Y, 
                                          A_train_1$poor), measure = "auc")@y.values[[1]]
AUC_valid_A <- ROCR::performance(prediction(A_valid$Y, 
                                            A_valid$poor), measure = "auc")@y.values[[1]]

A_test$poor <- stats::predict(xgbfit_A, newdata = A_test , type = "prob")$True


# B

B_train1 <- B_hh_tr[, !(colnames(B_hh_tr) %in% c("country"))]

B_train1 <- B_train1[,!sapply(B_train1,function(x) any(is.na(x)))]  # columns with missing values

B_train1 <- Filter(function(x)(length(unique(x))>1), B_train1) # columns with just one value

e2 = use_individual(B_train1, B_hh_te, B_ind_tr, B_ind_te, 70)

B_train <- merge(B_train1, e2, by = c("id"), sort = FALSE)
B_test <- merge(B_hh_te, e2, by = c("id"), sort = FALSE)

B_train <- B_train[, !(colnames(B_train) %in% c("id","country"))]

train.index <- createDataPartition(B_train$poor, p = .7, list = FALSE)
B_train_1 <- B_train[ train.index,]
B_valid  <- B_train[-train.index,]

B_train_1 <- B_train_1[, !(colnames(B_train_1) %in% c("TDBNBDil","StTakdgB","WTsZXhAX","RzaXNcgd"))]

tc_B <- trainControl(method = 'repeatedcv', 
                     number = 5, 
                     repeats = 5, #set repeats=5 later
                     verboseIter = TRUE,
                     classProbs = TRUE, # TRUE if using AUC, FALSE if using Kappa
                     summaryFunction = mnLogLoss , # mnLogLoss ; twoClassSummary if AUC, defaultSummary if Kappa
                     selectionFunction = "oneSE", # "best", "oneSE", to be more conservative,
                     returnData = F,
                     trim  = FALSE)

xgbfit_B <- caret::train(train_formula,
                       B_train_1,
                       method = "xgbTree",
                       tuneGrid = xgbGrid_B,
                       metric = "logLoss", # PRAUC # GainAUC
                       # preProc = c("center", "scale"),
                       trControl = tc_B,
                       nthread = 7)

# Kappa - 3.932389
# ROC - 3.6     
# logloss - 3.279433  

confusionMatrix(B_valid$poor, stats::predict(xgbfit_B, newdata = B_valid, type="raw" ))

B_train_1$Y <- stats::predict(xgbfit_B, newdata = B_train_1 , type = "prob")$True
B_valid$Y <- stats::predict(xgbfit_B, newdata = B_valid , type = "prob")$True

B_valid_Logloss <- logLoss(B_valid$poor, B_valid$Y)
B_tr_Logloss <- logLoss(B_train_1$poor, B_train_1$Y)

AUC_train_1_B <- ROCR::performance(prediction(B_train_1$Y, 
                                              B_train_1$poor), measure = "auc")@y.values[[1]]

AUC_valid_B <- ROCR::performance(prediction(B_valid$Y, 
                                            B_valid$poor), measure = "auc")@y.values[[1]]

# B_score_2 <- remove_missing_levels(xgbfit_B, B_test, B_train)

# B_score_2$poor <- stats::predict(xgbfit_B, newdata = B_score_2 , type = "prob")$True
B_test$poor <- stats::predict(xgbfit_B, newdata = B_test , type = "prob")$True


# C

C_train1 <- C_hh_tr[, !(colnames(C_hh_tr) %in% c("country"))]

C_train1 <- C_train1[,!sapply(C_train1,function(x) any(is.na(x)))]  # columns with missing values

C_train1 <- Filter(function(x)(length(unique(x))>1), C_train1) # columns with just one value

e3 = use_individual(C_train1, C_hh_te, C_ind_tr, C_ind_te, 70)

# 50 - 5.592
# 90 - 5.117

C_train <- merge(C_train1, e3, by = c("id"), sort = FALSE)
C_test <- merge(C_hh_te, e3, by = c("id"), sort = FALSE)

C_train <- C_train[, !(colnames(C_train) %in% c("id","country"))]

train.index <- createDataPartition(C_train$poor, p = .7, list = FALSE)
C_train_1 <- C_train[ train.index,]
C_valid  <- C_train[-train.index,]

C_train_1 <- C_train_1[, !(colnames(C_train_1) %in% c('DNnBfiSI', "gZWEypOM","rcVCcnDz","Bknkgmhs","IrEVgBSw","enTUTSQi","nuMtebks"))]

tc_C <- trainControl(method = 'repeatedcv', 
                     number = 5, 
                     repeats = 5, #set repeats=5 later
                     verboseIter = TRUE,
                     classProbs = TRUE, # TRUE if using AUC, FALSE if using Kappa
                     summaryFunction = twoClassSummary , # mnLogLoss ; twoClassSummary if AUC, defaultSummary if Kappa
                     selectionFunction = "oneSE", # "best", "oneSE", to be more conservative,
                     returnData = F,
                     trim  = FALSE)

xgbfit_C <- caret::train(train_formula,
                         C_train_1,
                         method = "xgbTree",
                         tuneGrid = xgbGrid,
                         metric = "ROC", # PRAUC # GainAUC
                         # preProc = c("center", "scale"),
                         trControl = tc_C,
                         nthread = 7)

# Kappa - 4.97
# ROC - 5.88
# logloss - 5.04


confusionMatrix(C_valid$poor, stats::predict(xgbfit_C, newdata = C_valid, type="raw" ))

C_valid$Y <- stats::predict(xgbfit_C, newdata = C_valid , type = "prob")$True
C_train_1$Y <- stats::predict(xgbfit_C, newdata = C_train_1 , type = "prob")$True

C_valid_Logloss <- logLoss(C_valid$poor, C_valid$Y)
C_tr_Logloss <- logLoss(C_train_1$poor, C_train_1$Y)

AUC_train_1_C <- ROCR::performance(prediction(C_train_1$Y, 
                                          C_train_1$poor), measure = "auc")@y.values[[1]]
AUC_valid_C <- ROCR::performance(prediction(C_valid$Y, 
                                            C_valid$poor), measure = "auc")@y.values[[1]]

# C_score_2 <- remove_missing_levels(xgbfit_C, C_test, C_train)
# 
# C_score_2$poor <- stats::predict(xgbfit_C, newdata = C_score_2 , type = "prob")$True

C_test$poor <- stats::predict(xgbfit_C, newdata = C_test , type = "prob")$True


# **************************************


a <- data.frame(A_test$id, A_test$country, A_test$poor)
# b <- data.frame(B_score_2$id, B_score_2$country, B_score_2$poor)
# c <- data.frame(C_score_2$id, C_score_2$country, C_score_2$poor)

b <- data.frame(B_test$id, B_test$country, B_test$poor)
c <- data.frame(C_test$id, C_test$country, C_test$poor)

colnames(a) <- c("id","country","poor")
colnames(b) <- c("id","country","poor")
colnames(c) <- c("id","country","poor")

valid_logloss <- A_valid_Logloss + B_valid_Logloss + C_valid_Logloss
train_logloss <- A_tr_Logloss + B_tr_Logloss + C_tr_Logloss

cat(c(train_logloss,valid_logloss))  # 8.38 8.459 
# 9.62 (1.12 + 3.41  + 5.24) - 9.77 (1.12 + 3.41 + 5.23)
#10.10 10.16 (1.00 + 3.35 + 5.74)

final <- rbind.fill(a,b,c)

 # write.table(final, file = "submission10.csv",sep=",", row.names = FALSE)
