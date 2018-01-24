setwd("/Users/krothaps/Downloads/kaggle/drivendata/")

library(plyr)
library(MASS)
flibs <- c(
  "plyr",
  "xgboost",
  "Hmisc",
  "vtreat",
  "caret",
  "ROCR",
  "ggplot2",
  "gmodels",
  "grid",
  "gridExtra",
  "lubridate",
  "knitr",
  "magrittr",
  "reshape2",
  "rmarkdown",
  "shiny",
  "stargazer",
  "tidyr",
  "readr",
  "stringi",
  "readxl",
  "doParallel",
  "futile.logger",
  "openxlsx",
  "stringr",
  "ModelMetrics",
  "gtools",
  "dplyr"
)

# Checks for required packages and installs if not existing
new_packages <-
  flibs[!(flibs %in% installed.packages()[, "Package"])]
if (length(new_packages))
  install.packages(new_packages, dependencies = TRUE)

lapply(
  flibs,
  FUN = function(X) {
    do.call("require", list(X))
  }
)

library(tidyr)
library(dplyr)
# import files


# e = use_individual(A_hh_tr, A_hh_te, A_ind_tr, A_ind_te)
# 
# A_train <- merge(e, A_hh_tr, by = c("id"))
# A_test <- merge(e, A_hh_te, by = c("id"))
# 


use_individual <- function(hh_tr, hh_te, ind_tr, ind_te, pct){

# hh_tr <- B_train1
# hh_te <- B_hh_te
# ind_tr <- B_ind_tr 
# ind_te <- B_ind_te
  
# hh_cols <- intersect(names(A_hh_tr), names(A_hh_te))
ind_cols <- intersect(names(ind_tr), names(ind_te))  
# ind_cols <- unique(append(ind_cols, c("data")))

ind_data <- rbind(ind_tr[,ind_cols],ind_te[,ind_cols])  
  
cols <- names(ind_data[, sapply(ind_data, class) == 'factor'])

ind_data$count <- 1

c <- ind_data

for (col in cols){
  a <- spread(ind_data, col, count)
  new_cols <- setdiff(colnames(a), cols)
  b <- a[,new_cols]
  
  c <- merge(c, b, by = c("id", "iid"))
  
}

c <- c[,unique(append(c("id"), setdiff(colnames(c), colnames(ind_data))))]

c[is.na(c)] <- 0

d <- aggregate(. ~ id, c, sum)

res <- colSums(d==0)/nrow(d)*100

d <- d[,setdiff(names(d), names(res[res>pct]))]

d
}


remove_missing_levels <- function(fit, test_data, train_data) {
  
  # https://stackoverflow.com/a/39495480/4185785
  
  # drop empty factor levels in test data
  test_data %>%
    droplevels() %>%
    as.data.frame() -> test_data
  
  # 'fit' object structure of 'lm' and 'glmmPQL' is different so we need to
  # account for it
  if (any(class(fit) == "glmmPQL")) {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$contrasts))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    map(fit$contrasts, function(x) names(unmatrix(x))) %>%
      unlist() -> factor_levels
    factor_levels %>% str_split(":", simplify = TRUE) %>%
      extract(, 1) -> factor_levels
    
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  } else {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$xlevels))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    factor_levels <- unname(unlist(fit$xlevels))
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  }
  
  # Select column names in test data that are factor predictors in
  # trained model
  
  predictors <- names(test_data[names(test_data) %in% factors])
  
  # For each factor predictor in your data, if the level is not in the model,
  # set the value to NA
  
  for (i in 1:length(predictors)) {
    found <- test_data[, predictors[i]] %in% model_factors[
      model_factors$factors == predictors[i], ]$factor_levels
    if (any(!found)) {
      # track which variable
      var <- predictors[i]
      # set to NA
      test_data[!found, predictors[i]] <- train_data[1,predictors[i]]
      # drop empty factor levels in test data
      test_data %>%
        droplevels() -> test_data
      # issue warning to console
      message(sprintf(paste0("Setting missing levels in '%s', only present",
                             " in test data but missing in train data,",
                             " to 'NA'."),
                      var))
    }
  }
  return(test_data)
}
