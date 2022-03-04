## Capstone Independent Project Script
## Can We Predict Happiness?
## Oscar Mak
## March 2022

# Load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(pdftools)) install.packages("pdftools", repos = "http://cran.us.r-project.org")
if(!require(haven)) install.packages("haven", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(pdftools)
library(haven)
library(xgboost)
library(scales)

# Download and unzip General Social Survery (GSS) data file
dl <- tempfile()
download.file("https://gss.norc.org/documents/stata/GSS_stata.zip", dl)
unzip(dl)
gss_data <- read_dta("gss7221_r1b.dta")

# View data
str(gss_data, list.len = 3)     # We see this has 68,846 obs of 6,309 variables

# Keep only replicating core variables (questions consistently asked each year)
# Download replicating core guide PDF
rep_core_pdf <- tempfile()
download.file("https://gss.norc.org/Documents/other/Replicating%20Core.pdf",
              rep_core_pdf)
rep_core_text <- pdf_text(rep_core_pdf)     # Extract pdf text

# Reading PDF shows necessary variable names on pages 2-12, remove excess pages
rep_core_text <- rep_core_text[2:12]  

# Variables names are ALLCAPS text 3 or more characters long, followed by 0-3 digits
# Extracts variables names for questions in replicating core into one long list
pattern <- "[A-Z]{3,}[0-9]{0,3}"
rep_core_codes <- str_extract_all(rep_core_text, pattern) %>%
  unlist() %>%  
  tolower()     # variables names are lowercase in gss_data
rep_core_codes <- rep_core_codes[rep_core_codes != "gss"]  # GSS is survey name

# Keep only columns in the replicating core
gss_data <- gss_data[, colnames(gss_data) %in% rep_core_codes] 
length(gss_data)     # 484 variables after filtering for replicating core variables

# Use nearZeroVar to remove variables with near-zero variance 
nzv <- nearZeroVar(gss_data)
gss_data <- gss_data[, -nzv]
length(gss_data)     # 468 variables after filtering for NZV

# Are there a lot of NAs?
dim(gss_data)     # 68,846 * 468 grid
sum(is.na(gss_data))     # 17,990,906 NAs in grid
sum(is.na(gss_data))/(dim(gss_data)[1] * dim(gss_data)[2]) # 56% of grid is NAs

# Do some variables have more NAs? Let's make a histogram!
colMeans(is.na(gss_data)) %>% 
hist(main = "Histogram of variables by % observations NA",
      xlab = "% of variable observations is NA",
      ylab = "Count of variables",
      ylim = c(0, 100),
      labels = TRUE)

# Keep variables with low missing/NA percentage
low_NAs <- colMeans(is.na(gss_data)) <= 0.4
gss_data <- gss_data[,low_NAs]
length(gss_data)  # 133 variables after filtering for NA percentage   

# Manually inspect remaining 133 variables for removal
n <- length(gss_data)
data_labels <- vector(length = n)     # create an empty vector
for(i in 1:n){
  data_labels[i] <- attributes(gss_data[[i]])$label
  } # fill vector with label metadata

# Create table with data for each variable
names_labels <- data.frame(
  index = seq.int(length(data_labels)),
  label = data_labels, 
  pct_NA = percent(colMeans(is.na(gss_data)), accuracy = 0.1)
  )
names_labels     # Display table of variables with labels
#write.csv(names_labels, "Names_And_Labels.csv")   # Export to CSV for viewing in Excel

# Create vector of column numbers for variables to keep
keep_index <- c(1:4, 6, 7, 11:14, 17, 20:29, 37, 42, 43, 55:65, 79:83, 94:96, 
                100, 107, 108, 111, 116, 117, 119:121, 123, 126) 
gss_data <- gss_data[, keep_index]  # Drop unwanted variables

# Display remaining variables
names_labels <- names_labels[keep_index,]     
names_labels

# Explore years data
unique(gss_data$year)     # Years from 1972 to 2021 with some skips 

# Remove 2021 data to get rid of pandemic effect
gss_data <- gss_data %>% filter(year != 2021)

# Check for NAs in happy and keep only observations with non-NA in happy
sum(is.na(gss_data$happy))
gss_data <- gss_data %>% drop_na(happy)
nrow(gss_data)

# Look at happy variable
table(gss_data$happy)
attributes(gss_data$happy)

# Convert data to be usable by XGBoost
outcomes <- as.integer(gss_data$happy) - 1  # Extract happy, convert to integer from zero
gss_data$happy <- NULL                      # Remove happy column from gss_data
features <- as.matrix(gss_data)

# Carve out 10% of data as validation set
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = outcomes, times = 1, p = 0.1, list = FALSE)
dev_features <- features[-test_index,]
dev_outcomes <- outcomes[-test_index]
validation_features <- features[test_index,]
validation_outcomes <- outcomes[test_index]

# Split development data 90% into train_set and 10% into test_set
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = dev_outcomes, times = 1, p = 0.1, list = FALSE)
train_features <- dev_features[-test_index,]
train_outcomes <- dev_outcomes[-test_index]
test_features <- dev_features[test_index,]
test_outcomes <- dev_outcomes[test_index]

# Most basic model - predict the most common value
most_freq <- names(sort(-table(train_outcomes)))[1]
most_freq
mean(test_outcomes == most_freq)     # 56.0% accuracy

# Remove unnecessary elements to save memory
rm(gss_data, names_labels, test_index, data_labels, dl, i, keep_index, low_NAs, 
   most_freq,n, nzv, pattern, rep_core_codes, rep_core_pdf, rep_core_text)
  
# Implement XGBoost
# Convert train and test sets into XGBoost matrix objects
xgboost_train <- xgb.DMatrix(data = train_features, label = train_outcomes)
xgboost_test <- xgb.DMatrix(data = test_features, label = test_outcomes)

# Train the model using defaults
num_class <- length(levels(as.factor(train_outcomes)))
fit <- xgb.train(data = xgboost_train,
                 objective = "multi:softprob",   # output class probabilities
                 booster = "gbtree",             # use tree for classification problems
                 eval_metric = "mlogloss",
                 watchlist = list(val_1 = xgboost_train),
                 eta = 0.3,                      # tuning parameter default = 0.3
                 nrounds = 100,                  # tuning parameter default = 100
                 max.depth = 6,                  # tuning parameter default = 6
                 num_class = num_class,
                 early_stopping_rounds = 10,
                 verbose = FALSE)                # turns off progress reports

# Make predictions on test features. Output is probability by class
pred <- as_tibble(predict(fit, test_features, reshape = TRUE))
colnames(pred) <- levels(as.factor(train_outcomes)) 

# Predict the class with the highest probability
pred$prediction <- apply(pred, 1, function(x) colnames(pred)[which.max(x)])
head(pred)   # Display first few rows

# Calculate accuracy of predictions
mean(pred$prediction == test_outcomes)  
#56.0% accuracy guessing the mode
#61.7% accuracy with defaults

#Tune eta - takes a while to run
etas <- seq(0, 1, 0.1)
accuracies <- sapply(etas, function(n){
  fit <- xgb.train(data = xgboost_train,
                   objective = "multi:softmax",   # output class probabilities
                   booster = "gbtree",            # use tree for classification problems
                   eval_metric = "error",
                   eta = n,                       # parameter being tuned
                   nrounds = 100,                 # tuning parameter default = 100
                   max.depth = 6,                 # tuning parameter default = 6
                   num_class = num_class,
                   verbose = FALSE)               # turns off progress reports
  pred <- as_tibble(predict(fit, test_features, reshape = TRUE))
  return(mean(pred == test_outcomes))  
})
best_eta <- etas[which.max(accuracies)]
best_eta
max(accuracies)

#56.0% accuracy guessing the mode
#61.7% accuracy using defaults
#62.5% with eta tuned to 0.1

#Tune nrounds - takes a while to run
nrounds <- seq(100, 200, 10)
accuracies <- sapply(nrounds, function(n){
  fit <- xgb.train(data = xgboost_train,
                 objective = "multi:softmax",   # output class probabilities
                 booster = "gbtree",            # use tree for classification problems
                 eval_metric = "error",
                 eta = best_eta,                # use prior tuning result
                 nrounds = n,                   # parameter being tuned
                 max.depth = 6,                 # tuning parameter default = 6
                 num_class = num_class,
                 verbose = FALSE)               # turns off progress reports
  pred <- as_tibble(predict(fit, test_features, reshape = TRUE))
  return(mean(pred == test_outcomes))  
})
best_nrounds <- nrounds[which.max(accuracies)]
best_nrounds
max(accuracies)

#56.0% accuracy guessing the mode
#61.7% when eta, max depth, num_rounds all defaults
#62.5% when eta tuned to 0.1
#62.7% when eta and nrounds tuned

#Tune max.depth - takes a while to run
depths <- seq(1, 10, 1)
accuracies <- sapply(depths, function(n){
  fit <- xgb.train(data = xgboost_train,
                   objective = "multi:softmax",   # output class probabilities
                   booster = "gbtree",            # use tree for classification problems
                   eval_metric = "error",
                   eta = best_eta,                # use prior tuning result
                   nrounds = best_nrounds,        # use prior tuning result
                   max.depth = n,                 # parameter being tuned
                   num_class = num_class,
                   verbose = FALSE)               # turns off progress reports
  pred <- as_tibble(predict(fit, test_features, reshape = TRUE))
  return(mean(pred == test_outcomes))  
})
best_depth <- depths[which.max(accuracies)]
best_depth
max(accuracies)

# 62.7% is best we can do after tuning nrounds, eta, and max depth
# Fit the model using tuned parameters
fit <- xgb.train(data = xgboost_train,
                 objective = "multi:softmax",
                 booster = "gbtree",         
                 eval_metric = "error",
                 max.depth = best_depth,    
                 eta = best_eta,            
                 nrounds = best_nrounds,    
                 num_class = num_class,
                 verbose = FALSE)

# Apply to validation set
pred <- as_tibble(predict(fit, validation_features, reshape = TRUE))
final_accuracy <- mean(pred == validation_outcomes)
final_accuracy   # 63.0% when applied to validation set

# See most important features
importance <- xgb.importance(model = fit)
head(importance, 5)
