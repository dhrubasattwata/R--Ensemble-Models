# Ensemble Learning: Bagging

### Import the packages

library(dplyr)       # for data wrangling
library(ggplot2)     # for awesome plotting
library(doParallel)  # for parallel backend to foreach
library(foreach)     # for parallel processing with for loops

library(caret)       # for general model fitting
library(rpart)       # for fitting decision trees
library(ipred)       # for fitting bagged decision trees
library(AmesHousing) # for dataset
library(rsample)     # for train-test split

### Dataset and Train-Test Split
ames <- AmesHousing::make_ames()
# Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.8, strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

### Setting up the model
# make bootstrapping reproducible
set.seed(123)

# train bagged model
ames_bag1 <- bagging(
  formula = Sale_Price ~ .,
  data = ames_train,
  nbagg = 100,  
  coob = TRUE,
  control = rpart.control(minsplit = 2, cp = 0)
)

ames_bag1

# Finding optimal number of stabilizing trees
rmse <- c()
### to find stabilizing tree
for (i in 1:100){
#  bag <- bagging(
    formula = Sale_Price ~ .,
    data = ames_train,
    nbagg = i,  
    coob = TRUE,
    control = rpart.control(minsplit = 2, cp = 0)
  )
  rmse[i] <- bag$err
}
plot(rmse, type="l",col="darkgreen")




## Parallel Computing
# Create a parallel socket cluster
cl <- makeCluster(8) # use 8 workers
registerDoParallel(cl) # register the parallel backend

# Fit trees in parallel and compute predictions on the test set
predictions <- foreach(
  icount(160), 
  .packages = "rpart", 
  .combine = cbind
) %dopar% {
  # bootstrap copy of training data
  index <- sample(nrow(ames_train), replace = TRUE)
  ames_train_boot <- ames_train[index, ]  
  
  # fit tree to bootstrap copy
  bagged_tree <- rpart(
    Sale_Price ~ ., 
    control = rpart.control(minsplit = 2, cp = 0),
    data = ames_train_boot
  ) 
  
  predict(bagged_tree, newdata = ames_test)
}

predictions[1:5, 1:7]

# Plot RMSE as additional trees are added
predictions %>%
  as.data.frame() %>%
  mutate(
    observation = 1:n(),
    actual = ames_test$Sale_Price) %>%
  tidyr::gather(tree, predicted, -c(observation, actual)) %>%
  group_by(observation) %>%
  mutate(tree = stringr::str_extract(tree, '\\d+') %>% as.numeric()) %>%
  ungroup() %>%
  arrange(observation, tree) %>%
  group_by(observation) %>%
  mutate(avg_prediction = cummean(predicted)) %>%
  group_by(tree) %>%
  summarize(RMSE = RMSE(avg_prediction, actual)) %>%
  ggplot(aes(tree, RMSE)) +
  geom_line() +
  xlab('Number of trees')

# Stop Parallel Computing
stopCluster(cl)
unregister <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}
unregister()

### Using CARET Package for Feature Implementation
ames_bag2 <- train(
  Sale_Price ~ .,
  data = ames_train,
  method = "treebag",
  trControl = trainControl(method = "cv", number = 2),
  nbagg = 200,  
  control = rpart.control(minsplit = 2, cp = 0)
)
ames_bag2
### Feature interpretation
library(vip)
vip::vip(ames_bag2, num_features = 40, bar = FALSE)

# Construct partial dependence plots
library(pdp)
p1 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Area",
  grid.resolution = 20
) %>% 
  autoplot()

p2 <- pdp::partial(
  ames_bag2, 
  pred.var = "Lot_Frontage", 
  grid.resolution = 20
) %>% 
  autoplot()

gridExtra::grid.arrange(p1, p2, nrow = 1)

