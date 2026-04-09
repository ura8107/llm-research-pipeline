# Beginner-friendly tidymodels tutorial
# Goal: predict flower species in the iris dataset
#
# This script is written as a step-by-step introduction to machine learning
# in R using the tidymodels ecosystem.


# -----------------------------------------------------------------------------
# 1. Install and load packages
# -----------------------------------------------------------------------------

# Uncomment this line the first time you run the script.
# install.packages(c("tidymodels", "kknn"))

library(tidymodels)

# tidymodels prints a lot of useful package startup messages for beginners.
tidymodels_prefer()


# -----------------------------------------------------------------------------
# 2. Reproducibility
# -----------------------------------------------------------------------------

# Setting a seed makes your train/test split and resampling reproducible.
set.seed(123)


# -----------------------------------------------------------------------------
# 3. Load and inspect the data
# -----------------------------------------------------------------------------

# We use the built-in iris dataset, so the script runs without downloading data.
data(iris)

# View the first few rows.
head(iris)

# Check the structure of the dataset.
str(iris)

# Check the target variable.
count(iris, Species)


# -----------------------------------------------------------------------------
# 4. Split the data into training and testing sets
# -----------------------------------------------------------------------------

# Training data is used to learn patterns.
# Testing data is used only at the end to evaluate performance.
iris_split <- initial_split(iris, prop = 0.8, strata = Species)

iris_train <- training(iris_split)
iris_test  <- testing(iris_split)

nrow(iris_train)
nrow(iris_test)


# -----------------------------------------------------------------------------
# 5. Create a recipe
# -----------------------------------------------------------------------------

# A recipe defines preprocessing steps.
# Here we normalize numeric predictors so they are on a similar scale.
iris_recipe <- recipe(Species ~ ., data = iris_train) |>
  step_normalize(all_numeric_predictors())

iris_recipe


# -----------------------------------------------------------------------------
# 6. Define a model
# -----------------------------------------------------------------------------

# We use k-nearest neighbors (k-NN), a classic beginner-friendly ML model.
# The value of "neighbors" controls how many nearby points vote on the class.
#
# We set neighbors = tune() because we want the computer to try several values
# and choose the best one with cross-validation.
knn_model <- nearest_neighbor(neighbors = tune()) |>
  set_mode("classification") |>
  set_engine("kknn")

knn_model


# -----------------------------------------------------------------------------
# 7. Combine recipe + model into a workflow
# -----------------------------------------------------------------------------

knn_workflow <- workflow() |>
  add_recipe(iris_recipe) |>
  add_model(knn_model)

knn_workflow


# -----------------------------------------------------------------------------
# 8. Create cross-validation folds
# -----------------------------------------------------------------------------

# Cross-validation gives a more reliable estimate than a single train/validation
# split because the model is tested on several different subsets of the data.
iris_folds <- vfold_cv(iris_train, v = 5, strata = Species)

iris_folds


# -----------------------------------------------------------------------------
# 9. Tune the model
# -----------------------------------------------------------------------------

# We try several k values and compare their accuracy.
knn_grid <- tibble(neighbors = c(1, 3, 5, 7, 9, 11, 15))

knn_tuned <- tune_grid(
  knn_workflow,
  resamples = iris_folds,
  grid = knn_grid,
  metrics = metric_set(accuracy)
)

knn_tuned

# View the tuning results.
collect_metrics(knn_tuned)


# -----------------------------------------------------------------------------
# 10. Choose the best hyperparameter
# -----------------------------------------------------------------------------

best_k <- select_best(knn_tuned, metric = "accuracy")
best_k


# -----------------------------------------------------------------------------
# 11. Finalize the workflow with the best value
# -----------------------------------------------------------------------------

final_knn_workflow <- finalize_workflow(knn_workflow, best_k)

final_knn_workflow


# -----------------------------------------------------------------------------
# 12. Fit the final model on the training data and evaluate on the test data
# -----------------------------------------------------------------------------

final_fit <- last_fit(
  final_knn_workflow,
  split = iris_split,
  metrics = metric_set(accuracy)
)

# Overall test performance.
collect_metrics(final_fit)

# Predicted values for each test observation.
test_predictions <- collect_predictions(final_fit)
head(test_predictions)


# -----------------------------------------------------------------------------
# 13. Confusion matrix
# -----------------------------------------------------------------------------

# A confusion matrix shows which classes are predicted correctly or incorrectly.
conf_mat(test_predictions, truth = Species, estimate = .pred_class)


# -----------------------------------------------------------------------------
# 14. Fit the final model to the training data for future predictions
# -----------------------------------------------------------------------------

# last_fit() is great for evaluation, but if you want to predict on new data
# later, it is helpful to fit the finalized workflow directly.
trained_knn_model <- fit(final_knn_workflow, data = iris_train)


# -----------------------------------------------------------------------------
# 15. Predict on new observations
# -----------------------------------------------------------------------------

# These are made-up flowers.
new_flowers <- tibble(
  Sepal.Length = c(5.1, 6.0, 6.9),
  Sepal.Width  = c(3.5, 2.9, 3.1),
  Petal.Length = c(1.4, 4.5, 5.4),
  Petal.Width  = c(0.2, 1.5, 2.1)
)

# Predicted class
predict(trained_knn_model, new_data = new_flowers)

# Predicted class probabilities
predict(trained_knn_model, new_data = new_flowers, type = "prob")


# -----------------------------------------------------------------------------
# 16. Key ideas to remember
# -----------------------------------------------------------------------------

# 1. recipe() handles preprocessing.
# 2. parsnip model specifications define the model.
# 3. workflow() combines preprocessing and modeling.
# 4. vfold_cv() creates resamples for cross-validation.
# 5. tune_grid() searches for good hyperparameters.
# 6. last_fit() gives an honest final test evaluation.


# -----------------------------------------------------------------------------
# 17. Optional next steps for learners
# -----------------------------------------------------------------------------

# Try changing the grid of neighbors values.
# Try a different model such as decision_tree() or rand_forest().
# Try plotting the tuning results:
#
# autoplot(knn_tuned)
