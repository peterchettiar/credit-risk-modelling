# credit-risk-modelling
_Objective_: To build an ML classification model using a credit risk dataset from Kaggle. Outcome is to predict the 'loan status' type from a given set of inputs. There are a few types of loan statuses, hence a multi-class algorithm needs to be used to predict which class the target variable belongs to.

> Quick note: The [kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data) dataset is larger than the size limit prescribed by Github and hence unable to upload. But feel free to download it directly by clicking [here](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/download?datasetVersionNumber=3).

## Introduction

I guess a great place to start would be the technical architecture of the project. The flow of the project is as follows:

![credit-risk-modelling](https://github.com/peterchettiar/credit-risk-modelling/assets/89821181/e6dd9079-9f2b-406a-915c-f2c06555cd60)

## Exploratory Data Analysis

An important first step would be to have a good understanding of the dataset. So typically there are a few things I would like to look at:
1. The proportion of **missing values** for each feature - depending on the size of the dataset, you can either choose to drop the rows with the missing values or impute based on the distribution of the data for the feature in question without having to compromise on accuracy while training the model (i.e. model must have enough data to be trained on)
2. I would also normally look for **outliers** to prevent results from being skewed, but for our case I did not do that to get a better understanding of the big picture (sometimes outliers are seen as a mistake, but I carried on to see if there is some influence of the final result) - typically people use either the turkey method (IQR) or standard deviation approah to identify outliers (I prefer the latter where you identify values that is beyound three SD from the mean as outliers, its just simpler!)
3. Next, would be to understand the **data types** as well as to plot out the **distributions** - commonly dates are read as strings and we need to convert them to date time objects, subsequently we would need to identify the features as being either numerical or categorical and do some light cleaning for consistency (useful for column transforamation down the line)

## Data Cleaning and Feature Selection

Now that we understood the data, the next step would be preprocessing. And in the spirit of cleaning, we would also want to reduce the number of features that we have, 74 columns is a little too much for our task. In this dataset we removed columns that had over 97% missing values, then dropping rows that have missing values and lastly converting date strings to date time object.

Moving forward, after the basic cleaning we take a few steps to reduce the number of columns through feature selection:
1. Removing Correlated variables
2. Apply a Low Variance Filter - remove columns with variance of less than 10
3. Now to apply a random forest regressor to help identify the 10 most useful features out of the columns that remain

## Feature Engineering

Conventionally this would be a good stage to add some new features using the existing features but given the simplicity of the task I omitted it as the final model accuracy was really high. Hence, we can move on from this section.

## Model Selection with Cross-Validation

Now to get into the crux of the project. We have our 10 features selected through the feature importance attribute of the random forest regressor algorithm, so let's re-slice our original dataset to only keep the features that we need and define column transformers for them.

The first step would be to create Pipeline objects that contain the steps for transformation, I've only included imputation and scaling as steps. For our dataset, I've created three Pipeline objects to deal with numeric or categorical columns, and to transform the target column seperately.

For feature transform, we can combine our Pipeline objects into one `ColumnTransformer` object.

At this point, since we have done the necessary transformations, we can do a train-test split to enable model building. Next, I selected the following algorithms because they are the most popular ones when dealing with multi-class classification:

| Model Abbreviation | Algorithm |
| ------------------ | --------- |
| 'knn' | KNeighboursClassifier() |     
| 'rfc' | RandomForestClassifier() |
| 'dtc' | DecisionTreeClassifier() |
| 'nbc' | GaussianNB() |
| 'gbc' | GradientBoostingClassifier() |

Next, I use a `cross-validation predict` to do a 5-fold `StratifiedKfold` of each algorithm and measure their performance using `roc_auc_score`. It should be noted that given the target classes are imbalanced, we need to specify the `average` parameter as 'weighted' as that is how the mean accuracy should be calculated.

A sample of how the model selection was done is as follows:
```
# cv argument is left at default for 2 reasons:
# 1. default cross validation is 5
# 2 if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used automatically - perfect
# for imbalanced labels

def model_validation(model, X, y):

    y_pred_proba = cross_val_predict(model, X, y, method='predict_proba')

    return roc_auc_score(y, y_pred_proba, average='weighted', multi_class='ovr')

score = 0.0
best_score = 0.0
best_model = None

for model in tqdm(parameters.multiclass_algos_list):
    score = model_validation(parameters.multiclass_algos_dict[model],
                             X_train,
                             y_train)
    if score > best_score:
        best_score = score
        best_model = model
```
And from this we realise that `GradientBoostingClassifier()` performed the best with a roc-auc-score of 98%.

## Model Pipeline and HyperParameter Tuning

Now comes the last part of the project. Our model selection gave us the best performing model but it should be noted that the parameters were not specifed, and that the default values were used. 

This is where hyperparameter tuning comes in. I decided to experiment with three parameters: `n-estimators`, `learning_rate`, `max-depth`, and do a `RandomisedGridSearch` on a specified range not too far from the default values given the performance on default parameters were pretty good.

A sample of the code for this portion is as follows:
```
# next we need to define our parameter grid
# since the default values performed pretty well, our grid search should not be too far off from the default

parameters.param_grid = {
    'best_model__n_estimators': randint(90,110), # default = 100
    'best_model__learning_rate': uniform(0.08,0.12), # default = 0.1
    'best_model__max_depth': randint(1,5) # default = 3
}

# now that we have our model and parameter grid, we can perform the grid search
# since the default parameters performed pretty well, 

best_model_tuned = RandomizedSearchCV(best_model_pipeline,
                                      param_distributions=parameters.param_grid,
                                      scoring='roc_auc_ovr_weighted',
                                      n_iter=2,
                                      random_state=42)

best_model_tuned.fit(X_train, y_train)
```

Now that we have trained the model, we can make predictions on the test set. And after comparing the predictions with the actual test target, we find that the model was 95.79% accurate. That concludes the project.
