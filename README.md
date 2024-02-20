# credit-risk-modelling
_Objective_: To build an ML classification model using a credit risk dataset from Kaggle. Outcome is to predict the 'loan status' type from a given set of inputs. There are a few types of loan statuses, hence a multi-class algorithm needs to be used to predict which class the target variable belongs to.

> Quick note: The [kaggle](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/data) dataset is larger than the size limit prescribed by Github and hence unable to upload. But feel free to download it directly by clicking [here](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset/download?datasetVersionNumber=3).

# Introduction

I guess a great place to start would be the technical architecture of the project. The flow of the project is as follows:

![credit-risk-modelling](https://github.com/peterchettiar/credit-risk-modelling/assets/89821181/e6dd9079-9f2b-406a-915c-f2c06555cd60)

# EDA

An important first step would be to have a good understanding of the dataset. So typically there are a few things I would like to look at:
1. The proportion of **missing values** for each feature - depending on the size of the dataset, you can either choose to drop the rows with the missing values or impute based on the distribution of the data for the feature in question without having to compromise on accuracy while training the model (i.e. model must have enough data to be trained on)
2. I would also normally look for **outliers** to prevent results from being skewed, but for our case I did not do that to get a better understanding of the big picture (sometimes outliers are seen as a mistake, but I carried on to see if there is some influence of the final result) - typically people use either the turkey method (IQR) or standard deviation approah to identify outliers (I prefer the latter where you identify values that is beyound three SD from the mean as outliers, its just simpler!)
3. Next, would be to understand the **data types** as well as to plot out the **distributions** - commonly dates are read as strings and we need to convert them to date time objects, subsequently we would need to identify the features as being either numerical or categorical and do some light cleaning for consistency (useful for column transforamation down the line)

# Data Cleaning and Feature Selection

Now that we understood the data, the next step would be preprocessing. And in the spirit of cleaning, we would also want to reduce the number of features that we have, 74 columns is a little too much for our task. In this dataset we removed columns that had over 97% missing values, then dropping rows that have missing values and lastly converting date strings to date time object.

Moving forward, after the basic cleaning we take a few steps to reduce the number of columns through feature selection:
1. Removing Correlated variables
2. Apply a Low Variance Filter - remove columns with variance of less than 10
3. Now to apply a random forest regressor to help identify the 10 most useful features out of the columns that remain

# Feature Engineering

Conventionally this would be a good stage to add some new features using the existing features but given the simplicity of the task I omitted it as the final model accuracy was really high. Hence, we can move on from this section.

