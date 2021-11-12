# {Prediction Project} COVID-19 Hospitalizations in EU Countries
## Context:
The COVID-19 pandemic is not news to anyone at this point, and for the past year and then some, front-line workers have been working as best as they can to treat patients who have fallen ill due to the virus as well as to prevent further spread of the virus. One of the biggest problems early on in the pandemic was a shortage of hospital beds and supplies for the sick, so, in order to prevent that from happening again, it would be useful for hospital workers to know how many patients to expect in the next week so that they might be able to prepare accordingly.

## Problem: 
To predict whether COVID-19 cases among different EU countries would increase or decrease in the next week
## Problem Type: 
Binary classification

## Why this project is being done (aka my intentions & goals for this project):
- This was part of a final project I completed for an Intro to Machine Learning (CS 4780) course I took at Cornell University during my undergrad junior year spring semester in 2021. 
- I am hoping to revise and polish this project to make it readable and presentable in order to improve my project documentation skills and perhaps also substantially add to the project and its data-driven insights.

# Table of Contents

1. [File Descriptions](#file_descriptions)
2. [Technologies Used](#technologies_used)
3. [Project Structure](#project_structure)
4. [Executive Summary](#executive_summary)
    - [4.1. Build Baseline Model Using All Features](#baseline_model)
    - [4.2. Build Additional Model to Try to Improve Upon Baseline Model](#additional_model)
    - [4.3. Repeat Step 4.2 to continue to build additional models and use optimal model out of all created models](#repeat)

## <a name="file_descriptions"></a>File Descriptions:
- [Data](/data): folder containing all data files
  - [test_baseline_no_label.csv](data/test_baseline_no_label.csv): raw test data with x values
  - [train_baseline.csv](data/train_baseline.csv): raw training data with x and y values
- [.gitignore](/.gitignore): this file lists the files I'd like git to ignore when making new commits
- [COVID-19.ipynb](/COVID-19.ipynb): notebook with data pre-processing, model brainstorming, model training, model validation, model selection, and model assessment

## <a name="technologies_used"></a>Technologies Used:
TODO

## <a name="project_structure"></a>Project Structure:
TODO

## <a name="executive_summary"></a>Executive Summary:

### <a name="baseline_model"></a>STEP 1 : Build Baseline Model Using All Features
#### (1.1) Data Pre-Processing: 
  - Check for missing values (aka NULL's); if there are missing values, do something about it **_(Q: what methods can I use to deal with missing values?)_**
  - Separate numerical and categorical features in given dataset
  - Transform categorical features into numerical features using one hot encoding
  - Standardize ONLY numerical features in given dataset (because categorical features cannot be standardized) **_(Q: can i standardize numerical features resulting     from one hot encoding?)_**
  - _Note: Be sure to clearly label dataset with one-hot-encoded categorical features and standardized numerical features as "STANDARDIZED DATA" for future use_
  - Choose validation method:
    - If I want to give a definitive statement about the prediction error (in other words, the acucuracy or performance) of the resulting model and/or I don't need       to optimize hyperparameters of the resulting model, choose train-test split validation method aka hold-out validation (usually 80/20 or 70/30 split)
    - If I want to optimize hyperparameters of the resulting model (and don't need to give any kind of definitive statement about the prediction error of the             model), choose k-fold cross validation or leave-one-out cross validation
  - If I chose to use the train-test split validation method, then randomly split the STANDARDIZED DATA into a training set and validation set (usually 80/20 or         70/30 split)
#### (1.2) Baseline Model Brainstorming: 
  - If this is a **single-feature regression** problem: choose **simple linear regression** 
  - If this is a **multi-feature regression** problem: choose **multiple linear regression**
  - If this is a **binary classification** problem: choose **logistic regression**
  - If this is a **multi-class classification** problem: can try **KNN, decision trees, SVM** to name a few
#### (1.3) Model Training:
  - Train baseline model using all standardized numerical features (and maybe also categorical features if possible?) from STANDARDIZED DATA
#### (1.4) Model Validation: 
  - Validate the resulting baseline model using the validation method I chose in Step (1.1) to produce the appropriate model performance measure --
    - If this is a **classification** problem, evaluate model performance using the **accuracy score** of the model on the validation data
    - If this is a **regression** problem, evaluate model performance using the **MSE** of the model on the validation data
  - Note: If the model uses hyperparameters, find the optimal value for each hyperparameter in this Step (1.4) and use the resulting optimal model and its               appropriate performance measure value
  
### <a name="additional_model"></a>STEP 2 : Build Additional Model to Try to Improve Upon Baseline Model
#### (2.1) Complex Model Brainstorming: 
  - If this is a **linear regression** problem, can try:
    - Ridge Regression
    - Lasso Regression
  - If this is a **non-linear regression** problem, can try:
    - Neural Network Regression
    - Decision Tree Regression
    - Random Forest
    - K-Nearest Neighbors
    - Support Vector Machines
  - If this is a **classification** problem, can try:
    - Discriminative Models:
      - K-Nearest Neighbors
      - Decision Trees
      - Random Forest
      - Support Vector Machines (hard-margin/soft-margin)
      - Neural Networks
    - Generative Models:
      - Naive Bayes
      - Linear Discriminant Analysis / Quadratic Discriminant Analysis (for unsupervised learning)
  - Note: Can also try these following methods to add onto performance of above models --
    - If the data are nonlinear, can try using kernels
    - If using decision trees or random forests, can try using boosting techniques (e.g. AdaBoost) (?)
#### (2.2) Feature Selection & Feature Transformation Brainstorming: 
  - Feature Selection Methods: https://www.google.com/search?q=how+to+do+feature+selection&oq=how+to+do+feature+sel&aqs=chrome.0.0l2j69i57j0l7.2140j0j1&sourceid=chrome&ie=UTF-8
  - Feature Transformation Methods: https://www.google.com/search?q=feature+transformation+methods&oq=feature+transformation+methods&aqs=chrome..69i57j0l4j0i22i30l5.5245j0j9&sourceid=chrome&ie=UTF-8
#### (2.3) Choose Method to Try:
  - Modify Baseline Model to use additional transformed features
  - Modify Baseline Model to use a selected set of features (among original and transformed features)
  - Choose a complex model and train this model with all features
  - Choose a complex model and train this model with a selected set of features (among original and transformed features)
#### (2.4) Data Pre-Processing:
  - Building upon STANDARDIZED DATA from (1.1), implement transformed features that I brainstormed and chose to try in (2.2) --> call this "TRANSFORMED DATA"
  - Choose validation method:
    - If I want to give a definitive statement about the prediction error (in other words, the acucuracy or performance) of the resulting model and/or I don't need       to optimize hyperparameters of the resulting model, choose train-test split validation method aka hold-out validation (usually 80/20 or 70/30 split)
    - If I want to optimize hyperparameters of the resulting model (and don't need to give any kind of definitive statement about the prediction error of the             model), choose k-fold cross validation or leave-one-out cross validation
  - If I chose to use the train-test split validation method, then randomly split the TRANSFORMED DATA into a training set and validation set (usually 80/20 or         70/30 split)
#### (2.5) Model Training:
  - Train the selected model using either all features or a selected set of features from TRANSFORMED DATA, depending on what method I chose to try in (2.3)
#### (2.6) Model Validation: 
  - Validate the resulting model using the validation method I chose in (2.4) to produce the appropriate model performance measure --
    - If this is a **classification** problem, evaluate model performance using the **accuracy score** of the model on the validation data
    - If this is a **regression** problem, evaluate model performance using the **MSE** of the model on the validation data
  - Note: If the model uses hyperparameters, find the optimal value for each hyperparameter in this (2.5) and use the resulting optimal model and its               appropriate performance measure value

### <a name="repeat"></a>STEP 3 : Repeat STEP 2 to continue to build additional models and use optimal model out of all created models


## Some visualizations illustrating key findings of this project:

