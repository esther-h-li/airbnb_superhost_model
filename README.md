# airbnb_superhost_model
Built a Logistic Regression classification model to predict Airbnb Superhost status. Optimized model hyperparameters using GridSearchCV, evaluated performance with ROC-AUC and precision-recall analysis, performed feature selection, and deployed the trained model as a persistent pickle file for future inference.

# Airbnb Superhost Prediction Model

This project uses a Logistic Regression machine learning model to predict whether an Airbnb host is a Superhost based on listing and host features.

The dataset was preprocessed with one-hot encoding, feature scaling, and missing value imputation before training. A baseline Logistic Regression model was evaluated and then optimized using GridSearchCV to find the best regularization parameter (`C`). The final model was evaluated using confusion matrices, precision-recall curves, ROC curves, and AUC scores.

## Model Workflow
- Loaded and prepared Airbnb listing data
- Defined `host_is_superhost` as the prediction target
- Split data into training and testing sets
- Trained a Logistic Regression model with default hyperparameters
- Used 5-fold cross-validation with GridSearchCV to optimize `C`
- Evaluated model performance using:
  - Confusion matrix
  - Precision-recall curve
  - ROC curve
  - AUC score
- Performed feature selection using SelectKBest
- Saved the final trained model using pickle for future predictions

## Model Persistence
The trained model is saved as: model.pkl


The saved model can be loaded and used for future predictions without retraining.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

## Model Type
**Algorithm:** Logistic Regression  
**Task:** Binary Classification  
**Target:** Predict whether an Airbnb host is a Superhost (`True`/`False`)
