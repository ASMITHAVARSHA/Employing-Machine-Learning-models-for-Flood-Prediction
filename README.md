# Employing-Machine-Learning-models-for-Flood-Prediction

# Objective
To develop a predictive model for floods using historical rainfall data from Kerala (1901–2018), identifying patterns in monthly and annual rainfall to classify flood events.

# Dataset
- **Source:** Kaggle
- **Features:** Rainfall data for each month, annual rainfall, and flood occurrence (FLOODS as YES or NO)
- **Size:** 118 rows, 16 columns

# Pre-processing
- Categorical target FLOODS encoded as binary (**1** for **YES**, **0** for **NO**)
- Standardized the features using MinMaxScaler.

# Data Visualization
- **Histograms:** Distribution of rainfall across months.
- **Bar Plot:** Mean rainfall for each month.
- **Count Plot:** Distribution of flood occurrences (imbalanced dataset observed).

# Model Implementation
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors
- Random Forest Classifier
- Support Vector Machine
- Ensemble Voting Classifier (combining Logistic Regression, Random Forest, and K-Nearest Neighbors)

# Cross Validation
Cross-validation was applied to assess model performance on unseen data, ensuring robust evaluation metrics.

# Evaluation Metrics
- **Accuracy:** Measures the percentage of correct predictions.
- **Recall:** Indicates the model's ability to identify positive cases.
- **ROC AUC Score:** Evaluates the trade-off between sensitivity and specificity.

# Results
- **Best Model:** Random Forest Classifier with hyperparameter tuning.
- Each model was trained, and its performance was measured using the above metrics to determine the most effective model for the task.

# Libraries
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit learn
