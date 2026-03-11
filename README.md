# insurance-risk-ml

Sure Tomorrow Insurance Company — Machine Learning Project
Overview
This project was developed for the fictional insurance company Sure Tomorrow, with the goal of evaluating the feasibility of applying machine learning to four real-world business problems: customer similarity search, insurance benefit prediction, benefit amount forecasting, and personal data protection through obfuscation.
The entire pipeline was built using Python, with implementations ranging from ready-made scikit-learn models to custom-built algorithms from scratch using linear algebra.

Dataset
File: insurance_us.csv
FeatureDescriptiongenderCustomer genderageCustomer ageincomeCustomer annual salaryfamily_membersNumber of family membersinsurance_benefitsNumber of insurance payments received in the last 5 years (target)

Total records: 5,000 customers
Missing values: None
Data types: Fixed during preprocessing (age converted from float to int)


Project Structure
├── insurance_us.csv
├── notebook.ipynb
└── README.md

Task 1 — Finding Similar Customers (KNN)
Objective
Develop a procedure to find the k nearest neighbors for a given customer based on distance metrics. This helps marketing agents identify customers with similar profiles.
Approach

Implemented a get_knn() function using sklearn.neighbors.NearestNeighbors
Tested 4 combinations of scaling and distance metrics:

Unscaled + Euclidean distance
Unscaled + Manhattan distance
Scaled (MaxAbsScaler) + Euclidean distance
Scaled (MaxAbsScaler) + Manhattan distance



Key Findings
Do unscaled data affect the KNN algorithm?
Yes. When features are not on the same scale, variables with larger values — such as income — dominate the distance calculation. This means the algorithm finds "similar" customers based almost entirely on income, while age, gender, and family members contribute very little. After scaling with MaxAbsScaler, all features contribute equally to the distance, producing more balanced and meaningful results.
How similar are the results using the Manhattan distance metric regardless of scaling?
The results obtained with Manhattan distance are relatively stable regardless of whether the data is scaled or not. Because Manhattan distance computes the sum of absolute differences rather than squaring them, it is less sensitive to large values in a single feature. This makes it more robust than Euclidean distance in unscaled scenarios.

Task 2 — Insurance Benefit Prediction (Binary Classification)
Objective
Predict whether a new customer is likely to receive at least one insurance payment. Evaluate whether a KNN classifier outperforms a dummy (random) model.
Approach

Created a binary target variable: insurance_benefits_received = 1 if insurance_benefits > 0, else 0
Evaluated KNN classifier for k = 1 to 10 on both original and scaled data using the F1 score
Built a dummy model with four probability settings:

P = 0 (always predicts no payment)
P = P(any payment) (uses true class frequency)
P = 0.5 (random 50/50)
P = 1 (always predicts payment)


Split: 70% training / 30% test

Key Findings

The KNN classifier consistently outperforms the dummy model across most values of k, particularly when data is scaled
Scaling had a significant positive impact on classification quality, confirming the findings from Task 1
The dummy model with P = 0 achieved F1 = 0.00, and with P = 1 it produced very low F1 due to poor precision, confirming that the KNN approach adds real predictive value


Task 3 — Predicting Number of Insurance Payments (Linear Regression)
Objective
Predict the number of insurance payments a customer is likely to receive using a custom linear regression model built from scratch.
Approach

Implemented MyLinearRegression class using the analytical (closed-form) solution:

w=(XTX)−1XTyw = (X^T X)^{-1} X^T yw=(XTX)−1XTy

Added a bias term (column of ones) to the feature matrix before training
Evaluated using RMSE (Root Mean Squared Error) and R²
Split: 70% training / 30% test
Tested on both original and scaled data

Key Findings

The RMSE values for original and scaled data were identical, confirming that linear scaling does not change the predictive quality of a linear regression model
The custom implementation produced results consistent with scikit-learn's LinearRegression, validating the mathematical approach


Task 4 — Data Obfuscation (Privacy Protection)
Objective
Protect customer personal data by applying a mathematical transformation that makes it difficult to recover the original information, while ensuring that machine learning model quality is not affected.
Approach

Generated a random invertible square matrix P
Verified invertibility by checking that det(P) ≠ 0
Transformed the feature matrix: X' = X @ P
Trained a linear regression model on both original and obfuscated data
Compared RMSE and R² values between the two models

Analytical Proof
The obfuscation does not affect the predictions of linear regression. Given the transformation X' = XP, the new weights are:
wP=[(XP)T(XP)]−1(XP)Ty=P−1(XTX)−1XTy=P−1ww_P = [(XP)^T(XP)]^{-1}(XP)^T y = P^{-1}(X^TX)^{-1}X^Ty = P^{-1}wwP​=[(XP)T(XP)]−1(XP)Ty=P−1(XTX)−1XTy=P−1w
The predicted values remain the same:
y^P=XP⋅wP=XP⋅P−1w=Xw=y^\hat{y}_P = XP \cdot w_P = XP \cdot P^{-1}w = Xw = \hat{y}y^​P​=XP⋅wP​=XP⋅P−1w=Xw=y^​
Since the predictions are identical, the RMSE and R² are also unchanged.
Key Findings

RMSE and R² were identical for original and obfuscated data (verified computationally)
The original data could be recovered using X_recovered = X_obfuscated @ P_inv, confirming the transformation is mathematically reversible
The obfuscation method is effective: the transformed data is unreadable without knowledge of matrix P, while the model retains full predictive power


Technologies Used
LibraryPurposenumpyMatrix operations, linear algebrapandasData manipulationscikit-learnKNN, scaling, metricsseabornExploratory visualizationmatplotlibPlotting support

Final Conclusions
All four tasks were successfully completed and validated:

Customer similarity can be effectively identified using KNN — scaling is essential for meaningful results with Euclidean distance, while Manhattan distance offers more stability without scaling.
Insurance benefit classification with KNN significantly outperforms a random dummy model, especially when data is properly scaled. This approach can support targeted marketing and risk assessment.
Benefit amount forecasting is achievable with a custom linear regression built entirely from linear algebra principles. The model performs consistently on both scaled and unscaled data.
Data obfuscation via invertible matrix multiplication is both mathematically proven and computationally verified to preserve model quality. This technique provides a practical and effective way to protect customer personal information without any loss in predictive performance.


Decision Recommendations
Based on the results of this project, the following actions are recommended for the Sure Tomorrow insurance company:

Deploy KNN-based customer similarity search as a marketing support tool. Ensure that data is always scaled before running the algorithm to guarantee consistent and meaningful neighbor identification.
Adopt the KNN classifier for benefit prediction as a baseline model. Its performance clearly exceeds random guessing, making it a reliable starting point for production. Further improvements could be explored with ensemble methods or gradient boosting.
Use the custom or scikit-learn linear regression for predicting the number of benefit payments. The model is interpretable and performs well on this dataset.
Implement data obfuscation before storing or transmitting customer data. The matrix multiplication approach is simple, reversible for authorized parties, and proven not to degrade model performance. The matrix P should be stored securely and treated as a private key.
Prioritize data scaling in all future ML pipelines involving distance-based algorithms, as demonstrated by its significant impact on KNN performance throughout this project.
