# insurance-risk-ml

# Sure Tomorrow Insurance Company — Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat&logo=scikit-learn)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=flat&logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat&logo=pandas)

---

## About the Project

I developed this project to explore how machine learning and linear algebra can be applied to real insurance business problems. Working with customer data from a fictional insurance company called **Sure Tomorrow**, I built a complete pipeline that covers four different challenges: identifying similar customers, predicting insurance outcomes, forecasting payment amounts, and protecting sensitive data — all in a single, end-to-end notebook.

The goal was not just to apply ready-made models, but to understand the math behind them, build some from scratch, and prove that data privacy and model performance can coexist.

---

## Dataset

The dataset contains information about **5,000 insurance customers**:

| Column | Description |
|---|---|
| `gender` | Customer gender |
| `age` | Customer age |
| `income` | Annual salary |
| `family_members` | Number of family members |
| `insurance_benefits` | Insurance payments received in the last 5 years *(target)* |

---

## Part 1 — Finding Similar Customers

One of the most practical things a company can do is understand who their customers are and find patterns among them. I started by building a nearest neighbor search that, given any customer, returns the most similar ones based on their profile.

To make it robust, I tested four combinations of distance metrics (Euclidean and Manhattan) and data scaling (original and MaxAbsScaler). What I found was that unscaled data causes income to completely dominate the distance calculation, since it has much larger values than the other features. After scaling, all features contribute equally and the results become far more meaningful. Manhattan distance also proved to be more stable than Euclidean when the data is not scaled, since it is less sensitive to large values in a single feature.

---

## Part 2 — Predicting Whether a Customer Will Receive a Benefit

The next step was to turn the problem into a binary classification challenge: will this customer receive any insurance payment or not? I trained a KNN classifier and evaluated it across different values of k, comparing the F1 score on both original and scaled data. To have a reference point, I also built a dummy random model that predicts outcomes based purely on probability.

The KNN classifier consistently and clearly outperformed the random baseline, especially when the data was properly scaled. This confirms that there are real, learnable patterns in the customer profiles that the algorithm is able to capture.

---

## Part 3 — Forecasting the Number of Insurance Payments

Rather than just predicting whether a customer receives a payment, I went further and built a model to predict exactly how many payments they are likely to receive. Instead of using a ready-made implementation, I built the linear regression model entirely from scratch using NumPy and the closed-form analytical solution derived from linear algebra:

$$w = (X^T X)^{-1} X^T y$$

I evaluated the model using RMSE and R² on both original and scaled data. The results were identical in both cases, which is expected — linear scaling does not change the geometry of a linear regression problem.

---

## Part 4 — Protecting Customer Personal Data

With a working model in hand, I tackled one of the most important real-world concerns in data science: privacy. I developed a data obfuscation method that multiplies the feature matrix by a randomly generated invertible matrix P, making the original data unreadable without knowledge of P:

$$X' = X \times P$$

Before testing it computationally, I proved analytically that this transformation does not affect the predictions of a linear regression model. The math shows that the predictions remain exactly the same because the matrix and its inverse cancel each other out during the calculation. I then confirmed this experimentally — the RMSE and R² values were identical before and after obfuscation.

This means sensitive customer information can be protected without sacrificing any model performance whatsoever.

---

## Technologies Used

| Library | Purpose |
|---|---|
| `numpy` | Matrix operations and linear algebra |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | KNN, MaxAbsScaler, metrics |
| `seaborn` | Exploratory data visualization |

---

## Conclusions

This project showed me that machine learning is most powerful when you understand what is happening underneath the surface. Scaling had a dramatic effect on distance-based algorithms, a custom regression model built from first principles performed just as well as a library implementation, and a simple matrix multiplication was enough to protect personal data without losing any predictive quality.

The combination of these four solutions forms a complete, production-ready pipeline that balances performance, interpretability, and privacy — which is exactly what a real insurance company would need.
