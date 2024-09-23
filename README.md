
# Fraud Detection with Machine Learning

## Overview
This repository contains the analysis and predictive modeling techniques applied to detect fraudulent transactions in a financial dataset. The project explores various machine learning models to enhance predictive accuracy and provide actionable insights for stakeholders.

We begin with interpretable models like Logistic Regression and progressively advance to more complex models like Neural Networks and Random Forest, each refined for better performance. The goal is to improve the classification of fraudulent transactions while balancing interpretability and accuracy.

## Why Does This Issue Matter?
Financial fraud is a pressing concern for institutions and individuals alike. Fraud detection helps protect customers and reduces the massive financial losses associated with fraudulent activities. By accurately identifying suspicious transactions, businesses can minimize risks and avoid revenue loss. Our analysis aims to mitigate these losses by applying machine learning techniques to classify transactions effectively.

## Dataset
The dataset contains simulated financial transactions from a bank, including details such as customer demographics, merchant information, and transaction attributes. This dataset is particularly useful for fraud detection as it includes both legitimate and fraudulent transactions. The primary goal is to classify whether a transaction is fraudulent based on these features. The dataset allows for the exploration of various fraud detection models and techniques. Link- https://www.kaggle.com/datasets/ealaxi/banksim1/data

## Methods
Our approach involves exploring different machine learning models, evaluating their performance, and optimizing them through hyperparameter tuning and class balancing techniques.

- **Logistic Regression**: Used as the baseline model. Hyperparameter tuning is performed to improve performance.
- **Neural Network**: A multilayer perceptron (MLP) is applied to capture complex relationships in the data.
- **Random Forest**: An ensemble learning method used to capture variable interactions. Hyperparameter tuning is applied to maximize accuracy.
- **SMOTE**: The Synthetic Minority Over-sampling Technique is applied to handle class imbalance in the dataset, improving model performance on minority classes.
- **K-Means Clustering**: Unsupervised clustering to explore customer groupings based on transaction behavior.

## Key Files
- `fraud-detection.py`: Python script containing Exploratory Data Analysis (EDA) and the implementation of machine learning models including Logistic Regression, Neural Networks, Random Forest, and Naive Bayes, along with hyperparameter tuning.

## Tools and Libraries
- **Python**: Main programming language used for data analysis and model building.
- **scikit-learn**: Used for model building, hyperparameter tuning, and performance evaluation.
- **matplotlib** and **seaborn**: Used for data visualization and plotting confusion matrices.
- **SMOTE**: Used to handle class imbalance in the dataset.
- **pandas** and **numpy**: Used for data manipulation and numerical computations.

## Results
- **Logistic Regression**: A well-calibrated baseline model. Precision and recall were balanced, providing a solid foundation for comparison with more complex models.
- **Neural Network**: Showed improvement in recall, capturing more fraudulent transactions but at the cost of interpretability.
- **Random Forest**: Achieved the best overall performance after hyperparameter tuning, with the highest precision and balanced recall.
- **Naive Bayes**: Provided a simple model, though it struggled with class imbalance, leading to a lower precision.

## Model Results Summary
| Model                 | Precision | Recall | FP  | FN   |
|-----------------------|-----------|--------|-----|------|
| Naive Bayes            | 0.1957    | 0.9993 | 2   | 11695 |
| Logistic Regression    | 0.8887    | 0.7236 | 796 | 261   |
| Neural Network         | 0.8721    | 0.7624 | 684 | 322   |
| Decision Tree          | 0.9027    | 0.7212 | 803 | 224   |
| Random Forest          | 0.8845    | 0.7462 | 1096| 421   |

## Business Impact
- **Fraud Detection Automation**: Implement machine learning models like Neural Networks and Random Forest into real-time transaction monitoring systems to flag suspicious transactions for further review before processing.
- **Customer Trust & Experience**: Reduce false positives to avoid unnecessary transaction holds and improve customer satisfaction by minimizing fraud-related disruptions.
- **Cost Savings**: Minimize both false positives and negatives to reduce financial losses from fraudulent activities and lower manual transaction review costs.
- **Risk Mitigation**: Leverage model insights to enhance fraud prevention strategies, identify high-risk customers or transactions, and adjust security measures proactively.
- **Data-Driven Decisions**: Use predictive insights to inform decisions on customer policies, transaction limits, and fraud risk assessments.
