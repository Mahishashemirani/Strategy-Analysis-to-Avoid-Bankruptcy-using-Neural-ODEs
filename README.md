# Strategy Analysis to Avoid Bankruptcy Using Neural ODEs

## Overview

Predicting a company's likelihood of bankruptcy is crucial for decision-makers who aim to preemptively address financial distress and evaluate strategies that could prevent financial failure. This repository presents a pipeline leveraging **Neural Ordinary Differential Equations (Neural ODEs)** to model company financial trends and assess whether certain strategies can help avoid bankruptcy.

## What are Neural ODEs?

**Neural ODEs** are a powerful method for modeling systems with continuous dynamics, useful in applications where data evolves over time. Unlike traditional neural networks with fixed discrete layers, Neural ODEs treat the network as a continuous function parameterized by differential equations. By learning an ODE that represents the time evolution of a company’s financial features, Neural ODEs enable us to forecast trajectories based on initial conditions and analyze potential future states under various strategic scenarios.

## Dataset

The dataset used for bankruptcy prediction is available on [Kaggle](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction). This dataset, collected by the Taiwan Economic Journal, spans from **1999 to 2009** and includes financial data on companies listed on the Taiwan Stock Exchange. Bankruptcy status in the data aligns with Taiwan Stock Exchange regulations, which allows for a realistic and regulated definition of bankruptcy.

### Dataset Features

The dataset consists of **95 financial features** for each company, covering various aspects of financial health, such as:
- **After-tax Net Interest Rate**
- **Cash Flow Rate**
- **Debt Ratio**
- **Working Capital to Total Assets**

In addition to these, a **bankruptcy state** column labels each company as either "bankrupt" or "not bankrupt."

## Feature Selection

To improve model efficiency and focus on the most impactful variables, we performed **feature selection** using **[Lasso Logistic Regression](https://en.wikipedia.org/wiki/Lasso_(statistics))**. This method penalizes less important features, effectively filtering out those with insignificant predictive power. As a result, we reduced the feature set from **95** to **69 key features**.

## Data Augmentation for Imbalanced Classes

The dataset contains a much smaller number of bankrupt companies compared to non-bankrupt ones, which could skew model training toward non-bankrupt predictions. To address this, we applied **data augmentation** to increase the number of bankrupt samples and create a more balanced dataset.

1. **Normalization**: All features were normalized to standardize values across companies and improve model training stability.
2. **Augmentation Process**: We calculated the **mean and variance** of each feature within the bankrupt category. Using these statistics, we generated synthetic data by sampling from a **Gaussian distribution** that reflects the feature distributions of bankrupt companies. This approach effectively enhances the representation of bankrupt companies in the dataset, improving model robustness for bankruptcy prediction.

