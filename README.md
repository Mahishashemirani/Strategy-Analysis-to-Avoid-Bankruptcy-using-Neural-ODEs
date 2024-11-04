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

# Model

## Encoding Bankruptcy State with One-Hot Categorical Distribution

To represent bankruptcy state, we use a **One-Hot Categorical Distribution** from Pyro’s [distribution library](https://docs.pyro.ai/en/stable/distributions.html#onehotcategorical). This approach represents the bankruptcy state as:
- `[1, 0]` for companies that are not bankrupt
- `[0, 1]` for companies that are bankrupt

This categorical representation allows us to easily map the binary outcome into a probabilistic framework for classification.

## Latent Variable Representation with Neural Network Encoder

A neural network encoder is used to reduce the high-dimensional feature space into a low-dimensional latent variable $z$. The encoder maps each company’s financial features to a 2D latent variable space, $z$, which serves as input to the Neural ODE model.

In mathematical terms, the model is structured as:
- $y$ ~ OneHotCategorical($p$): Defines the categorical output for bankruptcy prediction, where $p$ is the predicted probability vector.
- $z$ ~ $N(\mu(x), \sigma(x))$: A 2D latent variable sampled from a Normal distribution, where $\mu(x)$ and $\sigma(x)$ are outputs of the encoder network, parameterized by the input features $x$.

This reduction to a 2D space balances interpretability and efficiency while still capturing essential features for bankruptcy prediction.

## Defining the Neural ODE for Bankruptcy Probability Prediction

Given that we have two classes, $p$ is a 2D vector, representing the probability of bankruptcy. We define an **autonomous Neural ODE** as follows:
- Assume $u$ is a 2D dynamical variable governed by the neural ODE $h(u)$.
- Then, $\frac{du(t)}{dt} = h(u)$, with initial condition $u(0) = z$ and output $u(1) = p$.

This formulation allows the Neural ODE to take the latent variable $z$ as input and predict the probability of bankruptcy $p$ over time. Thus, the Neural ODE functions as a predictive model that evolves $z$ to output $p$, which directly represents bankruptcy likelihood.

## Implementation

We used **Pyro** for implementing the probabilistic encoding and prediction framework ([Pyro Documentation](https://pyro.ai)) and **torchdiffeq** for Neural ODE implementation ([torchdiffeq Repository](https://github.com/rtqichen/torchdiffeq)).

### ELBO Loss and Stochastic Variational Inference

The model’s loss function is based on the **Evidence Lower Bound (ELBO)**, optimized using **Stochastic Variational Inference (SVI)**. This approach helps estimate the posterior distribution of the latent variables and minimizes reconstruction errors by maximizing the ELBO.

### Training Details

- **Optimizer**: Adam
- **Learning Rate**: `1e-3`
- **Epochs**: 100




