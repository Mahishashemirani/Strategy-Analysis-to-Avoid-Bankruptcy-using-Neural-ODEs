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

To improve model efficiency and focus on the most impactful variables, I performed **feature selection** using **[Lasso Logistic Regression](https://en.wikipedia.org/wiki/Lasso_(statistics))**. This method penalizes less important features, effectively filtering out those with insignificant predictive power. As a result, I reduced the feature set from **95** to **69 key features**.

## Data Augmentation for Imbalanced Classes

The dataset contains a much smaller number of bankrupt companies compared to non-bankrupt ones, which could skew model training toward non-bankrupt predictions. To address this, I applied **data augmentation** to increase the number of bankrupt samples and create a more balanced dataset.

1. **Normalization**: All features were normalized to standardize values across companies and improve model training stability.
2. **Augmentation Process**: I calculated the **mean and variance** of each feature within the bankrupt category. Using these statistics, I generated synthetic data by sampling from a **Gaussian distribution** that reflects the feature distributions of bankrupt companies. This approach effectively enhances the representation of bankrupt companies in the dataset, improving model robustness for bankruptcy prediction.

# Model

## Encoding Bankruptcy State with One-Hot Categorical Distribution

To represent bankruptcy state, I use a **One-Hot Categorical Distribution** from Pyro’s [distribution library](https://docs.pyro.ai/en/stable/distributions.html#onehotcategorical). This approach represents the bankruptcy state as:
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

Given the two classes, $p$ is a 2D vector, representing the probability of bankruptcy. I define an **autonomous Neural ODE** as follows:
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

The following plots illustrates the maximization of the ELBO over the course of training, as well as accuracy of the model in predicting bankruptcy:  

![ELBO](plots/ELBO.png)

![Accuracy](plots/Accuracy.png)

# Results

## Latent Space Representation

The first step of our analysis involves visualizing the **latent representations** generated by the encoder function. In this plot, each point represents a company, projected into a 2D latent space, with points labeled according to their bankruptcy status. This visualization provides insight into the initial separability of the two classes within the latent space.

![Latent_reps](plots/Latent_reps.png)

## Role of Neural ODE in Separating Clusters

To improve class separation, I use a Neural ODE that transforms the latent points toward distinct **attractor states**:
- **Attractor `[1, 0]`**: Represents non-bankrupt companies.
- **Attractor `[0, 1]`**: Represents bankrupt companies.

The animation below illustrates how the Neural ODE guides each point in the latent space, evolving it toward its respective attractor point. This dynamic adjustment helps separate the two classes more effectively by the end of the transformation.

![Seperation](plots/scatter_animation.gif)

## Vector Field Induced by Neural ODE

Finally, I present the **vector field** induced by the Neural ODE over the 2D latent space. The vector field visualization captures the directional flow generated by the Neural ODE, with clear trajectories guiding points toward the two attractor states:
- **[1, 0]** for non-bankrupt
- **[0, 1]** for bankrupt

This vector field provides a powerful visualization of how the Neural ODE’s dynamics steer latent representations based on class, effectively illustrating the boundary between the two states.

![VectorField](plots/Vector_Field.png)

# Strategy Analysis to Avoid Bankruptcy

One of the core objectives of this framework is to analyze whether certain financial strategies could help at-risk companies avoid bankruptcy. By adjusting specific financial metrics, I can assess if and how these changes might influence the model’s bankruptcy predictions.

### Case Study: Testing a Strategy for an At-Risk Company

To illustrate, I took a company from the test dataset that had been labeled as bankrupt and tested a hypothetical strategy to improve its financial stability. I adjusted three key financial ratios by a factor of 0.3:
- **Operating Expense Rate**
- **Interest Expense Ratio**
- **Operating Profit Growth Rate**

Following these adjustments, the model's prediction shifted, indicating that the company would no longer be predicted to go bankrupt under the hypothetical changes.

![Strategies](plots/comparison_animation_with_distinct_heatmaps.gif)

> **Disclaimer**: This example is purely illustrative and does not serve as financial advice. I do not imply that this or any other strategy is feasible or effective. The framework provides insights for analysis rather than professional financial recommendations.

This application highlights the model's potential as a decision-support tool, allowing users to explore hypothetical strategies and their impact on bankruptcy risk. While the model does not predict real-world outcomes, it offers a data-driven approach to examining how changes in financial metrics could influence business stability.


