# Bayesian Logit-Normal Inference

## Overview
This project implements Bayesian inference for a logit-normal regression model, estimating the extent to which justice needs are met based on survey data. The model assumes that the logit transformation of the response variable follows a normal distribution with a linear predictor structure.

## Model Specification
We assume the following transformation and likelihood:

$z_i = \log\left(\frac{y_i}{1-y_i}\right) \sim \mathcal{N}(X_i\beta,\,\sigma^2)$

This results in the likelihood function:

$L(\beta, \sigma^2 \mid z) \propto (\sigma^2)^{-n/2} \exp\left\{-\frac{1}{2\sigma^2} (z - X\beta)^\top (z - X\beta) \right\}$

Using Jeffrey's prior:

$\pi(\beta,\sigma^2) \propto \frac{1}{\sigma^{p+2}}$

We derive the conditional posterior distributions:

- $\beta \mid \sigma^2, z \sim \mathcal{N}\left(\hat{\beta}, \sigma^2 (X^\top X)^{-1}\right)$, where $\hat{\beta} = (X^\top X)^{-1}X^\top z$.
- $\sigma^2 \mid \beta, z \sim \text{Inverse-Gamma}\left(\frac{n}{2},\,\frac{\text{SSE}}{2}\right)$, where $\text{SSE} = (z - X\beta)^\top (z - X\beta)$

## Data Processing
The dataset is preprocessed to:
- Encode categorical variables (e.g., employment and country) using one-hot encoding.
- Remove missing or invalid values for gender, education, age, urban status, and ethnic groups.
- Compute a justice gap score and apply logit transformation.
- Aggregate data at the regional level for modeling.

## Implementation
The Bayesian inference process follows these steps:
1. Split the dataset into training and testing sets.
2. Define priors for $\beta$ and $\sigma^2$.
3. Compute posterior parameters based on observed data.
4. Draw posterior samples using the inverse gamma and multivariate normal distributions.
5. Generate posterior predictive samples and estimate justice scores.
6. Evaluate model performance using:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R-squared $R^2$
   - Posterior Predictive p-value

## Results & Visualization
- Kernel density plots of predicted vs. actual justice scores.
- Scatter plots of predictions against actual values.
- Performance metrics to assess model accuracy.

## Dependencies
- `pandas`, `numpy`, `seaborn`, `matplotlib`
- `scipy.stats` for Bayesian inference
- `sklearn` for preprocessing and model evaluation

## Future Work
- Investigate different priors for $\beta$ and $\sigma^2$.
- Incorporate hierarchical modeling for regional effects.
- Extend the model to handle missing data robustly.
- Explore feature scaling techniques to improve stability.

## Author
Developed by Isabella Coddington and Alex Koutromanos.

For any questions or contributions, please reach out!

emails: imc42@georgetown.edu, aak183@georgetown.edu

