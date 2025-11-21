# Contextual Linear Bandits in Recommender Systems

## Project overview

This project studies **contextual linear bandits** for optimizing ad display in a recommender–system setting.  
The goal is to decide, at each time step, **which advertisement (arm)** to show to a user (context) in order to **maximize cumulative clicks / profit** over time. :contentReference[oaicite:0]{index=0}  

Two main approaches are implemented and compared on simulated data:

- **Online logistic regression** as a contextual bandit baseline  
- **Thompson Sampling with Laplace approximation** as a Bayesian, probabilistic bandit algorithm  

The project focuses on how these methods update their beliefs over ad parameters and how design choices (batch size, context dimension) affect profit and learning dynamics.

---

## Problem setting: contextual linear bandits

We consider a standard **contextual linear bandit** model tailored to ads:

- At each time step *t*, a user (or website) arrives with a **context vector**  
  \[
  z_t \in \mathbb{R}^d
  \]
- Each ad (arm) is associated with an unknown **parameter vector**  
  \[
  \theta_z \in \mathbb{R}^d
  \]
- The expected reward (click / no-click) when showing an ad in context \(z_t\) is a linear function:
  \[
  R_t = z_t^\top \theta_z + \varepsilon_t
  \]
  where \(\varepsilon_t\) is Gaussian noise. :contentReference[oaicite:1]{index=1}

In the simulation:

- Each coordinate of the context vector \(z\) is sampled from a **Binomial(1, p)** distribution.  
- The ad parameters \(\theta_z\) are drawn from a **multivariate normal** distribution  
  \[
  \theta_z \sim \mathcal{N}(\mu, I_d)
  \]
- The task is to **learn the distribution of \(\theta_z\)** and choose ads that maximize total reward (profit).

---

## Algorithm 1: Online Logistic Regression

Online logistic regression is used as a **contextual bandit baseline**. The click-through rate (CTR) is modeled as:

\[
p(y_t = 1 \mid z_t, \theta) = \sigma(z_t^\top \theta)
\]

where \(\sigma(\cdot)\) is the logistic (sigmoid) function and \(y_t \in \{0,1\}\) indicates click / no click. :contentReference[oaicite:2]{index=2}  

### Bayesian formulation

- **Prior**: multivariate Gaussian prior over parameters  
  \[
  \theta \sim \mathcal{N}(\mu_0, \Sigma_0)
  \]
- **Likelihood**: Bernoulli with logistic link  
  \[
  p(y_t \mid z_t, \theta) = \text{Bernoulli}(\sigma(z_t^\top \theta))
  \]

The log-posterior combines the log-likelihood and the Gaussian prior, and the **MAP estimate** \(\theta_{\text{MAP}}\) is found by maximizing this log-posterior.

### Online updates

In the online setting, parameters are updated incrementally using **stochastic gradient descent (SGD)**:

\[
\theta_{t+1} = \theta_t - \eta\,(\sigma(z_t^\top \theta_t) - y_t)\,z_t
\]

with learning rate \(\eta\). This allows the model to continuously adapt as new click data arrive.

### Laplace approximation (for uncertainty)

For approximate Bayesian treatment, a **Laplace approximation** is used:

1. Find the MAP estimate \(\theta_{\text{MAP}}\).
2. Compute the **Hessian** of the negative log-posterior at \(\theta_{\text{MAP}}\).
3. Approximate the posterior as a Gaussian:
   \[
   p(\theta \mid \text{data}) \approx \mathcal{N}(\theta_{\text{MAP}}, \Sigma),\quad
   \Sigma = H^{-1}
   \]

This gives a local Gaussian approximation around the MAP solution, which can be used to quantify uncertainty. :contentReference[oaicite:3]{index=3}  

---

## Algorithm 2: Thompson Sampling with Laplace approximation

The second algorithm is **Thompson Sampling (TS)** adapted to logistic/contextual bandits using the Laplace approximation.

### Core idea

At each time step:

1. **Sample** a parameter vector \(\tilde{\theta}\) from the approximate posterior:
   \[
   \tilde{\theta} \sim \mathcal{N}(\theta_{\text{MAP}}, \Sigma)
   \]
2. For each available ad (arm), compute the predicted CTR using \(\tilde{\theta}\).
3. **Select the ad** that maximizes the sampled expected reward.
4. Observe the click outcome and **update** the posterior (via MAP + Hessian update).

This method naturally handles the **exploration–exploitation trade-off**:  
sampling from the posterior occasionally selects arms with more uncertainty, allowing exploration, while still favoring arms with higher expected reward. :contentReference[oaicite:4]{index=4}  

---

## Experimental setup

The experiments are conducted on **synthetic data** designed to mimic an ad-serving environment:

- Number of ads (arms): **200**  
- Context vectors:
  - \(z \in \mathbb{R}^d\), each component \(z_i \sim \text{Binomial}(1, p)\)
  - Different **context dimensions d** are tested to study the effect of feature dimensionality.
- Parameters:
  - \(\theta_z \sim \mathcal{N}(\mu, I_d)\)
  - Prior over \(\theta\) is Gaussian with mean \(\mu\) and identity covariance.
- Two **batch sizes** are compared when updating parameters:
  - Batch size **100**
  - Batch size **1000**
- Learning rate for updating the mean \(\mu\): **α = 0.01**. :contentReference[oaicite:5]{index=5}  

Figure 1 in the report (page 11) shows cumulative profit curves for different combinations of:

- Number of websites (time horizon)
- Context dimensions
- Algorithms (logistic regression vs Thompson Sampling)
- Batch sizes 100 vs 1000

---

## Results summary

Key empirical findings:

- For **online logistic regression**, a **batch size of 100** performs better than 1000:
  - More frequent updates make the model more responsive to new data.
  - This leads to faster adaptation and higher accumulated profit in the simulation.
- For **Thompson Sampling with Laplace approximation**, a **batch size of 1000** performs better:
  - The probabilistic nature of TS and the richer batch updates give more stable posterior estimates.
  - This improves profit maximization compared to small-batch TS in this setting. :contentReference[oaicite:6]{index=6}  

Effect of latent feature dimension:

- Increasing the **dimension d** of the latent features generally **improves maximum profit** and helps the model learn a richer representation.
- However, this improvement **does not grow indefinitely**:
  - Beyond some dimension, performance can plateau or even degrade due to overfitting and increased complexity.
- There is a trade-off between **expressiveness** and **robustness**, so the latent dimension should be chosen carefully and validated empirically. :contentReference[oaicite:7]{index=7}  

---

## What this project demonstrates

Conceptually, the project showcases:

- How **contextual linear bandits** can be used for ad recommendation.
- The difference between:
  - A **frequentist online logistic regression** baseline, and  
  - A **Bayesian Thompson Sampling** method with approximate posterior sampling.
- Practical issues such as:
  - Batch size choice
  - Posterior approximation with Laplace
  - Sensitivity to feature dimensionality
  - Exploration–exploitation trade-offs in recommender systems

---

## Possible repository structure (suggested)

If you organize the accompanying code for this project, a typical layout might look like:

- `env/` — simulation environment for contexts, ads, and rewards  
- `algorithms/`  
  - `online_logistic.py` — online logistic regression implementation  
  - `thompson_laplace.py` — Thompson Sampling with Laplace approximation  
- `experiments/`  
  - scripts/notebooks to reproduce the plots and profit curves  
- `plots/` — generated figures (including cumulative reward plots)  
- `report/` — this PDF report

You can adapt the names to match your actual files and keep this README as the conceptual description of the project.
