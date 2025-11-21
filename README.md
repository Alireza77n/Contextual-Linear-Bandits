# Contextual Linear Bandits for Ad Recommendation

## Project overview

This project studies **contextual linear bandits** in the setting of an online
advertising system. At each time step a user (or website) arrives with some
features, and the algorithm must choose which ad to show. After the ad is
shown, we observe whether the user clicked or not and use this feedback to
improve future decisions. The objective is to **maximize cumulative profit**
(clicks or revenue) over time.

The work compares two approaches:

- A **contextual bandit based on online logistic regression**
- **Thompson Sampling** with an approximate Bayesian posterior (Laplace
  approximation)

All experiments are run on **synthetic data** designed to mimic an ad-serving
environment.

---

## Problem description

- Each arriving user is described by a **feature vector** (context), for
  example binary indicators of interests or attributes.
- Each ad is associated with an unknown **parameter vector** that tells us how
  attractive that ad is for different kinds of users.
- The probability of a click is modeled as a simple function of the context
  and the ad parameters.
- In the simulation, both contexts and ad parameters are generated randomly in
  a controlled way so we know the “true” data-generating process and can
  measure how well each algorithm learns it.

The bandit algorithm must balance:

- **Exploration** – trying ads we are uncertain about  
- **Exploitation** – showing ads that currently look most profitable

---

## Method 1: Online Logistic Regression Bandit

This method uses **logistic regression** to model the click probability of
each ad given the user’s features.

Main ideas:

- Maintain a parameter vector for the ad.
- When a user arrives, predict the click probability for each ad using
  logistic regression.
- Choose the ad with the highest predicted click probability (or profit).
- After observing click / no-click, update the parameters with an online
  gradient step.

This gives a **frequentist baseline**: it is simple, fast, and updates the
model continuously as new data arrive.

---

## Method 2: Thompson Sampling with Laplace Approximation

This method is a **Bayesian contextual bandit**:

- We maintain a **posterior distribution** over the ad parameters instead of a
  single point estimate.
- The posterior is approximated by a multivariate normal distribution using a
  **Laplace approximation** (centered at the maximum a posteriori estimate).
- At each time step:
  1. Sample a parameter vector from the posterior.
  2. For each ad, compute the click probability using this sample.
  3. Select the ad with the highest sampled expected reward.
  4. Update the posterior approximation using the new observation.

Thompson Sampling naturally handles the exploration–exploitation trade-off
because randomness comes from sampling the parameters.

---

## Experiments

The simulation studies the effect of:

- **Batch size** – how often parameters/posterior are updated  
  - Smaller batches give more frequent updates.  
  - Larger batches give more stable but slower updates.
- **Feature dimension** – the number of features in the context vector  
  - Higher dimension can represent users more richly but can also lead to
    overfitting or slower learning.

Key observations:

- For the online logistic regression bandit, **smaller batch sizes** tend to
  improve performance because the model adapts more quickly.
- For Thompson Sampling, a **larger batch size** can produce a more stable
  posterior and better long-run profit.
- Increasing the feature dimension helps up to a point, but very high
  dimensionality does not always give better results; there is a trade-off
  between expressiveness and robustness.

---

## Summary

This project provides a controlled environment to compare:

- A simple **online logistic regression** bandit
- A **Bayesian Thompson Sampling** bandit with approximate posterior updates

It shows how design choices such as batch size and feature dimension influence
learning speed, exploration, and cumulative profit in a contextual bandit
ad-recommendation setting.
