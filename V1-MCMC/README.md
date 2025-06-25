# Market Prediction Model

A sophisticated statistical model for predicting S&P 500 market movements using adaptive thresholds, market regime detection, and Markov Chain Monte Carlo (MCMC) with replica exchange.

## Overview

This model analyzes historical S&P 500 data to predict the next day's market movement classified into one of five states:
- `big_up`: Large positive move
- `small_up`: Small positive move
- `flat`: Minimal change
- `small_down`: Small negative move
- `big_down`: Large negative move

The prediction process dynamically adjusts to current market conditions through adaptive thresholds that vary based on volatility regimes. The system's architecture combines statistical techniques with financial domain knowledge for robust predictions.

## Key Features

- **Market Regime Detection**: Identifies current market conditions (crisis, high volatility, normal, low volatility)
- **Adaptive Thresholds**: Dynamically adjusts state classification thresholds based on volatility
- **Replica Exchange MCMC**: Implements advanced sampling techniques to avoid local optima
- **Bayesian Framework**: Provides probabilistic predictions with confidence metrics

## Algorithm Components

### Market Regime Detection

The function `detect_market_regime()` analyzes return data to identify the current market environment based on:
- Rolling volatility 
- Volatility-of-volatility
- Skewness and kurtosis
- Trend indicators
- Mean reversion signals

This contextual information helps adjust the model's sensitivity to different market conditions.

### Adaptive State Classification

The model uses dynamic thresholds through several interconnected components:

1. `calculate_regime_thresholds()`: Fits skewed t-distributions to different market regimes
2. `calculate_adaptive_thresholds()`: Creates asymmetric thresholds that adapt to current conditions
3. `apply_balanced_classification()`: Applies these thresholds to classify historical returns

### MCMC Prediction Framework

The prediction engine uses replica exchange MCMC to sample from the posterior distribution:

1. `fit_state_models()`: Creates statistical models of each market state
2. `log_likelihood()`: Evaluates the probability of transitions between states
3. `proposal()`: Generates candidate states using transition probabilities
4. `replica_exchange_mcmc()`: Implements the core sampling algorithm with temperature swaps
5. `get_prediction_confidence()`: Converts MCMC samples into probabilistic predictions

## Function Workflow

1. **Data Processing**:
   - `analyze_market_data()`: Entry point that fetches and processes S&P 500 data

2. **Market Context Analysis**:
   - `detect_market_regime()`: Identifies the current market regime
   - `calculate_regime_thresholds()`: Estimates appropriate thresholds for each regime

3. **State Classification**:
   - `calculate_adaptive_thresholds()`: Computes dynamic thresholds based on volatility
   - `apply_balanced_classification()`: Applies thresholds to create state labels

4. **Model Construction**:
   - `fit_state_models()`: Builds statistical models for each market state
   - `calculate_transition_probs()`: Estimates transition probabilities between states

5. **MCMC Sampling**:
   - `replica_exchange_mcmc()`: Core sampling algorithm with parallel chains
   - `log_likelihood()`: Evaluates candidate states
   - `proposal()`: Generates new candidate states

6. **Prediction Generation**:
   - `get_prediction_confidence()`: Converts samples to predictions with confidence
   - `calculate_state_metrics()`: Evaluates historical accuracy metrics

## Mathematical Foundation

The model combines several statistical approaches:
- Skewed t-distribution fitting for fat-tailed returns
- Bayesian inference for parameter estimation
- Metropolis-Hastings algorithm for sampling
- Replica exchange (parallel tempering) for better exploration

## Diagnostics

The model provides extensive diagnostics to evaluate prediction quality:

- Gelman-Rubin statistic (R-hat) for convergence monitoring
- Temperature swap acceptance rates for replica exchange efficiency
- Effective sample size estimation
- Prediction certainty metrics

## Dependencies

- numpy
- pandas
- scipy
- yfinance
- time
- datetime

## Disclaimer

This is a statistical model and should not be used as financial advice. Market conditions can change rapidly and unpredictably.
