# Market Direction Predictor

A sophisticated statistical model for predicting S&P 500 weekly market direction using Hidden Markov Models, discriminative spectral analysis, and Bayesian uncertainty quantification.

## Overview

This model analyzes historical S&P 500 data to predict the next week's market direction classified as either upward or downward movement. The prediction process incorporates market regime detection through Hidden Markov Models and employs discriminative spectral analysis with Fisher's Linear Discriminant Analysis to extract predictive features. The system's architecture combines advanced machine learning techniques with financial domain knowledge for robust probabilistic predictions.

## Key Features

- **Hidden Markov Model Regime Detection**: Identifies three market regimes (low volatility, normal, high volatility)
- **Discriminative Spectral Analysis**: Uses Fisher's LDA for optimal feature transformation
- **Bayesian Model Averaging**: Combines predictions across different market regimes
- **Uncertainty Quantification**: Provides both epistemic and aleatoric uncertainty estimates
- **Confidence Calibration**: Ensures prediction confidence scores are well-calibrated

## Algorithm Components

### Market Regime Detection

The `HiddenMarkovModel` class implements a three-state Hidden Markov Model to identify market volatility regimes based on:
- Weekly returns
- Rolling volatility measures  
- Momentum indicators
- Multivariate Gaussian emission probabilities

This regime information conditions the prediction models to adapt their behavior to current market conditions.

### Feature Engineering Pipeline

The model constructs a comprehensive feature set through several interconnected components:

1. `calculate_features()`: Generates 40+ technical and statistical features across multiple categories
2. `select_features()`: Applies multi-criteria feature selection using correlation, mutual information, and statistical significance
3. Feature categories include volatility metrics, momentum indicators, mean reversion signals, volume analysis, and regime indicators

### Discriminative Spectral Prediction Framework

The prediction engine uses discriminative spectral analysis for optimal class separation:

1. `DiscriminativeSpectralPredictor`: Main prediction class implementing Bayesian inference
2. `compute_discriminative_spectral_decomposition()`: Applies Fisher's LDA for feature transformation
3. `fit_regime_model()`: Creates Bayesian linear models for each market regime
4. `predict_proba()`: Generates probabilistic predictions with uncertainty quantification

## Function Workflow

1. **Data Processing**:
   - `prepare_data()`: Fetches S&P 500 data and converts to weekly frequency
   - `calculate_features()`: Generates comprehensive feature set

2. **Regime Analysis**:
   - `HiddenMarkovModel.fit()`: Estimates HMM parameters using Expectation-Maximization
   - `forward_backward()`: Computes regime probabilities for each time period

3. **Feature Processing**:
   - `select_features()`: Applies multi-criteria feature selection
   - `compute_discriminative_spectral_decomposition()`: Transforms features using Fisher's LDA

4. **Model Construction**:
   - `estimate_empirical_bayes_hyperparams()`: Estimates prior parameters from data
   - `fit_regime_model()`: Builds Bayesian models for each regime

5. **Prediction Generation**:
   - `predict_proba()`: Combines regime-specific predictions using Bayesian model averaging
   - `fit_confidence_calibrator()`: Calibrates confidence scores using isotonic regression

6. **Backtesting Framework**:
   - `run_backtest()`: Implements walk-forward validation with configurable retraining
   - `test_strategies()`: Evaluates multiple trading strategies with different risk profiles

## Mathematical Foundation

The model combines several advanced statistical approaches:
- Hidden Markov Models with multivariate Gaussian emissions
- Fisher's Linear Discriminant Analysis for discriminative feature extraction
- Bayesian linear regression with empirical Bayes hyperparameter estimation
- Isotonic regression for probability calibration
- Model averaging for uncertainty quantification

## Diagnostics

The model provides extensive diagnostics to evaluate prediction quality:

- Directional accuracy across different market regimes
- AUC-ROC and Brier scores for probabilistic prediction assessment
- Expected Calibration Error (ECE) for confidence reliability
- Uncertainty analysis comparing model confidence with actual accuracy
- Comprehensive backtesting with multiple performance metrics

## Strategy Implementation

The backtesting framework implements several trading strategies:

- **Simple Directional**: Basic probability threshold approach
- **High Confidence**: Trades only high-confidence predictions
- **Regime-Aware Bayesian**: Adapts decision thresholds by market regime
- **Multi-Factor Bayesian**: Combines confidence, uncertainty, and regime information

## Dependencies

- numpy
- pandas
- scipy
- sklearn
- yfinance
- matplotlib
- warnings
- datetime

## Disclaimers

This is a statistical model for research purposes and should not be used as financial advice. Market conditions can change rapidly and unpredictably. Past performance does not guarantee future results.

This README was generated by AI as I do not praticularly enjoy writing this sort of documentation
