import scipy
import yfinance as yf
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import BDay
import scipy.stats as stats
from scipy.special import gamma
from scipy import integrate, optimize, linalg

# Market Regime and Thresholds

def detect_market_regime(returns, window=7, lookback_period=252):
    # Calculate key metrics over the window
    rolling_vol = returns.rolling(window, min_periods=1, center=False).std() * np.sqrt(252)  # Annualized
    vol_of_vol = rolling_vol.rolling(window, min_periods=1, center=False).std()
    skewness = returns.rolling(window, min_periods=1, center=False).skew()
    kurtosis = returns.rolling(window, min_periods=1, center=False).kurt()

    # market trend indicators to improve context
    short_trend = returns.rolling(10, min_periods=5).mean()
    medium_trend = returns.rolling(30, min_periods=15).mean()
    trend_strength = (short_trend / rolling_vol).abs()  # Trend relative to volatility

    # mean reversion indicators
    cum_returns_20d = returns.rolling(20, min_periods=10).sum()
    mean_reversion_signal = -np.sign(cum_returns_20d) * (cum_returns_20d.abs() / rolling_vol)

    # Calculate rolling metrics for dynamic thresholds
    vol_of_vol_mean = vol_of_vol.rolling(63, min_periods=1, center=False).mean()

    # Create regime indicators
    regimes = pd.Series(index=returns.index, data='normal')

    # For each date, calculate dynamic thresholds based on historical data
    for i in range(lookback_period, len(returns)):
        # Get historical data for percentile calculation
        hist_window = slice(max(0, i - lookback_period), i)
        historical_vol = rolling_vol.iloc[hist_window]

        # Dynamic thresholds based on historical percentiles
        crisis_vol_threshold = historical_vol.quantile(0.95)  # 95th percentile for crisis
        high_vol_threshold = historical_vol.quantile(0.75)  # 75th percentile for high vol
        low_vol_threshold = historical_vol.quantile(0.25)  # 25th percentile for low vol

        # Current values
        current_vol = rolling_vol.iloc[i]
        current_vov = vol_of_vol.iloc[i]
        current_vov_mean = vol_of_vol_mean.iloc[i]
        current_skew = skewness.iloc[i]
        current_kurt = kurtosis.iloc[i]

        # NEW: Get trend and mean reversion values
        current_trend_str = trend_strength.iloc[i] if i < len(trend_strength) else 0
        current_mean_rev = mean_reversion_signal.iloc[i] if i < len(mean_reversion_signal) else 0

        # Dynamic skewness threshold based on tail heaviness
        # More negative skew is expected in higher volatility regimes
        skew_threshold = -0.5 - (0.5 * (current_vol / crisis_vol_threshold))
        # Dynamic kurtosis threshold based on tail heaviness
        # Higher kurtosis threshold for higher volatility
        kurt_threshold = 3.0 + (3.0 * (current_vol / high_vol_threshold))

        # Crisis detection
        crisis_mask = (
                (current_vol > crisis_vol_threshold) &
                (current_vov > current_vov_mean * 1.8) &
                (
                        (current_skew < skew_threshold) |
                        (current_kurt > kurt_threshold) |
                        (current_trend_str > 1.5)
                )
        )

        # High volatility detection
        high_vol_mask = (
                (current_vol > high_vol_threshold) &
                (~crisis_mask) &
                (
                        (current_vov > current_vov_mean * 0.8) |
                        (current_trend_str > 1.0)
                )
        )

        # Low volatility detection
        low_vol_mask = (
                (current_vol < low_vol_threshold) &
                (current_vov < current_vov_mean * 0.6) &
                (abs(current_mean_rev) < 0.5)
        )

        # Apply regime classifications
        if crisis_mask:
            regimes.iloc[i] = 'crisis'
        elif high_vol_mask:
            regimes.iloc[i] = 'high_vol'
        elif low_vol_mask:
            regimes.iloc[i] = 'low_vol'
        else:
            regimes.iloc[i] = 'normal'

    # For initial period where we don't have enough history, use simpler method
    if lookback_period > 0:
        initial_vol_75p = rolling_vol.iloc[:lookback_period].quantile(0.75)
        initial_vol_25p = rolling_vol.iloc[:lookback_period].quantile(0.25)
        initial_vol_95p = rolling_vol.iloc[:lookback_period].quantile(0.95)

        for i in range(min(lookback_period, len(returns))):
            if rolling_vol.iloc[i] > initial_vol_95p:
                regimes.iloc[i] = 'crisis'
            elif rolling_vol.iloc[i] > initial_vol_75p:
                regimes.iloc[i] = 'high_vol'
            elif rolling_vol.iloc[i] < initial_vol_25p:
                regimes.iloc[i] = 'low_vol'
            else:
                regimes.iloc[i] = 'normal'

    return regimes

def calculate_regime_thresholds(returns, regimes):
    def skewed_t_pdf(x, mu, sigma, nu, lambda_):
        if sigma <= 0 or nu <= 2 or abs(lambda_) >= 1:
            return np.zeros_like(x)

        a = 4 * lambda_ * (nu - 2) / (nu - 1)
        b_squared = 1 + 3 * lambda_ ** 2 - a ** 2
        if b_squared <= 0:
            return np.zeros_like(x)

        b = np.sqrt(b_squared)
        z = (x - mu) / max(sigma, 1e-10)

        pdf = np.zeros_like(x, dtype=float)
        z_mask = z < -a / b

        # Left tail
        left_const = b * gamma((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
        pdf[z_mask] = left_const * (1 + 1 / (nu - 2) * ((b * z[z_mask] + a) / (1 - lambda_)) ** 2) ** (-((nu + 1) / 2))

        # Right tail
        right_const = b * gamma((nu + 1) / 2) / (np.sqrt(np.pi * (nu - 2)) * gamma(nu / 2))
        pdf[~z_mask] = right_const * (1 + 1 / (nu - 2) * ((b * z[~z_mask] + a) / (1 + lambda_)) ** 2) ** (
            -((nu + 1) / 2))

        return pdf / sigma

    def skewed_t_loglike(params, data):
        mu, sigma, nu, lambda_ = params

        if (sigma <= 0 or
                nu <= 2.1 or
                abs(lambda_) >= 0.99 or
                not np.isfinite(mu)):
            return np.inf

        a = 4 * lambda_ * (nu - 2) / (nu - 1)
        b_squared = 1 + 3 * lambda_ ** 2 - a ** 2
        if b_squared <= 0:
            return np.inf

        pdf_values = skewed_t_pdf(data, mu, sigma, nu, lambda_)
        valid_pdf = pdf_values > 0
        if not np.any(valid_pdf):
            return np.inf

        return -np.sum(np.log(np.maximum(pdf_values, 1e-300)))

    def get_initial_params(data):
        data = np.array(data)
        valid_data = data[np.isfinite(data)]

        if len(valid_data) < 10:
            return None

        mean = np.mean(valid_data)
        std = np.std(valid_data)
        skewness = stats.skew(valid_data)
        kurtosis = stats.kurtosis(valid_data, fisher=True)  # Fisher's definition (excess kurtosis)

        # Calculate quantiles for robust estimation
        q25 = np.percentile(valid_data, 25)
        q75 = np.percentile(valid_data, 75)
        iqr = q75 - q25

        # Initial guess for mu
        mu_init = np.median(valid_data)

        # Initial guess for sigma
        sigma_init = iqr / 1.35

        # Initial guess for nu
        if kurtosis > 0:
            nu_init = min(max(4.1, 6 / kurtosis + 4), 30)
        else:
            nu_init = 10.0  # Default value for normal-like tails

        # Initial guess for lambda
        lambda_init = np.clip(skewness / 2, -0.95, 0.95)

        return [mu_init, sigma_init, nu_init, lambda_init]

    def fit_skewed_t(data, n_restarts=15):
        # Import warnings to handle them properly
        import warnings
        from scipy.optimize import minimize

        # Temporarily suppress the specific warning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    message='invalid value encountered in subtract')

            data = np.array(data)
            valid_data = data[np.isfinite(data)]
            if len(valid_data) < 10:
                print("Warning: Insufficient valid data points, using fallback estimation")
                return fallback_estimate(valid_data)

            # Get initial parameters using robust method
            init_params = get_initial_params(valid_data)
            if init_params is None:
                print("Warning: Failed to estimate initial parameters, using fallback")
                return fallback_estimate(valid_data)

            data_median = np.median(valid_data)
            q25, q75 = np.percentile(valid_data, [25, 75])
            data_scale = max((q75 - q25) / 1.35, 1e-4)  # Ensure non-zero scale

            # Ensure minimum scale to prevent division by very small numbers
            data_scale = max(data_scale, 1e-3 * np.abs(data_median) + 1e-6)

            scaled_data = (valid_data - data_median) / data_scale

            # Scale initial parameters
            scaled_init_params = init_params.copy()
            scaled_init_params[0] = (init_params[0] - data_median) / data_scale
            scaled_init_params[1] = init_params[1] / data_scale

            # Make bounds more conservative
            bounds = [
                (-3, 3),  # mu (more restricted range)
                (0.1, 5),  # sigma (increased minimum)
                (3.0, 30),  # nu (more restricted range)
                (-0.85, 0.85)  # lambda (more restricted range)
            ]

            # More robust regularization
            def robust_objective(params, data):
                mu, sigma, nu, lambda_ = params

                # Stricter parameter validation
                if (sigma <= 0.1 or  # Higher minimum sigma
                        nu <= 3.0 or nu > 30 or  # Tighter nu bounds
                        abs(lambda_) >= 0.85 or  # Tighter lambda bounds
                        not np.isfinite(mu)):
                    return 1e6  # High but finite value

                # Wrap calculation in try/except
                try:
                    # skewed t log-likelihood with handling for invalid values
                    pdf_values = skewed_t_pdf(data, mu, sigma, nu, lambda_)

                    # Check for valid PDF values
                    if np.any(~np.isfinite(pdf_values)) or np.all(pdf_values <= 0):
                        return 1e6

                    log_pdf = np.log(np.maximum(pdf_values, 1e-300))
                    if np.any(~np.isfinite(log_pdf)):
                        return 1e6

                    nll = -np.sum(log_pdf)

                    # Add strong regularization
                    nu_penalty = 0.05 * ((nu - 8) / 5) ** 2  # Center around 8 degrees of freedom
                    lambda_penalty = 0.2 * (abs(lambda_) / 0.7) ** 4  # Strongly discourage extreme skewness
                    sigma_penalty = 0.8 * (0.2 / sigma) ** 2 if sigma < 0.2 else 0  # Avoid small sigma

                    # Handle potential invalid values
                    if not np.isfinite(nll) or not np.isfinite(nu_penalty) or not np.isfinite(
                            lambda_penalty) or not np.isfinite(sigma_penalty):
                        return 1e6

                    return nll + nu_penalty + lambda_penalty + sigma_penalty

                except Exception as e:
                    # Catch any exceptions in the calculation
                    # print(f"Exception in objective function: {e}")
                    return 1e6

            # Try different optimization methods in sequence
            best_result = None
            best_nll = np.inf

            # List of optimization methods to try in order of preference
            methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'SLSQP']

            # Try each method with proper options
            for method in methods:
                try:
                    if method == 'Nelder-Mead':
                        options = {'maxiter': 3000, 'disp': False, 'adaptive': True}
                    elif method == 'Powell':
                        options = {'maxiter': 3000, 'disp': False}
                    elif method == 'L-BFGS-B':
                        options = {'maxfun': 3000, 'disp': False}
                    else:
                        options = {'maxiter': 3000, 'disp': False}

                    # Try optimization with current method
                    result = minimize(
                        lambda p: robust_objective(p, scaled_data),
                        scaled_init_params,
                        method=method,
                        bounds=bounds if method in ['L-BFGS-B', 'SLSQP'] else None,
                        options=options
                    )

                    # Check if this result is better
                    if (result.success or result.fun < 1e6) and result.fun < best_nll:
                        unscaled_params = result.x.copy()
                        unscaled_params[0] = result.x[0] * data_scale + data_median
                        unscaled_params[1] = result.x[1] * data_scale

                        best_nll = result.fun
                        best_result = type('OptResult', (), {
                            'x': unscaled_params,
                            'fun': result.fun,
                            'success': result.success
                        })

                        # If we got a good result, don't try additional methods
                        if result.success and result.fun < 1e4:
                            break

                except Exception as e:
                    print(f"Optimization with {method} failed: {e}")
                    continue

            # If all methods failed, use fallback
            if best_result is None:
                print("All optimization methods failed, using fallback estimation")
                return fallback_estimate(valid_data)

            # Check if best result is reasonable
            final_params = best_result.x

            # Validate parameters
            if (final_params[1] <= 0 or final_params[2] <= 2.1 or abs(final_params[3]) >= 0.99):
                print("Optimized parameters invalid, using fallback estimation")
                return fallback_estimate(valid_data)

            # Make sure parameters are within reasonable bounds
            final_params[1] = max(final_params[1], 1e-5)  # sigma > 0
            final_params[2] = min(max(final_params[2], 3.0), 30.0)  # 3 <= nu <= 30
            final_params[3] = np.clip(final_params[3], -0.85, 0.85)  # -0.85 <= lambda <= 0.85

            return final_params

    def fallback_estimate(data):
        data = np.array(data)
        valid_data = data[np.isfinite(data)]

        # Basic statistics
        mean = np.mean(valid_data)
        std = max(np.std(valid_data), 1e-5)  # Ensure non-zero std

        # Try to calculate skewness and kurtosis
        try:
            skewness = stats.skew(valid_data)
            kurtosis = stats.kurtosis(valid_data, fisher=True)
        except:
            skewness = 0
            kurtosis = 0

        # Estimate degrees of freedom from kurtosis
        if kurtosis > 0:
            est_df = 6 / max(0.1, kurtosis) + 4
            est_df = min(max(3.0, est_df), 20.0)
        else:
            est_df = 8.0  # Default value

        # Estimate skewness parameter
        est_lambda = np.clip(skewness / 3, -0.7, 0.7)

        # Return conservative estimates
        return np.array([mean, std, est_df, est_lambda])

    # Calculate regime-specific distributions and percentiles
    regime_params = {}
    regime_thresholds = {}

    for regime in ['crisis', 'high_vol', 'normal', 'low_vol']:
        regime_returns = returns[regimes == regime]
        if len(regime_returns) >= 50:  # Minimum sample size for fitting
            try:
                params = fit_skewed_t(regime_returns.values)
                regime_params[regime] = params

                mu, sigma, nu, lambda_ = params

                if regime == 'crisis':
                    big_percentile = 0.87
                    small_percentile = 0.70
                elif regime == 'high_vol':
                    big_percentile = 0.85
                    small_percentile = 0.65
                elif regime == 'normal':
                    big_percentile = 0.80
                    small_percentile = 0.60
                else:  # low_vol
                    big_percentile = 0.75
                    small_percentile = 0.55

                # Calculate thresholds using numerical optimization
                def find_percentile(target_prob):
                    def quantile_equation(x):
                        result = integrate.quad(
                            lambda t: skewed_t_pdf(t, mu, sigma, nu, lambda_),
                            -np.inf,
                            x,
                            limit=100
                        )[0] - target_prob
                        return result

                    try:
                        return optimize.brentq(
                            quantile_equation,
                            mu - 5 * sigma,
                            mu + 5 * sigma,
                            xtol=1e-4
                        )
                    except:
                        # Fallback approximation if the numeric method fails
                        samples = np.random.standard_t(df=nu, size=100000)
                        if lambda_ != 0:
                            a = 4 * lambda_ * (nu - 2) / (nu - 1)
                            b = np.sqrt(1 + 3 * lambda_ ** 2 - a ** 2)
                            samples = samples * (1 + lambda_ * np.sign(samples))

                        samples = mu + sigma * samples
                        return np.percentile(samples, target_prob * 100)

                # positive thresholds
                big_threshold = abs(find_percentile(big_percentile))
                small_threshold = abs(find_percentile(small_percentile))

                # Set dynamic bounds based on volatility regime
                regime_vol = regime_returns.std() * np.sqrt(252)  # Annualized volatility

                if regime == 'crisis':
                    min_big = regime_vol * 0.13
                    max_big = regime_vol * 0.45

                    min_small = regime_vol * 0.06
                    max_small = regime_vol * 0.25
                elif regime == 'high_vol':
                    min_big = regime_vol * 0.08
                    max_big = regime_vol * 0.35

                    min_small = regime_vol * 0.04
                    max_small = regime_vol * 0.18
                elif regime == 'low_vol':
                    min_big = regime_vol * 0.02
                    max_big = regime_vol * 0.12

                    min_small = regime_vol * 0.01
                    max_small = regime_vol * 0.07
                else:  # normal
                    min_big = regime_vol * 0.06
                    max_big = regime_vol * 0.25

                    min_small = regime_vol * 0.03
                    max_small = regime_vol * 0.15

                # Apply dynamic bounds
                big_threshold = np.clip(big_threshold, min_big, max_big)
                small_threshold = np.clip(small_threshold, min_small, max_small)

                # Create separate thresholds for up and down moves
                if regime in ['crisis', 'high_vol']:
                    # Stronger asymmetry in volatile regimes
                    down_big_threshold = big_threshold * 0.90
                    down_small_threshold = small_threshold * 0.90
                    up_big_threshold = big_threshold * 1.05
                    up_small_threshold = small_threshold * 1.05
                else:
                    # Less asymmetry in calm regimes
                    down_big_threshold = big_threshold * 0.95
                    down_small_threshold = small_threshold * 0.95
                    up_big_threshold = big_threshold * 1.00
                    up_small_threshold = small_threshold * 1.00

                # Store asymmetric thresholds
                regime_thresholds[regime] = {
                    'big_up': up_big_threshold,
                    'small_up': up_small_threshold,
                    'big_down': down_big_threshold,
                    'small_down': down_small_threshold
                }
            except Exception as e:
                print(f"Failed to fit distribution for {regime} regime: {e}")

    # Default thresholds with better scaling based on historical volatility
    historical_vol = returns.std() * np.sqrt(252)  # Annualized historical volatility

    default_thresholds = {
        'crisis': {
            'big_up': historical_vol * 0.26,
            'small_up': historical_vol * 0.16,
            'big_down': historical_vol * 0.23,
            'small_down': historical_vol * 0.14
        },
        'high_vol': {
            'big_up': historical_vol * 0.16,
            'small_up': historical_vol * 0.10,
            'big_down': historical_vol * 0.14,
            'small_down': historical_vol * 0.08
        },
        'normal': {
            'big_up': historical_vol * 0.10,
            'small_up': historical_vol * 0.06,
            'big_down': historical_vol * 0.09,
            'small_down': historical_vol * 0.055
        },
        'low_vol': {
            'big_up': historical_vol * 0.06,
            'small_up': historical_vol * 0.04,
            'big_down': historical_vol * 0.055,
            'small_down': historical_vol * 0.035
        }
    }

    for regime in ['crisis', 'high_vol', 'normal', 'low_vol']:
        if regime not in regime_thresholds:
            print(f"Using default thresholds for regime {regime}")
            regime_thresholds[regime] = default_thresholds[regime]

    return regime_thresholds, regime_params

def calculate_adaptive_thresholds(returns):
    returns = returns.ffill().bfill()

    regimes = detect_market_regime(returns, window=7)

    regime_thresholds, regime_params = calculate_regime_thresholds(returns, regimes)

    base_vol = returns.ewm(span=20, min_periods=5).std()

    # Calculate short and long term volatility components for volatility clustering
    vol_short = returns.ewm(span=10, min_periods=5).std()
    vol_long = returns.ewm(span=63, min_periods=10).std()

    # Volatility ratio to identify volatility clustering
    vol_ratio = (vol_short / vol_long).fillna(1.0).clip(0.5, 2.0)

    # Identify direction of recent returns for asymmetric effects
    # Negative returns typically lead to more volatility (leverage effect)
    sign_effect = returns.rolling(window=10).apply(
        lambda x: np.sum(np.sign(x) * np.abs(x) ** 1.5) / np.sum(np.abs(x) ** 0.5)
    ).fillna(0.0)

    sign_effect = sign_effect / sign_effect.rolling(63).std().fillna(1.0)

    up_trend = returns.rolling(window=10).apply(
        lambda x: np.sum(np.maximum(x, 0)) / (np.sum(np.abs(x)) + 1e-8)
    ).fillna(0.5)

    down_trend = returns.rolling(window=10).apply(
        lambda x: np.sum(np.minimum(x, 0)) / (np.sum(np.abs(x)) + 1e-8)
    ).fillna(-0.5)

    trend_factor = pd.Series(1.0, index=returns.index)

    up_adjust = 1.0 + np.clip(up_trend * 0.25, 0, 0.12) * vol_ratio  # Was 0.3/0.15
    down_adjust = 1.0 + np.clip(np.abs(down_trend) * 0.4, 0, 0.20) * vol_ratio  # Was 0.5/0.25

    # Apply adjustments based on recent market direction
    for i in range(len(returns)):
        # Current regime
        current_regime = regimes.iloc[i]

        if current_regime == 'crisis':
            regime_multiplier = 1.4
        elif current_regime == 'high_vol':
            regime_multiplier = 1.25
        elif current_regime == 'low_vol':
            regime_multiplier = 0.85
        else:  # normal
            regime_multiplier = 0.95

        # Get recent returns
        recent_ret = returns.iloc[max(0, i - 10):i + 1].mean() if i > 0 else 0

        if recent_ret > 0:
            trend_factor.iloc[i] = up_adjust.iloc[i] * regime_multiplier
        else:
            trend_factor.iloc[i] = down_adjust.iloc[i] * regime_multiplier

    vol_cluster_factor = vol_ratio ** 0.45

    leverage_factor = pd.Series(1.0, index=returns.index)

    for i in range(10, len(returns)):
        neg_returns = returns.iloc[i - 10:i][returns.iloc[i - 10:i] < 0]
        if len(neg_returns) > 0:
            weighted_neg = (np.abs(neg_returns) * np.linspace(0.5, 1.0, len(neg_returns))).sum() / len(neg_returns)
            leverage_factor.iloc[i] = 1.0 + np.clip(weighted_neg / base_vol.iloc[i] * 0.25, 0, 0.15)

    combined_factor = trend_factor * vol_cluster_factor * leverage_factor

    # exponential smoothing
    smooth_factor = combined_factor.ewm(span=5).mean()

    final_adjustment = smooth_factor.clip(0.65, 1.45)

    vol_adjusted = base_vol * final_adjustment

    # Initialize threshold Series for asymmetric thresholds
    big_up_threshold = pd.Series(index=returns.index)
    small_up_threshold = pd.Series(index=returns.index)
    big_down_threshold = pd.Series(index=returns.index)
    small_down_threshold = pd.Series(index=returns.index)

    for regime in regime_thresholds:
        mask = regimes == regime
        thresholds = regime_thresholds[regime]

        # dynamic range based on regime volatility
        if regime == 'crisis':
            lower_bound_up = 0.65
            upper_bound_up = 1.7
            lower_bound_down = 0.6
            upper_bound_down = 1.6
        elif regime == 'high_vol':
            lower_bound_up = 0.7
            upper_bound_up = 1.45
            lower_bound_down = 0.65
            upper_bound_down = 1.4
        elif regime == 'low_vol':
            lower_bound_up = 0.8
            upper_bound_up = 1.25
            lower_bound_down = 0.75
            upper_bound_down = 1.2
        else:  # normal
            lower_bound_up = 0.7
            upper_bound_up = 1.35
            lower_bound_down = 0.65
            upper_bound_down = 1.3

        # Calculate thresholds for this regime
        big_up = (vol_adjusted[mask] * thresholds['big_up']).clip(
            lower=thresholds['big_up'] * lower_bound_up,
            upper=thresholds['big_up'] * upper_bound_up
        )

        small_up = (vol_adjusted[mask] * thresholds['small_up']).clip(
            lower=thresholds['small_up'] * lower_bound_up,
            upper=thresholds['small_up'] * upper_bound_up
        )

        big_down = (vol_adjusted[mask] * thresholds['big_down']).clip(
            lower=thresholds['big_down'] * lower_bound_down,
            upper=thresholds['big_down'] * upper_bound_down
        )

        small_down = (vol_adjusted[mask] * thresholds['small_down']).clip(
            lower=thresholds['small_down'] * lower_bound_down,
            upper=thresholds['small_down'] * upper_bound_down
        )

        # Assign to threshold Series
        big_up_threshold[mask] = big_up
        small_up_threshold[mask] = small_up
        big_down_threshold[mask] = big_down
        small_down_threshold[mask] = small_down

    # Smooth transitions between regimes using exponential smoothing
    big_up_threshold = big_up_threshold.ewm(span=3, min_periods=1).mean()
    small_up_threshold = small_up_threshold.ewm(span=3, min_periods=1).mean()
    big_down_threshold = big_down_threshold.ewm(span=3, min_periods=1).mean()
    small_down_threshold = small_down_threshold.ewm(span=3, min_periods=1).mean()

    return {
        'big_up': big_up_threshold,
        'small_up': small_up_threshold,
        'big_down': big_down_threshold,
        'small_down': small_down_threshold,
        'regime': regimes,
        'adjustment_factor': final_adjustment
    }

def apply_balanced_classification(sp500_data):
    # Get balanced asymmetric thresholds
    thresholds = calculate_adaptive_thresholds(sp500_data['Day_Return'])

    # Apply state classification with asymmetric thresholds
    states = []
    for i in range(len(sp500_data)):
        ret = sp500_data['Day_Return'].iloc[i]

        # Get corresponding threshold values for this specific index
        if ret > 0:
            big_thresh = thresholds['big_up'].iloc[i] if isinstance(thresholds['big_up'], pd.Series) else thresholds[
                'big_up']
            small_thresh = thresholds['small_up'].iloc[i] if isinstance(thresholds['small_up'], pd.Series) else \
            thresholds['small_up']

            if ret > big_thresh:
                state = 'big_up'
            elif ret > small_thresh:
                state = 'small_up'
            else:
                state = 'flat'
        else:  # Negative returns
            big_thresh = thresholds['big_down'].iloc[i] if isinstance(thresholds['big_down'], pd.Series) else \
            thresholds['big_down']
            small_thresh = thresholds['small_down'].iloc[i] if isinstance(thresholds['small_down'], pd.Series) else \
            thresholds['small_down']


            if ret < -big_thresh:
                state = 'big_down'
            elif ret < -small_thresh:
                state = 'small_down'
            else:
                state = 'flat'

        states.append(state)

    sp500_data['Day_State'] = states

    if isinstance(thresholds['big_up'], pd.Series):
        big_up_val = thresholds['big_up'].mean()
        small_up_val = thresholds['small_up'].mean()
        big_down_val = thresholds['big_down'].mean()
        small_down_val = thresholds['small_down'].mean()
    else:
        big_up_val = thresholds['big_up']
        small_up_val = thresholds['small_up']
        big_down_val = thresholds['big_down']
        small_down_val = thresholds['small_down']

    print(f"Balanced thresholds (asymmetric): big_up={big_up_val:.4f}, small_up={small_up_val:.4f}, "
          f"big_down={big_down_val:.4f}, small_down={small_down_val:.4f}")

    print("\nResulting state frequencies:")
    state_freq = sp500_data['Day_State'].value_counts(normalize=True)
    for state, freq in sorted(state_freq.items(), key=lambda x: x[1], reverse=True):
        print(f"{state}: {freq:.1%}")

    return sp500_data, thresholds

# MCMC

def fit_state_models(data):
    state_models = {}
    states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

    # Calculate regime-conditional state probabilities
    regime_state_probs = calculate_regime_state_probabilities(data)

    # Get overall state frequencies for the prior
    state_counts = data['Day_State'].value_counts()
    total_count = len(data)

    for state in states:
        state_data = data[data['Day_State'] == state]
        if len(state_data) < 10:  # Need minimum data points
            continue

        # Extract features for modeling
        features = state_data[['Day_Return', 'Volume_Change', 'Volatility']].astype(float)

        # Calculate basic statistics
        mean_vector = features.mean().values
        cov_matrix = features.cov().values
        cov_matrix = ensure_positive_definite(cov_matrix)
        df = estimate_multivariate_t_df(features)

        # State probability for each regime
        state_regime_probs = {}
        for regime in regime_state_probs:
            state_regime_probs[regime] = regime_state_probs[regime][state]

        state_models[state] = {
            'mean_vector': mean_vector,
            'cov_matrix': cov_matrix,
            'df': df,
            'count': len(state_data),
            'frequency': state_counts.get(state, 0) / total_count,
            'transition_probs': calculate_transition_probs(data, state),
            'regime_probs': state_regime_probs  # Add regime-conditional probabilities
        }

    return state_models

def ensure_positive_definite(cov_matrix, epsilon=1e-6):
    min_eig = np.min(np.real(linalg.eigvals(cov_matrix)))
    if min_eig < epsilon:
        cov_matrix += np.eye(cov_matrix.shape[0]) * (epsilon - min_eig)
    return cov_matrix

def estimate_multivariate_t_df(data, max_df=30):
    # Standardize data
    standardized = (data - data.mean()) / data.std()

    # Mardia's kurtosis
    n_samples, n_features = standardized.shape
    squared_mahalanobis = np.sum(standardized ** 2, axis=1)
    kurtosis = np.mean(squared_mahalanobis ** 2)
    expected_kurt = n_features * (n_features + 2)

    # Estimate df based on kurtosis
    excess_kurtosis = kurtosis - expected_kurt
    if excess_kurtosis <= 0:
        return max_df

    # Higher kurtosis = lower df
    estimated_df = 4 + (8 / excess_kurtosis)
    return min(max(4, estimated_df), max_df)

def calculate_transition_probs(data, from_state):
    states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']
    transition_counts = {state: 0 for state in states}
    total_transitions = 0

    # Find transitions
    state_series = data['Day_State']
    for i in range(len(state_series) - 1):
        if state_series.iloc[i] == from_state:
            next_state = state_series.iloc[i + 1]
            transition_counts[next_state] += 1
            total_transitions += 1

    # Calculate probabilities with smoothing
    alpha = 0.1  # Smoothing parameter
    transition_probs = {}
    for state in states:
        if total_transitions > 0:
            # Apply smoothing to prevent zero probabilities
            transition_probs[state] = (transition_counts[state] + alpha) / (total_transitions + alpha * len(states))
        else:
            transition_probs[state] = 1.0 / len(states)  # Uniform if no data

    return transition_probs

def calculate_regime_state_probabilities(data):
    regimes = data['Market_Regime'].unique()
    states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

    regime_state_probs = {}
    for regime in regimes:
        regime_data = data[data['Market_Regime'] == regime]

        state_counts = regime_data['Day_State'].value_counts()
        total_counts = len(regime_data)

        # Laplace smoothing
        smoothed_probs = {}
        alpha = 1.0
        for state in states:
            count = state_counts.get(state, 0)
            smoothed_probs[state] = (count + alpha) / (total_counts + alpha * len(states))

        regime_state_probs[regime] = smoothed_probs

    return regime_state_probs

def log_likelihood(state_models, state, current_row):
    if state not in state_models:
        return -np.inf

    # Get current market regime (most recent day)
    current_regime = current_row['Market_Regime'].iloc[-1]

    # Base conditional probability P(state|regime)
    if current_regime in state_models[state]['regime_probs']:
        log_prob = np.log(max(state_models[state]['regime_probs'][current_regime], 1e-10))
    else:
        log_prob = np.log(max(state_models[state]['frequency'], 1e-10))

    if len(current_row) > 1:
        # Directional persistence factor
        recent_states = current_row['Day_State'].iloc[:-1].tolist()  # All but the most recent

        if recent_states:
            # Count transitions from recent states to the candidate state
            transition_log_probs = []
            for prev_state in recent_states:
                if prev_state in state_models and 'transition_probs' in state_models[prev_state]:
                    transition_prob = state_models[prev_state]['transition_probs'].get(state, 0.2)
                    transition_log_probs.append(np.log(max(transition_prob, 1e-10)))

            # If we found transition probabilities, average them in log space
            if transition_log_probs:
                log_prob += sum(transition_log_probs) / len(transition_log_probs)

    # 2. Volatility regime consistency
    recent_vol = current_row['Volatility'].iloc[-1]

    # Map states to appropriate volatility regimes
    vol_consistency = {
        'big_up': ['crisis', 'high_vol'],
        'big_down': ['crisis', 'high_vol'],
        'small_up': ['high_vol', 'normal'],
        'small_down': ['high_vol', 'normal'],
        'flat': ['normal', 'low_vol']
    }

    # Calculate volatility consistency factor
    if state in vol_consistency and current_regime in vol_consistency[state]:
        log_prob += np.log(1.2)  # Boost probability if state aligns with vol regime
    else:
        log_prob += np.log(0.8)  # Reduce probability if state doesn't align

    # Calculate short-term momentum and compare with state direction
    if len(current_row) >= 3:  # Need at least 3 days for momentum
        recent_returns = current_row['Day_Return'].values
        short_momentum = np.mean(recent_returns[-3:])

        # Adjust based on alignment of momentum and state
        if ('up' in state and short_momentum > 0) or ('down' in state and short_momentum < 0):
            # State aligns with momentum - increase probability
            momentum_factor = min(1.5, 1.0 + abs(short_momentum) * 10)
            log_prob += np.log(momentum_factor)
        elif ('up' in state and short_momentum < 0) or ('down' in state and short_momentum > 0):
            # State contradicts momentum - decrease probability
            momentum_factor = max(0.5, 1.0 - abs(short_momentum) * 8)
            log_prob += np.log(momentum_factor)

    # Consider volatility trend within the regime
    if len(current_row) >= 5:
        vol_values = current_row['Volatility'].values
        vol_trend = vol_values[-1] / vol_values[0] - 1  # Percent change in volatility

        # Adjust probability based on volatility trend
        if vol_trend > 0.1:  # Rising volatility
            if state in ['big_up', 'big_down']:  # Big moves more likely
                log_prob += np.log(1.2)
            elif state == 'flat':  # Flat less likely
                log_prob += np.log(0.8)
        elif vol_trend < -0.1:  # Falling volatility
            if state == 'flat':  # Flat more likely
                log_prob += np.log(1.2)
            elif state in ['big_up', 'big_down']:  # Big moves less likely
                log_prob += np.log(0.9)

    # Analyze volume pattern for additional signals
    if 'Volume_Change' in current_row.columns and len(current_row) >= 3:
        recent_volume = current_row['Volume_Change'].values[-3:]
        volume_trend = np.mean(recent_volume)

        # Higher volume often precedes bigger moves
        if volume_trend > 0.05:  # Volume increasing
            if 'big' in state:
                log_prob += np.log(1.15)
            elif state == 'flat':
                log_prob += np.log(0.9)
        elif volume_trend < -0.05:  # Volume decreasing
            if state == 'flat':
                log_prob += np.log(1.1)
            elif 'big' in state:
                log_prob += np.log(0.95)

    # Calculate rarity bonus based on historical frequency
    historical_freq = state_models[state]['frequency']
    rarity_bonus = np.log(0.3 + 0.7 * (1 - historical_freq))

    # Final log likelihood
    return log_prob + 0.2 * rarity_bonus

def log_prior():
    return np.log(0.2)

def gamma(x):
    try:
        return scipy.special.gamma(x)
    except:
        if x > 171:
            return np.inf
        return scipy.special.gamma(min(171, x))

def proposal(current_state, state_models, states=None, chain_history=None, iteration=None, temperature=1.0):
    if states is None:
        states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

    # Mixture probability for global vs. local moves
    # Higher temperatures use more global moves to promote exploration
    p_global_base = 0.3  # Base probability for global moves
    p_global = min(0.8, p_global_base * temperature ** 0.5)  # Scale with temperature

    # Detect chain stagnation
    is_stagnant = False
    if chain_history and len(chain_history) >= 3:
        last_states = chain_history[-3:]
        if all(s == current_state for s in last_states):
            is_stagnant = True
            p_global = min(0.9, p_global * 1.5)  # Increase global move probability when stuck

    # Decide whether to make a global or local move
    if np.random.random() < p_global or current_state not in state_models:
        # GLOBAL MOVE: Uniform random selection from all states except current
        other_states = [s for s in states if s != current_state]
        if not other_states:  # Edge case: only one state exists
            return current_state, 1.0

        proposed_state = np.random.choice(other_states)

        # Proposal ratio = 1.0 (symmetric for global moves between different states)
        proposal_ratio = 1.0

    else:
        # LOCAL MOVE: Use transition probabilities from the state model
        transition_probs = state_models[current_state].get('transition_probs', {})

        # Ensure there are transition probabilities for all states
        adjusted_probs = {}
        for state in states:
            # Start with model's transition probabilities
            if state in transition_probs:
                adjusted_probs[state] = transition_probs[state]
            else:
                # If missing, use a small probability
                adjusted_probs[state] = 0.01

        # Apply special case adjustments
        if is_stagnant:
            if current_state in adjusted_probs:
                adjusted_probs[current_state] *= 0.5

        # Normalize probabilities
        total = sum(adjusted_probs.values())
        normalized_probs = {s: p / total for s, p in adjusted_probs.items()}

        # Select state based on normalized probabilities
        states_list = list(normalized_probs.keys())
        probs_list = [normalized_probs[s] for s in states_list]
        proposed_state = np.random.choice(states_list, p=probs_list)

        # Calculate proposal ratio
        if proposed_state in state_models and 'transition_probs' in state_models[proposed_state]:
            reverse_probs = state_models[proposed_state]['transition_probs']

            # Get probability of moving back (proposed→current)
            reverse_prob = reverse_probs.get(current_state, 0.01)

            # Get probability of current move (current→proposed)
            forward_prob = normalized_probs[proposed_state]

            # Calculate ratio
            proposal_ratio = reverse_prob / forward_prob
        else:
            # If no reverse transition model, assume symmetry
            proposal_ratio = 1.0

    return proposed_state, proposal_ratio

def calculate_acceptance_ratio(current_state, proposed_state, proposal_ratio, state_models, current_row, temp):
    # Calculate log posteriors (likelihood × prior)
    current_log_posterior = log_likelihood(state_models, current_state, current_row) + log_prior()
    proposed_log_posterior = log_likelihood(state_models, proposed_state, current_row) + log_prior()

    # Log ratio of posteriors, scaled by temperature
    log_posterior_ratio = (proposed_log_posterior - current_log_posterior) / temp

    # Log of proposal ratio
    log_proposal_ratio = np.log(proposal_ratio + 1e-10)  # Add small constant to avoid log(0)

    # Final acceptance ratio (log scale)
    log_alpha = log_posterior_ratio + log_proposal_ratio

    return log_alpha

def replica_exchange_mcmc(data, current_row, initial_iterations=10000, min_iterations=5000,
                          max_iterations=50000, r_hat_threshold=1.01):

    states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

    # Fit state models (shared across chains)
    state_models = fit_state_models(data)

    temps = np.array([1.0, 5.0, 25.0])
    n_temps = len(temps)

    print(f"Using fixed temperature ladder: {temps}")

    # Equal chain allocation
    chains_per_temp = [4, 4, 4]
    total_chains = sum(chains_per_temp)

    print(f"Chain allocation: {chains_per_temp} (total: {total_chains})")

    # Initialize chains
    chains = [[] for _ in range(total_chains)]
    current_states = []

    # Initialize chains
    for t_idx in range(n_temps):
        for c_idx in range(chains_per_temp[t_idx]):
            state_idx = (t_idx * chains_per_temp[t_idx] + c_idx) % len(states)
            current_states.append(states[state_idx])

    # Verify chain count and distribution
    assert len(current_states) == total_chains, f"Expected {total_chains} states, got {len(current_states)}"

    # Count initial states distribution
    initial_state_counts = {}
    for state in states:
        initial_state_counts[state] = current_states.count(state)
    print(f"Initial state distribution: {initial_state_counts}")

    previous_states = [None] * total_chains

    # Tracking variables
    acceptance_windows = [[] for _ in range(total_chains)]
    state_history = [[] for _ in range(total_chains)]
    swap_attempts = 0
    swap_accepts = 0
    temp_swap_attempts = {i: 0 for i in range(n_temps - 1)}
    temp_swap_accepts = {i: 0 for i in range(n_temps - 1)}

    # Initial burn-in estimate
    initial_burn_in = initial_iterations // 5

    # Create chain index ranges for each temperature
    temp_chain_indices = []
    start_idx = 0
    for t_idx in range(n_temps):
        end_idx = start_idx + chains_per_temp[t_idx]
        temp_chain_indices.append(list(range(start_idx, end_idx)))
        start_idx = end_idx

    # Adaptation parameters
    adaptation_interval = 2000
    temp_swap_rates = {i: 0.0 for i in range(n_temps - 1)}

    # Run for initial iterations
    for i in range(initial_iterations):
        for c in range(total_chains):
            # Get chain's temperature index
            t_idx = next(idx for idx, indices in enumerate(temp_chain_indices) if c in indices)
            temp = temps[t_idx]

            proposed_state, proposal_ratio = proposal(
                current_states[c],
                state_models,
                states,
                chain_history=state_history[c],
                iteration=i,
                temperature=temp
            )

            # Calculate acceptance ratio
            log_alpha = calculate_acceptance_ratio(
                current_states[c],
                proposed_state,
                proposal_ratio,
                state_models,
                current_row,
                temp
            )

            # Accept or reject
            if np.log(np.random.random() + 1e-10) < min(0, log_alpha):
                previous_states[c] = current_states[c]
                current_states[c] = proposed_state
                accepted = 1
            else:
                accepted = 0

            # Update tracking variables
            chains[c].append(current_states[c])
            state_history[c].append(current_states[c])
            if len(state_history[c]) > 100:
                state_history[c] = state_history[c][-100:]

            acceptance_windows[c].append(accepted)
            if len(acceptance_windows[c]) > 100:
                acceptance_windows[c].pop(0)

        # Balanced swap schedule
        for temp_idx in range(n_temps - 1):
            # Consistent swap interval
            base_interval = 15

            # Only attempt swap at appropriate intervals
            if i % base_interval != 0:
                continue

            # Skip if either temperature has no chains
            if not temp_chain_indices[temp_idx] or not temp_chain_indices[temp_idx + 1]:
                continue

            # Select random chains from each temperature level
            chain1_idx = np.random.choice(temp_chain_indices[temp_idx])
            chain2_idx = np.random.choice(temp_chain_indices[temp_idx + 1])

            # Record attempt
            swap_attempts += 1
            temp_swap_attempts[temp_idx] += 1

            # Calculate swap probability
            state1 = current_states[chain1_idx]
            state2 = current_states[chain2_idx]

            log_p1 = (log_likelihood(state_models, state1, current_row) +
                      log_prior())
            log_p2 = (log_likelihood(state_models, state2, current_row) +
                      log_prior())

            # Use correct temperature values
            t1 = temps[temp_idx]
            t2 = temps[temp_idx + 1]

            # Calculate Metropolis ratio for temperature swap
            log_alpha_swap = (1 / t1 - 1 / t2) * (log_p2 - log_p1)

            # Attempt the swap
            if np.log(np.random.random() + 1e-10) < min(0, log_alpha_swap):
                # Swap accepted
                current_states[chain1_idx], current_states[chain2_idx] = current_states[chain2_idx], current_states[
                    chain1_idx]
                swap_accepts += 1
                temp_swap_accepts[temp_idx] += 1

        # Periodic reporting - only monitor, no temperature adaptation
        if i > 0 and i % adaptation_interval == 0:
            # Calculate swap rates (for reporting only)
            temp_swap_rates = {t: temp_swap_accepts[t] / max(1, temp_swap_attempts[t])
                               for t in range(n_temps - 1) if temp_swap_attempts[t] >= 20}

            print(f"\nSwap rates at iteration {i}:")
            for t in range(n_temps - 1):
                if temp_swap_attempts[t] >= 20:
                    print(f"  Temps {temps[t]:.1f} ↔ {temps[t + 1]:.1f}: {temp_swap_rates[t]:.1%}")

            # Calculate state frequencies - for monitoring only
            state_freqs = {}
            for c in temp_chain_indices[0]:  # Base temperature chains
                chain_states = chains[c][-min(1000, len(chains[c])):]  # Recent states
                for s in states:
                    if s not in state_freqs:
                        state_freqs[s] = 0
                    state_freqs[s] += chain_states.count(s)

            total = sum(state_freqs.values())
            if total > 0:
                state_freqs = {s: count / total for s, count in state_freqs.items()}
                print(f"  Current base temp state frequencies: {state_freqs}")

    # Extract results from base temperature chains
    base_temp_chains = [chains[c] for c in temp_chain_indices[0]]

    # Calculate convergence diagnostics
    r_hat, r_hat_diag = calculate_gelman_rubin(base_temp_chains)
    burn_in, burn_in_diag = adaptive_burn_in_detection(base_temp_chains)
    burn_in = max(burn_in, initial_iterations // 10)  # Ensure minimum burn-in

    # Continue sampling if needed for convergence or minimum samples
    total_iterations = initial_iterations
    extension_count = 0

    # Extensions if needed for convergence
    while ((total_iterations < min_iterations) or
           (r_hat > r_hat_threshold)) and total_iterations < max_iterations:
        extension_count += 1
        if extension_count > 3:  # Limit extensions
            print("Maximum extensions reached - proceeding with best available result")
            break

        # Determine extension size
        if r_hat > 1.1:
            additional_iterations = initial_iterations // 2
        elif r_hat > 1.05:
            additional_iterations = initial_iterations // 3
        else:
            additional_iterations = initial_iterations // 4

        additional_iterations = min(additional_iterations, max_iterations - total_iterations)
        if additional_iterations < 500:
            break  # Not worth continuing for small extensions

        print(f"Adding {additional_iterations} iterations, current R-hat: {r_hat:.4f}")

        # Continue sampling with same parameters
        for i in range(additional_iterations):
            # Standard MH updates for each chain
            for c in range(total_chains):
                t_idx = next(idx for idx, indices in enumerate(temp_chain_indices) if c in indices)
                temp = temps[t_idx]

                proposed_state, proposal_ratio = proposal(
                    current_states[c],
                    state_models,
                    states,
                    chain_history=state_history[c],
                    iteration=i,
                    temperature=temp
                )

                # Calculate acceptance ratio
                log_alpha = calculate_acceptance_ratio(
                    current_states[c],
                    proposed_state,
                    proposal_ratio,
                    state_models,
                    current_row,
                    temp
                )

                if np.log(np.random.random() + 1e-10) < min(0, log_alpha):
                    previous_states[c] = current_states[c]
                    current_states[c] = proposed_state
                    accepted = 1
                else:
                    accepted = 0

                chains[c].append(current_states[c])
                state_history[c].append(current_states[c])
                if len(state_history[c]) > 100:
                    state_history[c] = state_history[c][-100:]

                acceptance_windows[c].append(accepted)
                if len(acceptance_windows[c]) > 100:
                    acceptance_windows[c].pop(0)

            # Balanced swap schedule
            for temp_idx in range(n_temps - 1):
                if (total_iterations + i) % 15 != 0:  # Same interval for all pairs
                    continue

                if not temp_chain_indices[temp_idx] or not temp_chain_indices[temp_idx + 1]:
                    continue

                chain1_idx = np.random.choice(temp_chain_indices[temp_idx])
                chain2_idx = np.random.choice(temp_chain_indices[temp_idx + 1])

                swap_attempts += 1
                temp_swap_attempts[temp_idx] += 1

                state1 = current_states[chain1_idx]
                state2 = current_states[chain2_idx]

                log_p1 = (log_likelihood(state_models, state1, current_row) +
                          log_prior())
                log_p2 = (log_likelihood(state_models, state2, current_row) +
                          log_prior())

                t1 = temps[temp_idx]
                t2 = temps[temp_idx + 1]

                log_alpha_swap = (1 / t1 - 1 / t2) * (log_p2 - log_p1)

                if np.log(np.random.random() + 1e-10) < min(0, log_alpha_swap):
                    current_states[chain1_idx], current_states[chain2_idx] = current_states[chain2_idx], current_states[
                        chain1_idx]
                    swap_accepts += 1
                    temp_swap_accepts[temp_idx] += 1

        # Update total iterations
        total_iterations += additional_iterations

        # Recalculate metrics
        base_temp_chains = [chains[c][:total_iterations] for c in temp_chain_indices[0]]
        r_hat, r_hat_diag = calculate_gelman_rubin(base_temp_chains)
        burn_in, _ = adaptive_burn_in_detection(base_temp_chains)
        burn_in = max(burn_in, total_iterations // 10)  # Maintain minimum burn-in

    # Calculate final burn-in for base temperature chains
    optimal_burn_in, _ = adaptive_burn_in_detection(base_temp_chains)
    optimal_burn_in = max(optimal_burn_in, total_iterations // 10)  # Ensure reasonable burn-in

    # Extract final samples from base temperature chains after burn-in
    samples = []
    for chain in base_temp_chains:
        samples.extend(chain[optimal_burn_in:])

    # Calculate acceptance rates for each temperature
    temp_acceptance_rates = {}
    for t in range(n_temps):
        t_chains = temp_chain_indices[t]
        if t_chains:
            acc_rate = np.mean([np.mean(acceptance_windows[c]) for c in t_chains])
            temp_acceptance_rates[temps[t]] = acc_rate

    # Calculate swap acceptance rates for each temperature pair
    swap_acceptance_rates = {t: temp_swap_accepts[t] / max(1, temp_swap_attempts[t])
                             for t in range(n_temps - 1)}

    # Calculate final base temperature chain acceptance rates
    if temp_chain_indices[0]:
        base_temp_acceptance_rates = []
        for c in temp_chain_indices[0]:
            if acceptance_windows[c]:
                base_temp_acceptance_rates.append(np.mean(acceptance_windows[c]))

        if base_temp_acceptance_rates:
            base_temp_mean_acceptance = np.mean(base_temp_acceptance_rates)
            print(f"Base temperature chain mean acceptance rate: {base_temp_mean_acceptance:.1%}")
        else:
            base_temp_mean_acceptance = None
    else:
        base_temp_mean_acceptance = None

    final_diagnostics = {
        'r_hat': r_hat,
        'r_hat_diagnostics': r_hat_diag,
        'optimal_iterations': total_iterations,
        'optimal_burn_in': optimal_burn_in,
        'temperatures': list(temps),
        'temperature_acceptance_rates': temp_acceptance_rates,
        'swap_acceptance_rates_by_temp': swap_acceptance_rates,
        'swap_attempts': swap_attempts,
        'swap_accepts': swap_accepts,
        'overall_swap_acceptance_rate': swap_accepts / max(1, swap_attempts),
        'base_temp_acceptance_rate': base_temp_mean_acceptance,
        'convergence_achieved': r_hat < r_hat_threshold,
        'extension_count': extension_count,
        'chains_per_temperature': chains_per_temp
    }

    return samples, final_diagnostics

def calculate_gelman_rubin(chains):
    # Convert categorical states to numerical values if they're strings
    state_map = {
        'big_up': 2,
        'small_up': 1,
        'flat': 0,
        'small_down': -1,
        'big_down': -2
    }

    numerical_chains = []
    for chain in chains:
        # Check if chain already contains numerical values
        if isinstance(chain[0], (int, np.integer, float, np.floating)):
            numerical_chains.append(np.array(chain))
        else:
            # Convert string states to numerical values
            try:
                numerical_chain = np.array([state_map[state] for state in chain])
                numerical_chains.append(numerical_chain)
            except KeyError as e:
                raise KeyError(f"Unknown state {e} in chain. Valid states are: {list(state_map.keys())}")

    # Number of chains and samples per chain
    m = len(chains)
    n = min(len(chain) for chain in chains)  # Use minimum length for balanced calculation

    if n < 20:  # Minimum sample requirement
        return float('inf'), {'error': 'Insufficient samples for reliable R-hat calculation'}

    # Truncate chains to equal length
    truncated_chains = [chain[:n] for chain in numerical_chains]

    # Add tiny noise to break exact zeros in discrete chains
    noisy_chains = []
    for chain in truncated_chains:
        # Add tiny random noise (1e-6) to prevent exact zeros in variance calculations
        noise = np.random.normal(0, 1e-6, size=len(chain))
        noisy_chains.append(chain + noise)

    truncated_chains = noisy_chains

    # Split each chain in half to detect non-stationarity
    split_chains = []
    for chain in truncated_chains:
        mid = len(chain) // 2
        split_chains.append(chain[:mid])
        split_chains.append(chain[mid:])

    # Calculate within-chain and between-chain variance using both
    # the original chains and the split chains

    # Original chains
    chain_means_orig = np.array([np.mean(chain) for chain in truncated_chains])
    overall_mean_orig = np.mean(chain_means_orig)

    # Split chains
    chain_means_split = np.array([np.mean(chain) for chain in split_chains])
    overall_mean_split = np.mean(chain_means_split)

    # Between-chain variance (original chains)
    B_orig = (n / (m - 1)) * np.sum((chain_means_orig - overall_mean_orig) ** 2) if m > 1 else 0

    # Between-chain variance (split chains)
    m_split = len(split_chains)
    n_split = len(split_chains[0])
    B_split = (n_split / (m_split - 1)) * np.sum((chain_means_split - overall_mean_split) ** 2) if m_split > 1 else 0

    # Within-chain variances
    epsilon = 1e-8  # Small value to prevent zero variance
    W_orig = max(epsilon, np.mean([np.var(chain, ddof=1) for chain in truncated_chains]))
    W_split = max(epsilon, np.mean([np.var(chain, ddof=1) for chain in split_chains]))

    # Calculate weighted average of within and between variance
    var_plus_orig = ((n - 1) / n) * W_orig + (1 / n) * B_orig
    var_plus_split = ((n_split - 1) / n_split) * W_split + (1 / n_split) * B_split

    # Calculate R-hat values with safeguards against extreme values
    R_hat_orig = np.sqrt(var_plus_orig / W_orig)
    R_hat_split = np.sqrt(var_plus_split / W_split)

    # Cap R-hat at a reasonable maximum value
    max_rhat_value = 2.0
    R_hat_orig = min(R_hat_orig, max_rhat_value)
    R_hat_split = min(R_hat_split, max_rhat_value)

    # Calculate quantile-based R-hat for more robust assessment
    def calc_quantile_rhat(chains, q):
        quantiles = np.array([np.quantile(chain, q) for chain in truncated_chains])
        quantile_mean = np.mean(quantiles)
        between_var = np.var(quantiles, ddof=1) if len(quantiles) > 1 else 0

        within_vars = []
        for i, chain in enumerate(truncated_chains):
            # Get values near the quantile for this chain
            target = quantiles[i]
            # Find closest values to the target quantile
            idx = np.argsort(np.abs(chain - target))[:max(5, len(chain) // 10)]
            values = chain[idx]
            # Add epsilon to prevent zero variance
            within_vars.append(max(epsilon, np.var(values, ddof=1) if len(values) > 1 else epsilon))

        within_var = np.mean(within_vars) if within_vars else epsilon

        var_plus = within_var + between_var * (n + 1) / (m * n)
        rhat = np.sqrt(var_plus / within_var)
        return min(rhat, max_rhat_value)  # Apply the same cap

    # Calculate R-hat for 10% and 90% quantiles
    rhat_low = calc_quantile_rhat(truncated_chains, 0.1)
    rhat_high = calc_quantile_rhat(truncated_chains, 0.9)

    # Use rank-normalized split R-hat (more robust for discrete distributions)
    def rank_normalized_split_rhat(chains):
        if len(chains) < 2:
            return max_rhat_value

        # Pool all chains
        pooled = np.concatenate(chains)

        # Get ranks of all values
        from scipy import stats
        ranks = stats.rankdata(pooled)

        # Split rankings back into chains
        chain_lengths = [len(chain) for chain in chains]
        rank_chains = np.split(ranks, np.cumsum(chain_lengths)[:-1])

        # Calculate split R-hat on the ranks
        rank_means = [np.mean(rc) for rc in rank_chains]
        overall_rank_mean = np.mean(rank_means)

        # Between-chain variance
        B_rank = sum((rm - overall_rank_mean) ** 2 for rm in rank_means) * len(pooled) / (len(chains) - 1)

        # Within-chain variance
        W_rank = max(epsilon, np.mean([np.var(rc, ddof=1) for rc in rank_chains]))

        # Weighted average
        var_plus_rank = ((len(pooled) - 1) / len(pooled)) * W_rank + (1 / len(pooled)) * B_rank

        # Final R-hat
        rhat = np.sqrt(var_plus_rank / W_rank)
        return min(rhat, max_rhat_value)  # Apply the same cap

    rank_rhat = rank_normalized_split_rhat(truncated_chains)

    # Max of all R-hat values
    max_rhat = max(R_hat_orig, R_hat_split, rhat_low, rhat_high, rank_rhat)

    # Final diagnostics dictionary with all metrics
    diagnostics = {
        'standard_rhat': R_hat_orig,
        'split_rhat': R_hat_split,
        'rank_rhat': rank_rhat,
        'quantile_rhat_low': rhat_low,
        'quantile_rhat_high': rhat_high,
        'final_rhat': max_rhat,
        'between_chain_variance': B_orig,
        'within_chain_variance': W_orig,
        'pooled_variance': var_plus_orig,
        'chain_means': list(chain_means_orig),
        'chain_std': [float(np.std(chain)) for chain in truncated_chains],
        'reliability': 'high' if n > 100 and max_rhat < 1.05 else 'medium' if n > 50 and max_rhat < 1.1 else 'low'
    }

    return max_rhat, diagnostics

def adaptive_burn_in_detection(chains, window_sizes=[100, 200, 500], threshold=0.05):
    # Convert chains to numerical form
    state_map = {
        'big_up': 2,
        'small_up': 1,
        'flat': 0,
        'small_down': -1,
        'big_down': -2
    }

    # Check if chains are already numerical
    if isinstance(chains[0][0], (int, np.integer, float, np.floating)):
        numerical_chains = chains
    else:
        numerical_chains = []
        for chain in chains:
            numerical_chain = np.array([state_map[state] for state in chain])
            numerical_chains.append(numerical_chain)

    min_length = min(len(chain) for chain in numerical_chains)
    r_hats = []

    # Calculate R-hat for increasing prefixes of the chains
    min_check = min(100, min_length // 10)
    check_points = []
    current = min_check
    while current < min_length:
        check_points.append(current)
        current = min(min_length, int(current * 1.2))  # 20% increase each step

    for i in check_points:
        prefix_chains = [chain[:i] for chain in numerical_chains]

        # Convert prefix_chains to string representation if needed
        string_state_map = {v: k for k, v in state_map.items()}

        # Only convert if chains are numeric
        if isinstance(chains[0][0], (int, np.integer, float, np.floating)):
            string_prefix_chains = []
            for chain in prefix_chains:
                # Convert numeric values back to string states
                string_chain = []
                for val in chain:
                    # Handle potential rounding issues
                    closest_key = min(state_map.values(), key=lambda x: abs(x - val))
                    string_chain.append(string_state_map[closest_key])
                string_prefix_chains.append(string_chain)
            prefix_chains_for_gelman = string_prefix_chains
        else:
            # If chains are already string states, use as is
            prefix_chains_for_gelman = prefix_chains

        r_hat, _ = calculate_gelman_rubin(prefix_chains_for_gelman)
        r_hats.append((i, r_hat))

    # Find where R-hat first goes below thresholds
    convergence_points = {
        'strict': min([i for i, r_hat in r_hats if r_hat < 1.02], default=min_length),
        'standard': min([i for i, r_hat in r_hats if r_hat < 1.05], default=min_length),
        'lenient': min([i for i, r_hat in r_hats if r_hat < 1.1], default=min_length)
    }

    # Enhanced stationarity tests
    stationarity_results = {}
    for window in window_sizes:
        if window >= min_length // 2:
            continue

        # For each window size, test multiple statistics
        mean_stationary_points = []
        var_stationary_points = []
        ks_stationary_points = []

        for chain in numerical_chains:
            # Skip short chains
            if len(chain) < window * 2:
                continue

            # Test increasing segments for mean stationarity
            for i in range(window, len(chain) - window, window):
                segment1 = chain[i - window:i]
                segment2 = chain[i:i + window]

                # Test for mean stationarity
                t_stat, p_val = stats.ttest_ind(segment1, segment2, equal_var=False)
                if p_val > threshold:
                    mean_stationary_points.append(i)
                    break

            # Test for variance stationarity
            for i in range(window, len(chain) - window, window):
                segment1 = chain[i - window:i]
                segment2 = chain[i:i + window]

                # Levene test for variance equality
                _, p_val = stats.levene(segment1, segment2)
                if p_val > threshold:
                    var_stationary_points.append(i)
                    break

            # Kolmogorov-Smirnov test for distribution stationarity
            for i in range(window, len(chain) - window, window):
                segment1 = chain[i - window:i]
                segment2 = chain[i:i + window]

                # Non-parametric test for distribution equality
                _, p_val = stats.ks_2samp(segment1, segment2)
                if p_val > threshold:
                    ks_stationary_points.append(i)
                    break

        if mean_stationary_points:
            stationarity_results[f'mean_{window}'] = int(np.median(mean_stationary_points))
        if var_stationary_points:
            stationarity_results[f'var_{window}'] = int(np.median(var_stationary_points))
        if ks_stationary_points:
            stationarity_results[f'ks_{window}'] = int(np.median(ks_stationary_points))

    # Combine convergence evidence
    if stationarity_results:
        # Weight results by test type and window size
        weighted_sum = 0
        total_weight = 0

        for test_name, point in stationarity_results.items():
            # Parse test type and window size
            test_type, window_str = test_name.split('_')
            window = int(window_str)

            # Weights based on test type and window size
            if test_type == 'ks':
                # KS test is strongest
                weight = window * 2.0
            elif test_type == 'mean':
                # Mean test is standard
                weight = window * 1.5
            else:  # var
                # Variance test gets lower weight
                weight = window * 1.0

            weighted_sum += point * weight
            total_weight += weight

        # Combined stationarity point
        stationarity_point = int(weighted_sum / total_weight) if total_weight > 0 else min_length // 2

        r_hat_point = convergence_points['standard']

        burn_in = min(r_hat_point, stationarity_point)
    else:
        # Fallback to R-hat only
        burn_in = convergence_points['standard']

    burn_in = max(burn_in, min(500, min_length // 10))

    # Cap at 40% of chain length
    burn_in = min(burn_in, min_length * 2 // 5)

    return burn_in, {
        'r_hat_convergence': convergence_points,
        'stationarity_points': stationarity_results,
        'r_hat_progression': r_hats,
        'min_chain_length': min_length,
        'final_burn_in': burn_in,
        'burn_in_fraction': burn_in / min_length
    }

def get_next_market_day(current_date):
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=current_date.strftime('%Y-%m-%d'), end=(current_date + timedelta(days=10)).strftime('%Y-%m-%d'))
    next_day = pd.Timestamp(current_date.strftime('%Y-%m-%d')) + BDay(1)
    while next_day in holidays or next_day.weekday() >= 5:
        next_day = next_day + BDay(1)
    return next_day

def get_prediction_confidence(mcmc_samples):
    all_states = ['big_up', 'small_up', 'flat', 'small_down', 'big_down']

    # Calculate raw counts and probabilities
    state_counts = pd.Series(mcmc_samples).value_counts()
    total_samples = len(mcmc_samples)
    raw_probs = {state: state_counts.get(state, 0) / total_samples for state in all_states}

    # Print raw distribution
    print("\nState Counts:")
    for state in all_states:
        count = state_counts.get(state, 0)
        print(f"{state}: {count} samples ({count / total_samples:.1%})")

    # Extract predictions using raw probabilities
    sorted_states = sorted([(prob, state) for state, prob in raw_probs.items()], reverse=True)
    primary_state = sorted_states[0][1]
    primary_prob = sorted_states[0][0]
    secondary_state = sorted_states[1][1]
    secondary_prob = sorted_states[1][0]

    # Calculate prediction certainty
    uniform_prob = 1.0 / len(all_states)
    uniform_dist = {state: uniform_prob for state in all_states}

    # KL divergence for certainty calculation
    def kl_divergence(p, q):
        return sum(p[s] * np.log(p[s] / q[s]) for s in p.keys() if p[s] > 0)

    # Jensen-Shannon divergence from uniform
    m_dist = {s: (raw_probs[s] + uniform_dist[s]) / 2 for s in all_states}
    js_div = 0.5 * kl_divergence(raw_probs, m_dist) + 0.5 * kl_divergence(uniform_dist, m_dist)

    # Normalize to 0-1 scale
    max_js_div = np.log(len(all_states))
    certainty = js_div / max_js_div

    # Simple certainty measure
    simple_certainty = primary_prob - secondary_prob

    return {
        'primary_state': primary_state,
        'primary_probability': primary_prob,
        'secondary_state': secondary_state,
        'secondary_probability': secondary_prob,
        'prediction_certainty': certainty,
        'simple_certainty': simple_certainty,
        'state_probabilities': raw_probs,
        'raw_probabilities': raw_probs
    }

def calculate_state_metrics(states, returns):
    if len(states) == 0:
        return {
            'state_distribution': pd.Series(),
            'state_stability': np.nan,
            'extreme_capture': np.nan,
            'false_signals': np.nan
        }

    state_series = pd.Series(states)
    state_distribution = state_series.value_counts(normalize=True)

    # Calculate state stability (how often state remains the same)
    transitions = 0
    for i in range(1, len(state_series)):
        if state_series.iloc[i] != state_series.iloc[i - 1]:
            transitions += 1
    state_stability = 1 - (transitions / (len(state_series) - 1)) if len(state_series) > 1 else np.nan

    # Calculate extreme capture (accuracy during large market moves)
    big_moves = (abs(returns) > returns.quantile(0.9))
    correct_signals = 0
    total_big_moves = sum(big_moves)

    if total_big_moves > 0:
        for i, is_big_move in enumerate(big_moves):
            if is_big_move:
                is_correct = (returns.iloc[i] > 0 and 'up' in state_series.iloc[i]) or \
                             (returns.iloc[i] < 0 and 'down' in state_series.iloc[i])
                if is_correct:
                    correct_signals += 1
        extreme_capture = correct_signals / total_big_moves
    else:
        extreme_capture = np.nan

    # Calculate false signals (incorrect big/small signals)
    false_signals = 0
    total_signals = 0

    for i, state in enumerate(state_series):
        if state != 'flat':
            total_signals += 1
            is_false = (('up' in state and returns.iloc[i] < 0) or
                        ('down' in state and returns.iloc[i] > 0))
            if is_false:
                false_signals += 1

    false_signal_rate = false_signals / total_signals if total_signals > 0 else np.nan

    return {
        'state_distribution': state_distribution,
        'state_stability': state_stability,
        'extreme_capture': extreme_capture,
        'false_signals': false_signal_rate
    }

# Apply
def analyze_market_data(start_date='2010-01-01', use_replica_exchange=True):
    current_date = datetime.now()
    next_market_day = get_next_market_day(current_date)

    # Fetch S&P 500 data
    sp500 = yf.download('^GSPC', start=start_date,
                        end=(current_date + timedelta(days=1)).strftime('%Y-%m-%d'))

    # (current_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Calculate returns
    sp500['Day_Return'] = sp500['Close'] / sp500['Open'] - 1
    sp500['Volume_Change'] = sp500['Volume'].pct_change()
    sp500['Volatility'] = sp500['Day_Return'].ewm(span=20, min_periods=5).std()
    sp500['Market_Regime'] = detect_market_regime(sp500['Day_Return'])

    # Apply balanced classification
    sp500, thresholds = apply_balanced_classification(sp500)

    print("\nMarket Regime Analysis for last 50 days:")
    regime_counts = sp500['Market_Regime'].tail(50).value_counts()
    print(regime_counts)

    print("\nCurrent Market Regime:", sp500['Market_Regime'].iloc[-1])

    print(sp500.iloc[-20:])
    print("\nDate Range:")
    print("First Date:", sp500.index[0])
    print("Last Date:", sp500.index[-1])

    # Remove NaN rows
    sp500 = sp500.dropna()

    metrics = calculate_state_metrics(sp500['Day_State'], sp500['Day_Return'])

    # Make prediction for next market day
    N = 5
    current_row = sp500.iloc[-N:].copy()
    current_regime = current_row['Market_Regime'].iloc[-1]

    # Determine MCMC parameters based on market conditions
    if use_replica_exchange:
        if current_regime == 'crisis':
            initial_iterations = 16000
            r_hat_threshold = 1.01
        elif current_regime == 'high_vol':
            initial_iterations = 12000
            r_hat_threshold = 1.02
        elif current_regime == 'low_vol':
            initial_iterations = 6000
            r_hat_threshold = 1.03
        else:  # normal
            initial_iterations = 8000
            r_hat_threshold = 1.02

        # Fine-tune based on recent stability
        state_stability = metrics['state_stability']
        if state_stability is not None and not np.isnan(state_stability):
            if state_stability < 0.4:  # Less stable markets
                initial_iterations = int(initial_iterations * 1.25)
            elif state_stability > 0.6:  # More stable markets
                initial_iterations = int(initial_iterations * 0.9)

        mcmc_samples, mcmc_diagnostics = replica_exchange_mcmc(
            data=sp500,
            current_row=current_row,
            initial_iterations=initial_iterations,
            r_hat_threshold=r_hat_threshold
        )
    else:
        raise Exception("Only replica exchange MCMC is supported in this version")

    confidence = get_prediction_confidence(mcmc_samples)

    # Calculate average threshold values for reporting
    if isinstance(thresholds['big_up'], pd.Series):
        threshold_values = {
            'big_up': thresholds['big_up'].iloc[-1],
            'small_up': thresholds['small_up'].iloc[-1],
            'big_down': thresholds['big_down'].iloc[-1],
            'small_down': thresholds['small_down'].iloc[-1]
        }
    else:
        threshold_values = {
            'big_up': thresholds['big_up'],
            'small_up': thresholds['small_up'],
            'big_down': thresholds['big_down'],
            'small_down': thresholds['small_down']
        }

    results = {
        'prediction': confidence['primary_state'],
        'confidence': confidence,
        'next_market_day': next_market_day,
        'replica_exchange_used': use_replica_exchange,
        'mcmc_diagnostics': mcmc_diagnostics,
        'data': sp500,
        'thresholds': threshold_values
    }

    # Fix swap rate calculation if needed
    if results['replica_exchange_used'] and 'overall_swap_acceptance_rate' in mcmc_diagnostics:
        if mcmc_diagnostics['overall_swap_acceptance_rate'] == 0 and mcmc_diagnostics['swap_accepts'] > 0:
            corrected_rate = mcmc_diagnostics['swap_accepts'] / max(1, mcmc_diagnostics['swap_attempts'])
            mcmc_diagnostics['overall_swap_acceptance_rate'] = corrected_rate
            print(f"Note: Fixed swap rate calculation from 0 to {corrected_rate:.1%}")

    # Print diagnostic information
    if results['replica_exchange_used']:
        print(f"R-hat: {results['mcmc_diagnostics'].get('r_hat', 'N/A'):.6f}")
        print(f"Overall swap acceptance rate: {results['mcmc_diagnostics'].get('overall_swap_acceptance_rate', 0):.1%}")

    # Print current regime and thresholds
    print("\nCurrent Market Regime:", results['data']['Market_Regime'].iloc[-1])
    print(f"Thresholds: big_up={threshold_values['big_up']:.4f}, small_up={threshold_values['small_up']:.4f}, "
          f"big_down={threshold_values['big_down']:.4f}, small_down={threshold_values['small_down']:.4f}")

    return results

if __name__ == "__main__":
    # Run with more detailed console output
    print("\n=== Market Prediction Model ===\n")
    print("Running analysis with improved MCMC replica exchange...")

    start_time = time.time()

    # Run the analysis
    results = analyze_market_data(use_replica_exchange=True)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds")

    # Print summary header
    print("\n" + "=" * 50)
    print(f"MARKET PREDICTION FOR {results['next_market_day'].strftime('%A, %B %d, %Y').upper()}")
    print("=" * 50)

    # Print prediction with confidence
    print(f"\nPrimary prediction: {results['prediction']} "
          f"(probability: {results['confidence']['primary_probability']:.1%})")
    print(f"Secondary prediction: {results['confidence']['secondary_state']} "
          f"(probability: {results['confidence']['secondary_probability']:.1%})")
    print(f"Prediction certainty: {results['confidence']['prediction_certainty']:.1%}")

    # Print full state probabilities in descending order
    print("\nState probabilities (ranked):")
    for state, prob in sorted(results['confidence']['state_probabilities'].items(),
                              key=lambda x: x[1], reverse=True):
        print(f"{state}: {prob:.1%}")

    # Print technical diagnostics
    print("\n" + "-" * 20)
    print("TECHNICAL DIAGNOSTICS")
    print("-" * 20)
    print(f"Method used: {'Replica Exchange' if results['replica_exchange_used'] else 'Traditional'}")
    print(f"Optimal iterations: {results['mcmc_diagnostics']['optimal_iterations']}")
    print(f"Optimal burn-in: {results['mcmc_diagnostics']['optimal_burn_in']}")

    if results['replica_exchange_used']:
        print(f"R-hat: {results['mcmc_diagnostics'].get('r_hat', 'N/A'):.6f}")
        print(f"Overall swap acceptance rate: {results['mcmc_diagnostics'].get('overall_swap_acceptance_rate', 0):.1%}")
        if 'base_temp_acceptance_rate' in results['mcmc_diagnostics']:
            print(f"Base chain proposal acceptance rate: {results['mcmc_diagnostics']['base_temp_acceptance_rate']:.1%}")

    # Print regime and asymmetric thresholds
    print("\nCurrent Market Regime:", results['data']['Market_Regime'].iloc[-1])
    print(f"Thresholds: big_up={results['thresholds']['big_up']:.4f}, small_up={results['thresholds']['small_up']:.4f},")
    print(f"           big_down={results['thresholds']['big_down']:.4f}, small_down={results['thresholds']['small_down']:.4f}")

    # Print disclaimer
    print("\nNOTE: This is a statistical model and should not be used as financial advice.")
    print("Market conditions can change rapidly and unpredictably.")
