import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')

from PredictorV2 import (
    HiddenMarkovModel,
    DiscriminativeSpectralPredictor,
    calculate_features,
    select_features
)

class Backtest:

    def __init__(self, start_date='2015-01-01', end_date=None, retrain_frequency='M'):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.retrain_frequency = retrain_frequency

        self.predictions = []  # Probability of up movement
        self.confidences = []  # Prediction confidence
        self.uncertainties = []  # Model uncertainty
        self.actuals = []  # Actual directions (0/1)
        self.returns = []  # Actual returns for strategy testing
        self.dates = []
        self.regime_history = []
        self.selected_features_history = []

    def prepare_data(self, data_start='2010-01-01'):
        print("Fetching market data...")

        end_date_extended = (pd.Timestamp(self.end_date) + timedelta(days=30)).strftime('%Y-%m-%d')
        daily_data = yf.download('^GSPC', start=data_start, end=end_date_extended, progress=False)

        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_data.columns = daily_data.columns.get_level_values(0)

        # Convert to weekly data with Friday alignment
        self.weekly_data = daily_data.resample('W-FRI').agg({
            daily_data.columns[0]: 'first',
            daily_data.columns[1]: 'max',
            daily_data.columns[2]: 'min',
            daily_data.columns[3]: 'last',
            daily_data.columns[4]: 'sum'
        }).dropna()

        self.weekly_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Returns and directions
        self.weekly_data['returns'] = self.weekly_data['Close'].pct_change()
        self.weekly_data['direction'] = (self.weekly_data['returns'] > 0).astype(int)

        print(f"Prepared weekly data: {len(self.weekly_data)} weeks")
        print(f"Direction distribution: {self.weekly_data['direction'].value_counts().to_dict()}")

        # Enhanced features
        print("Calculating features...")
        self.features = calculate_features(self.weekly_data['returns'], self.weekly_data['Close'],
                                           self.weekly_data['Volume'])

        # HMM features
        self.hmm_features = pd.DataFrame({
            'returns': self.weekly_data['returns'],
            'volatility': self.features['vol_20d'],
            'momentum': self.features['ret_4w']
        }).dropna()

        print(f"Generated {len(self.features.columns)} features")

    def train_models(self, train_end_date):
        # Training data
        train_mask = self.hmm_features.index <= train_end_date
        hmm_train = self.hmm_features[train_mask].copy()

        if len(hmm_train) < 60:  # Need more data for robust training
            raise ValueError(f"Insufficient training data: {len(hmm_train)} samples")

        # Scale HMM features
        scaler_hmm = RobustScaler()
        hmm_train_scaled = scaler_hmm.fit_transform(hmm_train)
        hmm_train_scaled = pd.DataFrame(hmm_train_scaled, columns=hmm_train.columns, index=hmm_train.index)

        # Train HMM
        hmm = HiddenMarkovModel(n_states=3, n_features=3)
        hmm.fit(hmm_train_scaled.values)

        # Regime probabilities for training data
        regime_probs_all = []
        for i in range(len(hmm_train_scaled)):
            if i < 10:
                regime_probs_all.append(hmm.pi)
            else:
                obs_window = hmm_train_scaled.values[max(0, i - 20):i + 1]
                regime_probs = hmm.predict_proba(obs_window)
                regime_probs_all.append(regime_probs)

        regime_probs_all = np.array(regime_probs_all)

        # Direction prediction data
        direction_target = self.weekly_data['direction'].loc[:train_end_date]
        features_train = self.features.loc[:train_end_date]

        # Lagged features to prevent look-ahead bias
        features_lagged = features_train.shift(1)

        common_index = features_lagged.index.intersection(direction_target.index)
        X_full = features_lagged.loc[common_index]
        y_direction = direction_target.loc[common_index]

        valid_mask = ~(X_full.isnull().any(axis=1) | y_direction.isnull())
        X_clean = X_full[valid_mask]
        y_clean = y_direction[valid_mask]

        # feature selection
        X_selected, selected_features = select_features(X_clean, y_clean, n_features=15)

        # alignment with HMM data
        final_common_index = X_selected.index.intersection(hmm_train.index)
        X_train = X_selected.loc[final_common_index]
        y_train = y_clean.loc[final_common_index]
        regime_probs_train = regime_probs_all[:len(final_common_index)]

        print(f"Training models on {len(X_train)} samples with {len(selected_features)} features")

        # Scale features for spectral analysis
        scaler_X = RobustScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)

        # Train Enhanced Discriminative Spectral predictor
        spectral_predictor = DiscriminativeSpectralPredictor(n_components=8, n_regimes=3)
        spectral_predictor.fit(X_train_scaled, y_train.values, regime_probs_train)

        return {
            'hmm': hmm,
            'spectral_predictor': spectral_predictor,
            'scaler_hmm': scaler_hmm,
            'scaler_X': scaler_X,
            'selected_features': selected_features,
            'hmm_features': hmm_train.columns.tolist()
        }

    def make_prediction(self, models, prediction_date):
        # recent HMM data
        recent_mask = self.hmm_features.index < prediction_date
        recent_hmm = self.hmm_features[recent_mask].tail(20)

        if len(recent_hmm) < 10:
            raise ValueError("Insufficient recent data for prediction")

        # Scale HMM features
        hmm_scaled = models['scaler_hmm'].transform(recent_hmm)
        current_regime_probs = models['hmm'].predict_proba(hmm_scaled)

        # prediction features
        feature_mask = self.features.index < prediction_date
        features_available = self.features[feature_mask]

        if len(features_available) == 0:
            raise ValueError("No features available for prediction")

        recent_features = features_available[models['selected_features']].tail(1)
        recent_features_scaled = models['scaler_X'].transform(recent_features)

        # prediction
        prob_up, confidence = models['spectral_predictor'].predict_proba(
            recent_features_scaled,
            current_regime_probs.reshape(1, -1)
        )

        prob_up = prob_up[0]
        confidence = confidence[0]
        uncertainty = 1.0 - confidence

        return {
            'prob_up': prob_up,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'regime_probs': current_regime_probs,
            'regime': np.argmax(current_regime_probs)
        }

    def run_backtest(self, min_train_years=4):
        self.prepare_data()

        backtest_start = pd.Timestamp(self.start_date)
        backtest_end = pd.Timestamp(self.end_date)

        min_train_date = backtest_start - pd.DateOffset(years=min_train_years)
        if self.weekly_data.index[0] > min_train_date:
            print(f"Warning: Limited training data. First data point: {self.weekly_data.index[0]}")

        if self.retrain_frequency == 'M':
            retrain_dates = pd.date_range(backtest_start, backtest_end, freq='MS')
        elif self.retrain_frequency == 'Q':
            retrain_dates = pd.date_range(backtest_start, backtest_end, freq='QS')
        elif self.retrain_frequency == 'W':
            retrain_dates = pd.date_range(backtest_start, backtest_end, freq='W-FRI')

        print(f"Running backtest from {backtest_start.date()} to {backtest_end.date()}")
        print(f"Retraining frequency: {self.retrain_frequency}")

        current_models = None
        last_train_date = None
        trading_weeks = self.weekly_data.loc[backtest_start:backtest_end].index

        for i, week_date in enumerate(trading_weeks[:-1]):
            retrain_needed = False
            if current_models is None:
                retrain_needed = True
            else:
                next_retrain = retrain_dates[retrain_dates > last_train_date]
                if len(next_retrain) > 0 and week_date >= next_retrain[0]:
                    retrain_needed = True

            if retrain_needed:
                print(f"\nRetraining at {week_date.date()}...")
                train_end = week_date - pd.DateOffset(weeks=1)
                try:
                    current_models = self.train_models(train_end)
                    last_train_date = week_date
                except Exception as e:
                    print(f"Training failed at {week_date}: {e}")
                    continue

            try:
                # Make prediction
                prediction = self.make_prediction(current_models, week_date)

                # Get actual result (next week)
                next_week_idx = trading_weeks.get_loc(week_date) + 1
                if next_week_idx < len(trading_weeks):
                    actual_date = trading_weeks[next_week_idx]
                    actual_direction = int(self.weekly_data['direction'].loc[actual_date])
                    actual_return = float(self.weekly_data['returns'].loc[actual_date])

                    # Store results with proper alignment
                    self.predictions.append(float(prediction['prob_up']))
                    self.confidences.append(float(prediction['confidence']))
                    self.uncertainties.append(float(prediction['uncertainty']))
                    self.actuals.append(actual_direction)
                    self.returns.append(actual_return)
                    self.dates.append(actual_date)
                    self.regime_history.append(int(prediction['regime']))
                    self.selected_features_history.append(current_models['selected_features'])

                    if i % 20 == 0:
                        print(f"Processed {i + 1}/{len(trading_weeks) - 1} weeks...")

            except Exception as e:
                print(f"Prediction failed for {week_date}: {e}")
                continue

        print(f"\nBacktest completed: {len(self.predictions)} directional predictions made")

        # Convert to arrays
        self.predictions = np.array(self.predictions)
        self.confidences = np.array(self.confidences)
        self.uncertainties = np.array(self.uncertainties)
        self.actuals = np.array(self.actuals)
        self.returns = np.array(self.returns)
        self.dates = pd.DatetimeIndex(self.dates)
        self.regime_history = np.array(self.regime_history)

    def calculate_metrics(self):
        if len(self.predictions) == 0:
            return {}

        metrics = {}

        # Convert probabilities to predictions
        pred_directions = (self.predictions > 0.5).astype(int)

        accuracy = np.mean(pred_directions == self.actuals)
        metrics['directional_accuracy'] = accuracy

        # confidence-weighted accuracy
        confidence_threshold = np.percentile(self.confidences, 70)
        high_conf_mask = self.confidences > confidence_threshold
        if np.sum(high_conf_mask) > 0:
            metrics['high_confidence_accuracy'] = np.mean(
                pred_directions[high_conf_mask] == self.actuals[high_conf_mask]
            )
            metrics['high_confidence_count'] = np.sum(high_conf_mask)
        else:
            metrics['high_confidence_accuracy'] = np.nan
            metrics['high_confidence_count'] = 0

        # low confidence accuracy
        low_conf_mask = self.confidences <= np.percentile(self.confidences, 30)
        if np.sum(low_conf_mask) > 0:
            metrics['low_confidence_accuracy'] = np.mean(
                pred_directions[low_conf_mask] == self.actuals[low_conf_mask]
            )
        else:
            metrics['low_confidence_accuracy'] = np.nan

        # Regime-specific accuracy
        regime_names = ['Low Vol', 'Normal', 'High Vol']
        for i in range(3):
            regime_mask = self.regime_history == i
            if np.sum(regime_mask) > 0:
                regime_acc = np.mean(pred_directions[regime_mask] == self.actuals[regime_mask])
                metrics[f'{regime_names[i]}_accuracy'] = regime_acc
                metrics[f'{regime_names[i]}_count'] = np.sum(regime_mask)

        # Probabilistic metrics
        if len(np.unique(self.actuals)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(self.actuals, self.predictions)
            except:
                metrics['auc_roc'] = np.nan

            # Brier Score
            brier_score = np.mean((self.predictions - self.actuals) ** 2)
            metrics['brier_score'] = brier_score

            # Log loss
            eps = 1e-15
            pred_clipped = np.clip(self.predictions, eps, 1 - eps)
            log_loss = -np.mean(self.actuals * np.log(pred_clipped) +
                                (1 - self.actuals) * np.log(1 - pred_clipped))
            metrics['log_loss'] = log_loss

        try:
            # calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.actuals, self.predictions, n_bins=10, strategy='uniform'
            )

            # ECE
            bin_sizes = []
            for i in range(10):
                lower = i / 10
                upper = (i + 1) / 10
                if i == 9:
                    mask = (self.predictions >= lower) & (self.predictions <= upper)
                else:
                    mask = (self.predictions >= lower) & (self.predictions < upper)
                bin_sizes.append(np.sum(mask))

            bin_sizes = np.array(bin_sizes)
            total_samples = len(self.predictions)

            ece = 0
            for i, (acc, conf, size) in enumerate(zip(fraction_of_positives, mean_predicted_value, bin_sizes)):
                if size > 0:
                    ece += (size / total_samples) * abs(acc - conf)

            metrics['expected_calibration_error'] = ece
            metrics['calibration_curve'] = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'bin_sizes': bin_sizes
            }

        except Exception as e:
            print(f"Calibration calculation error: {e}")
            metrics['expected_calibration_error'] = np.nan

        # pred dist metrics
        metrics['prediction_mean'] = np.mean(self.predictions)
        metrics['prediction_std'] = np.std(self.predictions)
        metrics['prediction_extremes'] = np.mean((self.predictions < 0.1) | (self.predictions > 0.9))

        try:
            uncertainty_quartiles = np.percentile(self.uncertainties, [25, 50, 75])

            for i, (q_name, threshold) in enumerate([
                ('Low', uncertainty_quartiles[0]),
                ('Medium', uncertainty_quartiles[1]),
                ('High', uncertainty_quartiles[2])
            ]):
                if i == 0:
                    mask = self.uncertainties <= threshold
                elif i == 1:
                    mask = (self.uncertainties > uncertainty_quartiles[0]) & (self.uncertainties <= threshold)
                else:
                    mask = self.uncertainties > threshold

                if np.sum(mask) > 0:
                    acc = np.mean(pred_directions[mask] == self.actuals[mask])
                    metrics[f'{q_name}_uncertainty_accuracy'] = acc
                    metrics[f'{q_name}_uncertainty_count'] = np.sum(mask)

        except Exception as e:
            print(f"Uncertainty calibration error: {e}")

        return metrics

    def test_strategies(self,report):

        if report:
            print("\n" + "=" * 80)
            print("TRADING STRATEGY COMPARISON")
            print("=" * 80)

        strategies = {}

        # Simple directional strategy
        simple_signals = np.where(self.predictions > 0.5, 1, -1)
        strategies['Simple Directional'] = simple_signals

        # High confidence strategy (using Bayesian confidence)
        enhanced_signals = np.zeros_like(self.predictions)
        high_conf_threshold = np.percentile(self.confidences, 75)  # Top 25%
        high_conf_mask = self.confidences > high_conf_threshold
        enhanced_signals[high_conf_mask] = np.where(
            self.predictions[high_conf_mask] > 0.5, 1, -1
        )
        strategies['High-Conf'] = enhanced_signals

        # Uncertainty-filtered strategy
        uncertainty_signals = np.zeros_like(self.predictions)
        low_uncertainty_threshold = np.percentile(self.uncertainties, 25)  # Bottom 25% uncertainty
        low_uncertainty_mask = self.uncertainties < low_uncertainty_threshold
        uncertainty_signals[low_uncertainty_mask] = np.where(
            self.predictions[low_uncertainty_mask] > 0.5, 1, -1
        )
        strategies['Low Uncertainty'] = uncertainty_signals

        # Regime-aware Bayesian strategy
        regime_signals = np.zeros_like(self.predictions)
        for i, (pred, conf, regime, uncertainty) in enumerate(zip(
                self.predictions, self.confidences, self.regime_history, self.uncertainties
        )):
            if regime == 0:
                conf_threshold = 0.65
                uncertainty_threshold = 0.3
            elif regime == 1:
                conf_threshold = 0.70
                uncertainty_threshold = 0.25
            else:
                conf_threshold = 0.75
                uncertainty_threshold = 0.2

            if conf > conf_threshold and uncertainty < uncertainty_threshold:
                regime_signals[i] = 1 if pred > 0.5 else -1
        strategies['Regime-Aware Bayesian'] = regime_signals

        # Bayesian Kelly criterion strategy
        kelly_signals = np.zeros_like(self.predictions)
        for i, (pred, conf, uncertainty) in enumerate(zip(self.predictions, self.confidences, self.uncertainties)):
            if conf > 0.7 and uncertainty < 0.25:
                uncertainty_discount = max(0.1, 1.0 - uncertainty)

                if pred > 0.5:
                    kelly_fraction = min((2 * pred - 1) * conf * uncertainty_discount, 1.0)
                    kelly_signals[i] = kelly_fraction
                else:
                    kelly_fraction = min((1 - 2 * pred) * conf * uncertainty_discount, 1.0)
                    kelly_signals[i] = -kelly_fraction
        strategies['Bayesian Kelly'] = kelly_signals

        # Multi-factor Bayesian strategy
        multi_factor_signals = np.zeros_like(self.predictions)
        for i, (pred, conf, regime, uncertainty) in enumerate(zip(
                self.predictions, self.confidences, self.regime_history, self.uncertainties
        )):
            base_signal = 2 * pred - 1
            confidence_weight = conf
            uncertainty_discount = max(0.1, 1.0 - uncertainty)

            regime_adjustment = [1.1, 1.0, 0.9][regime]

            composite_score = base_signal * confidence_weight * uncertainty_discount * regime_adjustment

            # Threshold for trading
            if abs(composite_score) > 0.4:
                multi_factor_signals[i] = np.sign(composite_score)

        strategies['Multi-Factor Bayesian'] = multi_factor_signals

        results = {}
        for name, signals in strategies.items():
            results[name] = self.calculate_performance(signals)

        # buy & hold benchmark
        buy_hold_signals = np.ones_like(self.returns)
        results['Buy & Hold'] = self.calculate_performance(buy_hold_signals, 0)

        # sort by Sharpe
        sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe_ratio'], reverse=True)

        if report:
            print(
                f"{'Strategy':<25} {'Total Ret':<10} {'Volatility':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8} {'Hit Rate':<8}")
            print("-" * 95)

            for name, metrics in sorted_results:
                print(f"{name:<25} {metrics['total_return'] * 100:>8.1f}% "
                    f"{metrics['volatility'] * 100:>8.1f}% "
                    f"{metrics['sharpe_ratio']:>6.2f} "
                    f"{metrics['max_drawdown'] * 100:>6.1f}% "
                    f"{metrics['num_trades']:>6.0f} "
                    f"{metrics['hit_rate']:>6.1%}")

        return results

    def calculate_performance(self, signals, transaction_cost=0.0003):
        if len(signals) != len(self.returns):
            raise ValueError("Signal length must match returns length")

        strategy_returns = signals * self.returns

        # transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], signals])))
        transaction_costs = position_changes * transaction_cost
        strategy_returns[1:] -= transaction_costs[1:]

        # Performance
        total_return = np.prod(1 + strategy_returns) - 1
        volatility = np.std(strategy_returns) * np.sqrt(52)

        if np.std(strategy_returns) > 0:
            sharpe = np.sqrt(52) * np.mean(strategy_returns) / np.std(strategy_returns)
        else:
            sharpe = 0

        # Max drawdown
        cum_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        positive_returns = strategy_returns > 0
        hit_rate = np.mean(positive_returns) if len(positive_returns) > 0 else 0

        num_trades = np.sum(position_changes > 0.01)

        # Calmar ratio
        calmar = abs(total_return / max_drawdown) if max_drawdown < 0 else 0

        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'num_trades': num_trades,
            'calmar_ratio': calmar,
            'strategy_returns': strategy_returns
        }

    def create_visualizations(self):
        if len(self.predictions) == 0:
            print("No predictions available for visualization")
            return None

        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Market Prediction Analysis', fontsize=16, fontweight='bold')

        # Prediction Probability Distribution
        ax = axes[0, 0]
        ax.hist(self.predictions, bins=30, alpha=0.7, edgecolor='black', density=True)
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.set_xlabel('Probability of Up Movement')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        mean_prob = np.mean(self.predictions)
        std_prob = np.std(self.predictions)
        extremes_pct = np.mean((self.predictions < 0.1) | (self.predictions > 0.9))
        ax.text(0.05, 0.95, f'Mean: {mean_prob:.3f}\nStd: {std_prob:.3f}\nExtremes: {extremes_pct:.1%}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax = axes[0, 1]

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                self.actuals, self.predictions, n_bins=10
            )

            ax.plot(mean_predicted_value, fraction_of_positives, 'bo-', label='Discriminative Model', markersize=6)
            ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title('Calibration Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        except Exception as e:
            ax.text(0.5, 0.5, f'Calibration plot error: {str(e)[:50]}...',
                    transform=ax.transAxes, ha='center', va='center')

        # Strategy Performance
        ax = axes[1, 0]
        strategy_results = self.test_strategies(False)

        # Plot top strategies
        top_strategies = sorted(strategy_results.items(),
                                key=lambda x: x[1]['sharpe_ratio'], reverse=True)[:4]

        for name, metrics in top_strategies:
            if 'strategy_returns' in metrics:
                cum_returns = np.cumprod(1 + metrics['strategy_returns'])
                ax.plot(self.dates, (cum_returns - 1) * 100,
                        label=name, linewidth=2, alpha=0.8)

        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Top Strategy Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # uncertainty vs accuracy analysis
        ax = axes[1, 1]

        # uncertainty bins and accuracy in each
        uncertainty_bins = np.linspace(0, np.max(self.uncertainties), 11)
        bin_centers = (uncertainty_bins[:-1] + uncertainty_bins[1:]) / 2
        bin_accuracies = []
        bin_counts = []

        pred_directions = (self.predictions > 0.5).astype(int)

        for i in range(len(uncertainty_bins) - 1):
            if i == len(uncertainty_bins) - 2:  # Last bin
                mask = (self.uncertainties >= uncertainty_bins[i]) & (self.uncertainties <= uncertainty_bins[i + 1])
            else:
                mask = (self.uncertainties >= uncertainty_bins[i]) & (self.uncertainties < uncertainty_bins[i + 1])

            if np.sum(mask) > 0:
                accuracy = np.mean(pred_directions[mask] == self.actuals[mask])
                bin_accuracies.append(accuracy)
                bin_counts.append(np.sum(mask))
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)

        # Plot
        valid_mask = ~np.isnan(bin_accuracies)
        ax.plot(bin_centers[valid_mask], np.array(bin_accuracies)[valid_mask], 'ro-', label='Observed')
        ax.axhline(0.5, color='blue', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Uncertainty vs Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Confidence vs Accuracy
        ax = axes[2, 0]

        # confidence bins and accuracy in each
        conf_bins = np.linspace(0.5, 1, 11)  # Start from 0.5 since that's our minimum
        bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
        bin_accuracies = []

        for i in range(len(conf_bins) - 1):
            if i == len(conf_bins) - 2:  # Last bin
                mask = (self.confidences >= conf_bins[i]) & (self.confidences <= conf_bins[i + 1])
            else:
                mask = (self.confidences >= conf_bins[i]) & (self.confidences < conf_bins[i + 1])

            if np.sum(mask) > 0:
                accuracy = np.mean(pred_directions[mask] == self.actuals[mask])
                bin_accuracies.append(accuracy)
            else:
                bin_accuracies.append(np.nan)

        # Plot
        valid_mask = ~np.isnan(bin_accuracies)
        ax.plot(bin_centers[valid_mask], np.array(bin_accuracies)[valid_mask], 'go-', label='Observed')
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Confidence vs Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Regime Analysis
        ax = axes[2, 1]
        regime_names = ['Low Vol', 'Normal', 'High Vol']

        regime_accuracies = []
        regime_counts = []
        for i in range(3):
            regime_mask = self.regime_history == i
            if np.sum(regime_mask) > 0:
                acc = np.mean(pred_directions[regime_mask] == self.actuals[regime_mask])
                regime_accuracies.append(acc * 100)
                regime_counts.append(np.sum(regime_mask))
            else:
                regime_accuracies.append(0)
                regime_counts.append(0)

        colors = ['blue', 'orange', 'red']
        bars = ax.bar(regime_names, regime_accuracies, color=colors, alpha=0.7)
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.set_xlabel('Market Regime')
        ax.set_ylabel('Directional Accuracy (%)')
        ax.set_title('Accuracy by Regime')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for bar, acc, count in zip(bars, regime_accuracies, regime_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{acc:.1f}%\n({count})', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def generate_report(self):
        if len(self.predictions) == 0:
            print("No predictions available. Run backtest first.")
            return

        metrics = self.calculate_metrics()

        print("\n" + "=" * 80)
        print("MARKET PREDICTION PERFORMANCE REPORT")
        print("=" * 80)

        print(f"\nBacktest Period: {self.dates[0].date()} to {self.dates[-1].date()}")
        print(f"Total Predictions: {len(self.predictions)}")
        print(f"Retraining Frequency: {self.retrain_frequency}")

        print("\n--- BAYESIAN SPECTRAL ACCURACY METRICS ---")
        print(f"Overall Accuracy: {metrics['directional_accuracy']:.1%}")

        if not np.isnan(metrics['high_confidence_accuracy']):
            print(f"High Confidence Accuracy: {metrics['high_confidence_accuracy']:.1%} "
                  f"({metrics['high_confidence_count']} predictions)")
        if not np.isnan(metrics['low_confidence_accuracy']):
            print(f"Low Confidence Accuracy: {metrics['low_confidence_accuracy']:.1%}")

        print("\n--- PROBABILISTIC METRICS ---")
        if 'auc_roc' in metrics and not np.isnan(metrics['auc_roc']):
            print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
        if 'brier_score' in metrics:
            print(f"Brier Score: {metrics['brier_score']:.3f} (lower is better)")
        if 'log_loss' in metrics:
            print(f"Log Loss: {metrics['log_loss']:.3f} (lower is better)")

        print("\n--- PREDICTION DISTRIBUTION ---")
        print(f"Mean Prediction: {metrics['prediction_mean']:.3f}")
        print(f"Prediction Std: {metrics['prediction_std']:.3f}")
        print(f"Extreme Predictions (<10% or >90%): {metrics['prediction_extremes']:.1%}")

        print("\n--- BAYESIAN CALIBRATION ---")
        if 'expected_calibration_error' in metrics and not np.isnan(metrics['expected_calibration_error']):
            print(f"Expected Calibration Error: {metrics['expected_calibration_error']:.3f}")

        print("\n--- UNCERTAINTY ANALYSIS ---")
        for uncertainty_level in ['Low', 'Medium', 'High']:
            if f'{uncertainty_level}_uncertainty_accuracy' in metrics:
                acc = metrics[f'{uncertainty_level}_uncertainty_accuracy']
                count = metrics[f'{uncertainty_level}_uncertainty_count']
                print(f"{uncertainty_level} Uncertainty: {count} predictions ({acc:.1%} accuracy)")

        print("\n--- REGIME ANALYSIS ---")
        regime_names = ['Low Vol', 'Normal', 'High Vol']
        for i, name in enumerate(regime_names):
            if f'{name}_count' in metrics and metrics[f'{name}_count'] > 0:
                accuracy = metrics[f'{name}_accuracy']
                count = metrics[f'{name}_count']
                print(f"{name}: {count} weeks ({accuracy:.1%} accuracy)")

        return metrics


if __name__ == "__main__":
    print("Running Weekly Market Predictor Backtest")
    print("=" * 80)

    # Run backtest
    backtest = Backtest(
        start_date='2022-01-01',
        end_date='2023-01-01',
        retrain_frequency='W'
    )

    backtest.run_backtest(min_train_years=4)

    metrics = backtest.generate_report()

    strategy_results = backtest.test_strategies(True)

    fig = backtest.create_visualizations()
    if fig:
        plt.show()

    # Summary
    best_strategy = max(strategy_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"\nBEST DISCRIMINATIVE STRATEGY: {best_strategy[0]}")
    print(f"   Sharpe Ratio: {best_strategy[1]['sharpe_ratio']:.3f}")
    print(f"   Total Return: {best_strategy[1]['total_return'] * 100:.1f}%")
    print(f"   Max Drawdown: {best_strategy[1]['max_drawdown'] * 100:.1f}%")

    print(f"\nDISCRIMINATIVE STATS:")
    print(f"   Directional accuracy: {metrics['directional_accuracy']:.1%}")
    if 'auc_roc' in metrics and not np.isnan(metrics['auc_roc']):
        print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
    if 'expected_calibration_error' in metrics and not np.isnan(metrics['expected_calibration_error']):
        print(f"   Calibration error: {metrics['expected_calibration_error']:.3f}")
    if 'prediction_extremes' in metrics:
        print(f"   Extreme predictions: {metrics['prediction_extremes']:.1%}")

