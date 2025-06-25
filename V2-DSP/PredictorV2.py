import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy.linalg import eigh
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')


class HiddenMarkovModel:

    def __init__(self, n_states=3, n_features=3):
        self.n_states = n_states
        self.n_features = n_features

        self.pi = np.ones(n_states) / n_states
        self.A = self.initialize_transition_matrix()
        self.means = None
        self.covs = None

    def initialize_transition_matrix(self):
        A = np.eye(self.n_states) * 0.8
        for i in range(self.n_states):
            for j in range(self.n_states):
                if i != j:
                    A[i, j] = 0.2 / (self.n_states - 1)
        return A

    def multivariate_normal_pdf(self, x, mean, cov):
        d = len(mean)
        diff = x - mean

        cov_reg = cov + 1e-2 * np.eye(d)

        try:
            inv_cov = np.linalg.inv(cov_reg)
            det_cov = np.linalg.det(cov_reg)
        except:
            return 1e-10

        if det_cov <= 0:
            return 1e-10

        mahal = np.dot(np.dot(diff, inv_cov), diff)
        return np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** d * det_cov)

    def forward_backward(self, observations):
        T = len(observations)
        log_alpha = np.zeros((T, self.n_states))
        log_beta = np.zeros((T, self.n_states))

        # Forward pass
        for j in range(self.n_states):
            log_alpha[0, j] = np.log(self.pi[j] + 1e-10) + np.log(self.emission_prob(observations[0], j) + 1e-10)

        for t in range(1, T):
            for j in range(self.n_states):
                log_probs = log_alpha[t - 1] + np.log(self.A[:, j] + 1e-10)
                log_alpha[t, j] = np.logaddexp.reduce(log_probs) + np.log(
                    self.emission_prob(observations[t], j) + 1e-10)

        # Backward pass
        log_beta[T - 1] = 0
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_probs = (np.log(self.A[i] + 1e-10) +
                             np.array([np.log(self.emission_prob(observations[t + 1], j) + 1e-10)
                                       for j in range(self.n_states)]) + log_beta[t + 1])
                log_beta[t, i] = np.logaddexp.reduce(log_probs)

        return log_alpha, log_beta

    def emission_prob(self, observation, state):
        return self.multivariate_normal_pdf(observation, self.means[state], self.covs[state])

    def fit(self, observations, max_iter=50, tol=1e-3):
        n_samples = len(observations)
        observations = np.array(observations)

        valid_mask = ~np.any(np.isnan(observations), axis=1)
        observations = observations[valid_mask]
        n_samples = len(observations)

        if n_samples < 20:
            raise ValueError("Too few valid observations for HMM training")

        # k-means initialization
        kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=5)
        labels = kmeans.fit_predict(observations)

        self.means = np.zeros((self.n_states, self.n_features))
        self.covs = np.zeros((self.n_states, self.n_features, self.n_features))

        for i in range(self.n_states):
            mask = labels == i
            if np.sum(mask) > 1:
                self.means[i] = observations[mask].mean(axis=0)
                cov = np.cov(observations[mask].T)
                self.covs[i] = cov + 0.1 * np.eye(self.n_features)
            else:
                # fallback for empty clusters
                self.means[i] = observations.mean(axis=0) + np.random.randn(self.n_features) * 0.1
                self.covs[i] = np.cov(observations.T) + 0.1 * np.eye(self.n_features)

        # EM algo with early stopping
        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            try:
                log_alpha, log_beta = self.forward_backward(observations)

                # gamma and convert
                log_gamma = log_alpha + log_beta
                log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)

                # Calc xi
                xi = np.zeros((n_samples - 1, self.n_states, self.n_states))
                for t in range(n_samples - 1):
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            log_xi = (log_alpha[t, i] + np.log(self.A[i, j] + 1e-10) +
                                      np.log(self.emission_prob(observations[t + 1], j) + 1e-10) +
                                      log_beta[t + 1, j])
                            xi[t, i, j] = np.exp(log_xi)
                    xi_sum = xi[t].sum()
                    if xi_sum > 0:
                        xi[t] /= xi_sum

                # M-step
                self.pi = gamma[0] + 0.01

                for i in range(self.n_states):
                    for j in range(self.n_states):
                        num = xi[:, i, j].sum() + 0.01
                        den = gamma[:-1, i].sum() + 0.01
                        self.A[i, j] = num / den

                # update emission params
                for j in range(self.n_states):
                    weight_sum = gamma[:, j].sum()
                    if weight_sum > 1e-3:
                        self.means[j] = np.sum(gamma[:, j, np.newaxis] * observations, axis=0) / weight_sum

                        diff = observations - self.means[j]
                        weighted_cov = np.zeros((self.n_features, self.n_features))
                        for t in range(n_samples):
                            weighted_cov += gamma[t, j] * np.outer(diff[t], diff[t])
                        self.covs[j] = weighted_cov / weight_sum + 0.1 * np.eye(self.n_features)

                # Check convergence
                log_likelihood = np.sum(np.logaddexp.reduce(log_alpha[-1]))
                if abs(log_likelihood - prev_log_likelihood) < tol:
                    print(f"HMM converged after {iteration + 1} iterations")
                    break
                prev_log_likelihood = log_likelihood

            except Exception as e:
                print(f"HMM training error at iteration {iteration}: {e}")
                break

        return self

    def predict_proba(self, observations):
        log_alpha, _ = self.forward_backward(observations)
        log_probs = log_alpha[-1]
        log_probs -= np.logaddexp.reduce(log_probs)
        return np.exp(log_probs)


class DiscriminativeSpectralPredictor:

    def __init__(self, n_components=8, n_regimes=3):
        self.n_components = n_components
        self.n_regimes = n_regimes
        self.is_fitted = False

        self.discriminant_eigenvalues = None
        self.discriminant_eigenvectors = None
        self.feature_means = None
        self.feature_stds = None

        self.global_alpha = 1.0
        self.global_beta = 1.0
        self.regime_alphas = np.ones(n_regimes)
        self.regime_betas = np.ones(n_regimes)

        self.regime_models = []

        self.confidence_calibrator = None

    def estimate_empirical_bayes_hyperparams(self, y_direction, regime_probs):
        print("Estimating empirical Bayes hyperparameters...")

        # global hyperparameters
        overall_mean = np.mean(y_direction)
        overall_var = np.var(y_direction)

        # beta distribution method of moments
        if overall_var > 0 and overall_mean > 0 and overall_mean < 1:
            if overall_var < overall_mean * (1 - overall_mean):
                self.global_alpha = overall_mean * (overall_mean * (1 - overall_mean) / overall_var - 1)
                self.global_beta = (1 - overall_mean) * (overall_mean * (1 - overall_mean) / overall_var - 1)
            else:
                self.global_alpha = 2.0
                self.global_beta = 2.0
        else:
            self.global_alpha = 2.0
            self.global_beta = 2.0

        # regime-specific hyperparameters
        for r in range(self.n_regimes):
            regime_weights = regime_probs[:, r]
            regime_mean = np.average(y_direction, weights=regime_weights)

            regime_var = np.average((y_direction - regime_mean) ** 2, weights=regime_weights)

            if regime_var > 0 and regime_mean > 0 and regime_mean < 1:
                if regime_var < regime_mean * (1 - regime_mean):
                    self.regime_alphas[r] = regime_mean * (regime_mean * (1 - regime_mean) / regime_var - 1)
                    self.regime_betas[r] = (1 - regime_mean) * (regime_mean * (1 - regime_mean) / regime_var - 1)
                else:
                    self.regime_alphas[r] = 2.0
                    self.regime_betas[r] = 2.0
            else:
                self.regime_alphas[r] = 2.0
                self.regime_betas[r] = 2.0

    def compute_discriminative_spectral_decomposition(self, X, y_direction):
        # Center data
        self.feature_means = np.mean(X, axis=0)
        X_centered = X - self.feature_means

        # standardization
        self.feature_stds = np.std(X_centered, axis=0) + 1e-6
        X_normalized = X_centered / self.feature_stds

        class_0_mask = y_direction == 0
        class_1_mask = y_direction == 1

        X_class_0 = X_normalized[class_0_mask]
        X_class_1 = X_normalized[class_1_mask]

        if len(X_class_0) < 5 or len(X_class_1) < 5:
            print("Warning: Too few samples per class, falling back to PCA")
            return self.fallback_pca_decomposition(X_normalized)

        mean_0 = np.mean(X_class_0, axis=0)
        mean_1 = np.mean(X_class_1, axis=0)

        S_w = np.zeros((X_normalized.shape[1], X_normalized.shape[1]))

        for x in X_class_0:
            diff = x - mean_0
            S_w += np.outer(diff, diff)

        for x in X_class_1:
            diff = x - mean_1
            S_w += np.outer(diff, diff)

        # Between-class scatter matrix
        mean_diff = mean_1 - mean_0
        S_b = len(X_class_0) * len(X_class_1) / len(X_normalized) * np.outer(mean_diff, mean_diff)

        S_w += 1e-4 * np.eye(S_w.shape[0])

        try:
            eigenvalues, eigenvectors = eigh(S_b, S_w)

            idx = np.argsort(eigenvalues)[::-1]
            self.discriminant_eigenvalues = eigenvalues[idx]
            self.discriminant_eigenvectors = eigenvectors[:, idx]

            n_components_to_keep = min(self.n_components, len(eigenvalues))
            self.discriminant_eigenvalues = self.discriminant_eigenvalues[:n_components_to_keep]
            self.discriminant_eigenvectors = self.discriminant_eigenvectors[:, :n_components_to_keep]

        except Exception as e:
            print(f"Fisher's LDA failed: {e}, falling back to PCA")
            return self.fallback_pca_decomposition(X_normalized)

        return self.transform_discriminative_features(X)

    def fallback_pca_decomposition(self, X_normalized):
        cov_matrix = np.cov(X_normalized.T)
        cov_matrix += 1e-4 * np.eye(cov_matrix.shape[0])

        eigenvalues, eigenvectors = eigh(cov_matrix)
        idx = np.argsort(eigenvalues)[::-1]

        self.discriminant_eigenvalues = eigenvalues[idx][:self.n_components]
        self.discriminant_eigenvectors = eigenvectors[:, idx][:, :self.n_components]

        return X_normalized @ self.discriminant_eigenvectors

    def transform_discriminative_features(self, X):
        X_centered = X - self.feature_means
        X_normalized = X_centered / self.feature_stds
        return X_normalized @ self.discriminant_eigenvectors

    def fit_confidence_calibrator(self, predictions, actuals):

        try:
            # Convert predictions to confidence scores
            confidence_scores = np.abs(predictions - 0.5) * 2

            # Actual accuracy for each prediction
            pred_directions = (predictions > 0.5).astype(int)
            actual_correctness = (pred_directions == actuals).astype(float)

            self.confidence_calibrator = IsotonicRegression(out_of_bounds='clip')

            if len(np.unique(confidence_scores)) > 5:
                self.confidence_calibrator.fit(confidence_scores, actual_correctness)
                print("Confidence calibrator fitted successfully")
            else:
                self.confidence_calibrator = None
                print("Insufficient data diversity for confidence calibration")

        except Exception as e:
            print(f"Confidence calibration failed: {e}")
            self.confidence_calibrator = None

    def fit_regime_model(self, X_spectral, y_direction, regime_weights, regime_idx):

        weights = regime_weights + 1e-6

        # prior params
        prior_alpha = self.regime_alphas[regime_idx]
        prior_beta = self.regime_betas[regime_idx]

        # posterior params
        weighted_successes = np.sum(weights * y_direction)
        weighted_total = np.sum(weights)

        posterior_alpha = prior_alpha + weighted_successes
        posterior_beta = prior_beta + weighted_total - weighted_successes

        n_features = X_spectral.shape[1]

        prior_precision = np.ones(n_features) * 1.0
        noise_precision = 1.0

        S_inv = np.diag(prior_precision) + noise_precision * (X_spectral.T @ np.diag(weights) @ X_spectral)
        S = np.linalg.inv(S_inv)

        # Posterior mean
        m = noise_precision * S @ X_spectral.T @ (weights * y_direction)

        feature_importance = 1.0 / (np.diag(S) + 1e-6) #inverse of variance

        return {
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'feature_weights': m,
            'feature_covariance': S,
            'feature_importance': feature_importance,
            'regime_mean': posterior_alpha / (posterior_alpha + posterior_beta),
            'regime_confidence': np.sqrt(posterior_alpha * posterior_beta /
                                         ((posterior_alpha + posterior_beta) ** 2 * (
                                                     posterior_alpha + posterior_beta + 1)))
        }

    def fit(self, X, y_direction, regime_probs):
        n_samples, n_features = X.shape

        self.estimate_empirical_bayes_hyperparams(y_direction, regime_probs)

        # Fisher's LDA
        X_spectral = self.compute_discriminative_spectral_decomposition(X, y_direction)

        # fit regime-specific models
        self.regime_models = []
        for r in range(self.n_regimes):
            regime_weights = regime_probs[:, r]
            model = self.fit_regime_model(X_spectral, y_direction, regime_weights, r)
            self.regime_models.append(model)

        print("Fitting confidence calibrator...")

        calibration_predictions = []
        calibration_actuals = []

        calibration_indices = np.random.choice(len(X), size=min(200, len(X)), replace=False)

        for i in calibration_indices:
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[i] = False

            mini_pred = self.quick_predict_for_calibration(X_spectral[train_mask], y_direction[train_mask],
                                                           regime_probs[train_mask], X_spectral[i:i + 1],
                                                           regime_probs[i:i + 1])

            calibration_predictions.append(mini_pred)
            calibration_actuals.append(y_direction[i])

        self.fit_confidence_calibrator(np.array(calibration_predictions), np.array(calibration_actuals))

        self.is_fitted = True
        return self

    def quick_predict_for_calibration(self, X_train, y_train, regime_train, X_test, regime_test):
        try:
            regime_weights = regime_test[0]

            prediction = 0
            for r in range(self.n_regimes):
                regime_mask = np.argmax(regime_train, axis=1) == r
                if np.sum(regime_mask) > 5:
                    regime_mean = np.mean(y_train[regime_mask])
                    prediction += regime_weights[r] * regime_mean
                else:
                    prediction += regime_weights[r] * np.mean(y_train)

            return prediction
        except:
            return 0.5  # Fallback

    def predict_proba(self, X, regime_probs):
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        n_samples = X.shape[0]

        X_spectral = self.transform_discriminative_features(X)

        predictions = np.zeros(n_samples)
        epistemic_uncertainties = np.zeros(n_samples)
        aleatoric_uncertainties = np.zeros(n_samples)

        for i in range(n_samples):
            regime_pred = np.zeros(self.n_regimes)
            regime_var = np.zeros(self.n_regimes)

            for r in range(self.n_regimes):
                model = self.regime_models[r]

                linear_pred = X_spectral[i] @ model['feature_weights']

                regime_baseline = model['regime_mean']
                discrimination_boost = np.sum(X_spectral[i] * model['feature_importance'][:len(X_spectral[i])]) * 0.1

                enhanced_logit = linear_pred + np.log(
                    regime_baseline / (1 - regime_baseline + 1e-6)) + discrimination_boost
                prob = self.calibrated_sigmoid(enhanced_logit)

                pred_var = X_spectral[i] @ model['feature_covariance'] @ X_spectral[i]
                pred_var *= (1 + discrimination_boost ** 2)

                regime_pred[r] = prob
                regime_var[r] = pred_var

            # Bayesian model averaging
            regime_weights = regime_probs[i]
            predictions[i] = np.sum(regime_weights * regime_pred)

            epistemic_uncertainties[i] = np.sqrt(np.sum(regime_weights * (regime_pred - predictions[i]) ** 2))
            aleatoric_uncertainties[i] = np.sqrt(np.sum(regime_weights * regime_var))

        raw_uncertainties = np.sqrt(epistemic_uncertainties ** 2 + aleatoric_uncertainties ** 2)

        calibrated_confidences = np.zeros(n_samples)

        for i in range(n_samples):
            raw_confidence = np.abs(predictions[i] - 0.5) * 2

            if self.confidence_calibrator is not None:
                try:
                    calibrated_confidence = self.confidence_calibrator.predict([raw_confidence])[0]
                    calibrated_confidences[i] = np.clip(calibrated_confidence, 0.5, 0.95)
                except:
                    calibrated_confidences[i] = np.clip(raw_confidence, 0.5, 0.95)
            else:
                #fallback: use uncertainty-based confidence
                uncertainty_confidence = 1.0 / (1.0 + raw_uncertainties[i])
                calibrated_confidences[i] = np.clip(uncertainty_confidence, 0.5, 0.95)

        # bounds check
        predictions = np.clip(predictions, 0.01, 0.99)
        raw_uncertainties = np.clip(raw_uncertainties, 0.001, 0.5)

        return predictions, calibrated_confidences

    def calibrated_sigmoid(self, x):
        x_clipped = np.clip(x, -15, 15)
        return 1.0 / (1.0 + np.exp(-x_clipped))


def calculate_features(returns, prices, volume, window=20):
    features = pd.DataFrame(index=returns.index)

    returns = returns.fillna(0)
    prices = prices.fillna(method='ffill')
    volume = volume.fillna(method='ffill')

    # VOLATILITY FEATURES
    features['vol_5d'] = returns.rolling(5).std() * np.sqrt(52)
    features['vol_10d'] = returns.rolling(10).std() * np.sqrt(52)
    features['vol_20d'] = returns.rolling(20).std() * np.sqrt(52)
    features['vol_60d'] = returns.rolling(60).std() * np.sqrt(52)

    # Volatility ratios and persistence
    features['vol_ratio_short'] = features['vol_5d'] / (features['vol_20d'] + 1e-6)
    features['vol_ratio_long'] = features['vol_20d'] / (features['vol_60d'] + 1e-6)
    features['vol_acceleration'] = (features['vol_5d'] - features['vol_10d']) / (features['vol_10d'] + 1e-6)

    # MOMENTUM FEATURES
    features['ret_1w'] = returns.rolling(1).sum()
    features['ret_2w'] = returns.rolling(2).sum()
    features['ret_4w'] = returns.rolling(4).sum()
    features['ret_8w'] = returns.rolling(8).sum()
    features['ret_12w'] = returns.rolling(12).sum()
    features['ret_26w'] = returns.rolling(26).sum()

    # Momentum acceleration and persistence
    features['mom_accel_short'] = features['ret_2w'] - features['ret_4w']
    features['mom_accel_long'] = features['ret_8w'] - features['ret_12w']
    features['mom_persistence'] = np.sign(features['ret_4w']) * np.sign(features['ret_8w'])

    # MEAN REVERSION
    sma_5 = prices.rolling(5).mean()
    sma_10 = prices.rolling(10).mean()
    sma_20 = prices.rolling(20).mean()
    sma_60 = prices.rolling(60).mean()

    features['price_sma5_ratio'] = (prices - sma_5) / (sma_5 + 1e-6)
    features['price_sma20_ratio'] = (prices - sma_20) / (sma_20 + 1e-6)
    features['price_sma60_ratio'] = (prices - sma_60) / (sma_60 + 1e-6)

    # SMA convergence/divergence
    features['sma_convergence'] = (sma_5 - sma_20) / (sma_20 + 1e-6)
    features['sma_trend'] = (sma_20 - sma_60) / (sma_60 + 1e-6)

    # VOLUME ANALYSIS
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_20 = volume.rolling(20).mean()
    vol_ma_60 = volume.rolling(60).mean()

    features['volume_ratio_5d'] = volume / (vol_ma_5 + 1e-6)
    features['volume_ratio_20d'] = volume / (vol_ma_20 + 1e-6)
    features['volume_trend'] = (vol_ma_5 - vol_ma_20) / (vol_ma_20 + 1e-6)
    features['volume_momentum'] = (vol_ma_20 - vol_ma_60) / (vol_ma_60 + 1e-6)

    # Price-Volume relationships
    features['price_volume_corr'] = returns.rolling(20).corr(volume.pct_change())

    # TECHNICAL INDICATORS
    # Enhanced RSI
    price_changes = prices.diff()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)

    avg_gains_14 = gains.rolling(14).mean()
    avg_losses_14 = losses.rolling(14).mean()
    avg_gains_30 = gains.rolling(30).mean()
    avg_losses_30 = losses.rolling(30).mean()

    rs_14 = avg_gains_14 / (avg_losses_14 + 1e-6)
    rs_30 = avg_gains_30 / (avg_losses_30 + 1e-6)

    features['rsi_14'] = (100 - (100 / (1 + rs_14)) - 50) / 50
    features['rsi_30'] = (100 - (100 / (1 + rs_30)) - 50) / 50
    features['rsi_divergence'] = features['rsi_14'] - features['rsi_30']

    # Bollinger Bands
    bb_ma = prices.rolling(20).mean()
    bb_std = prices.rolling(20).std()
    features['bb_position'] = (prices - bb_ma) / (2 * bb_std + 1e-6)
    features['bb_width'] = bb_std / (bb_ma + 1e-6)

    # MARKET MICROSTRUCTURE
    # High-Low spread
    high_low_spread = (prices.rolling(5).max() - prices.rolling(5).min()) / (prices + 1e-6)
    features['hl_spread'] = high_low_spread

    # Price efficiency (random walk deviation)
    features['price_efficiency'] = np.abs(returns) / (features['vol_20d'] + 1e-6)

    # REGIME INDICATORS
    # Volatility clustering
    features['vol_clustering'] = features['vol_5d'].rolling(10).std()

    # Return asymmetry
    up_returns = returns.where(returns > 0, 0)
    down_returns = returns.where(returns < 0, 0)
    features['return_asymmetry'] = (up_returns.rolling(20).std() - (-down_returns).rolling(20).std()) / (
                features['vol_20d'] + 1e-6)

    # Tail risk
    features['tail_risk'] = returns.rolling(60).quantile(0.05) / (features['vol_60d'] + 1e-6)

    # INTERACTION TERMS
    features['vol_mom_interaction'] = features['vol_ratio_short'] * features['ret_4w']
    features['price_vol_interaction'] = features['price_sma20_ratio'] * features['volume_ratio_20d']

    # Clean up :-)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

    # remove consts
    for col in features.columns:
        if features[col].std() < 1e-6:
            features = features.drop(columns=[col])

    print(f"Generated {len(features.columns)} features")
    return features


def select_features(X, y_direction, n_features=15):
    if isinstance(y_direction, pd.Series):
        common_index = X.index.intersection(y_direction.index)
        X_aligned = X.loc[common_index]
        y_aligned = y_direction.loc[common_index]
    else:
        X_aligned = X.copy()
        y_aligned = pd.Series(y_direction, index=X.index)

    valid_mask = ~(X_aligned.isnull().any(axis=1) | y_aligned.isnull())
    X_clean = X_aligned[valid_mask]
    y_clean = y_aligned[valid_mask]

    print(f"Feature selection: {len(X_aligned)} → {len(X_clean)} valid samples")

    if len(X_clean) < 50:
        raise ValueError(f"Too few valid samples: {len(X_clean)}")

    # Remove highly correlated features
    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]

    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features")
        X_clean = X_clean.drop(columns=to_drop)

    feature_scores = {}

    # Correlation with target
    for col in X_clean.columns:
        try:
            corr = np.corrcoef(X_clean[col], y_clean)[0, 1]
            if not np.isnan(corr):
                feature_scores[col] = abs(corr)
        except:
            feature_scores[col] = 0

    # Mutual information
    try:
        mi_scores = mutual_info_classif(X_clean, y_clean, random_state=42)
        for i, col in enumerate(X_clean.columns):
            feature_scores[col] = feature_scores.get(col, 0) + mi_scores[i]
    except:
        print("Mutual information calculation failed")

    # Statistical significance
    try:
        from scipy.stats import pointbiserialr
        for col in X_clean.columns:
            stat, p_value = pointbiserialr(y_clean, X_clean[col])
            if p_value < 0.05:
                feature_scores[col] = feature_scores.get(col, 0) + abs(stat) * 2
    except:
        print("Statistical significance test failed")

    # Regime-specific predictive power
    # high vs low volatility
    vol_proxy = X_clean['vol_20d'] if 'vol_20d' in X_clean.columns else X_clean.iloc[:, 0]
    high_vol_mask = vol_proxy > vol_proxy.median()

    for col in X_clean.columns:
        try:
            # Corr in high vol regime
            if np.sum(high_vol_mask) > 10:
                corr_high = np.corrcoef(X_clean.loc[high_vol_mask, col], y_clean[high_vol_mask])[0, 1]
                if not np.isnan(corr_high):
                    feature_scores[col] = feature_scores.get(col, 0) + abs(corr_high) * 0.5

            # Corr in low vol regime
            if np.sum(~high_vol_mask) > 10:
                corr_low = np.corrcoef(X_clean.loc[~high_vol_mask, col], y_clean[~high_vol_mask])[0, 1]
                if not np.isnan(corr_low):
                    feature_scores[col] = feature_scores.get(col, 0) + abs(corr_low) * 0.5
        except:
            pass

    # Select top features
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_cols = [feat for feat, _ in sorted_features[:n_features]]

    print(f"Selected {len(selected_cols)} features with scores:")
    for i, (feat, score) in enumerate(sorted_features[:n_features]):
        print(f"  {i + 1:2d}. {feat:<25} {score:.4f}")

    return X_clean[selected_cols], selected_cols


def get_next_market_week(current_date):
    days_ahead = 4 - current_date.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    return current_date + timedelta(days=days_ahead)


def discriminative_spectral_directional_predictor(start_date='2010-01-01', prediction_date=None):
    print("=== ENHANCED DISCRIMINATIVE SPECTRAL DIRECTIONAL PREDICTOR ===")

    if prediction_date is None:
        prediction_date = get_next_market_week(datetime.now())

    # Fetch data
    print("Fetching market data...")
    end_date = (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d')
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

    # Handle MultiIndex columns
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)

    # Convert to weekly
    weekly_data = sp500.resample('W-FRI').agg({
        sp500.columns[0]: 'first',
        sp500.columns[1]: 'max',
        sp500.columns[2]: 'min',
        sp500.columns[3]: 'last',
        sp500.columns[4]: 'sum'
    }).dropna()

    weekly_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    weekly_data['returns'] = weekly_data['Close'].pct_change()
    weekly_data['direction'] = (weekly_data['returns'] > 0).astype(int)

    print(f"Weekly data: {len(weekly_data)} weeks")
    print(f"Direction distribution: {weekly_data['direction'].value_counts().to_dict()}")

    print("Calculating features...")
    features = calculate_features(weekly_data['returns'], weekly_data['Close'], weekly_data['Volume'])

    # Prepare HMM features
    hmm_features = pd.DataFrame({
        'returns': weekly_data['returns'],
        'volatility': features['vol_20d'],
        'momentum': features['ret_4w']
    }).dropna()

    scaler_hmm = RobustScaler()
    hmm_scaled = scaler_hmm.fit_transform(hmm_features)
    hmm_scaled = pd.DataFrame(hmm_scaled, columns=hmm_features.columns, index=hmm_features.index)

    # fit HMM
    print("Fitting HMM for regime detection...")
    hmm = HiddenMarkovModel(n_states=3, n_features=3)
    hmm.fit(hmm_scaled.values)

    # Calculate regime probabilities
    regime_probs = []
    for i in range(len(hmm_scaled)):
        if i < 10:
            regime_probs.append(hmm.pi)
        else:
            obs_window = hmm_scaled.values[max(0, i - 20):i + 1]
            regime_prob = hmm.predict_proba(obs_window)
            regime_probs.append(regime_prob)

    regime_probs = np.array(regime_probs)

    # Feature selection
    print("Selecting enhanced features...")
    direction_target = weekly_data['direction'].copy()

    features_lagged = features.shift(1)

    common_index = features_lagged.index.intersection(direction_target.index)
    X_full = features_lagged.loc[common_index]
    y_direction = direction_target.loc[common_index]

    valid_mask = ~(X_full.isnull().any(axis=1) | y_direction.isnull())
    X_clean = X_full[valid_mask]
    y_clean = y_direction[valid_mask]

    X_selected, selected_features = select_features(X_clean, y_clean, n_features=15)

    final_common_index = X_selected.index.intersection(hmm_features.index)
    X_train = X_selected.loc[final_common_index]
    y_train = y_clean.loc[final_common_index]
    regime_train = regime_probs[:len(final_common_index)]

    print(f"Training on {len(X_train)} samples with {len(selected_features)} features")

    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    print("Fitting Discriminative Spectral Model...")
    spectral_predictor = DiscriminativeSpectralPredictor(n_components=8, n_regimes=3)
    spectral_predictor.fit(X_train_scaled, y_train.values, regime_train)

    print("Making prediction...")

    recent_features = X_selected.iloc[-1:].copy()
    recent_features_scaled = scaler_X.transform(recent_features)

    recent_hmm = hmm_scaled.iloc[-20:].values
    current_regime_probs = hmm.predict_proba(recent_hmm)

    # Predict
    prob_up, confidence = spectral_predictor.predict_proba(
        recent_features_scaled,
        current_regime_probs.reshape(1, -1)
    )

    prob_up = prob_up[0]
    confidence = confidence[0]

    uncertainty = 1.0 - confidence

    # Results
    regime_names = ['Low Volatility', 'Normal Market', 'High Volatility']
    current_regime = regime_names[np.argmax(current_regime_probs)]

    direction_pred = "UP" if prob_up > 0.5 else "DOWN"
    prob_direction = max(prob_up, 1 - prob_up)

    print("\n" + "=" * 80)
    print(f"DISCRIMINATIVE SPECTRAL PREDICTION FOR WEEK OF {prediction_date.strftime('%B %d, %Y')}")
    print("=" * 80)

    print(f"\nMARKET DIRECTION: {direction_pred}")
    print(f"PROBABILITY: {prob_direction:.1%}")
    print(f"CONFIDENCE: {confidence:.1%}")
    print(f"UNCERTAINTY: {uncertainty:.3f}")

    print(f"\nCURRENT REGIME: {current_regime}")
    print("Regime Probabilities:")
    for name, prob in zip(regime_names, current_regime_probs):
        print(f"  {name}: {prob:.1%}")

    print(f"\nDISCRIMINATIVE SPECTRAL MODEL:")
    print(f"  Discriminative spectral components: {spectral_predictor.n_components}")
    print(f"  Features used: {len(selected_features)}")
    print(f"  Enhanced feature engineering ({len(features.columns)} -> {len(selected_features)})")

    return {
        'prediction_date': prediction_date,
        'direction': direction_pred,
        'probability': prob_direction,
        'confidence': confidence,
        'uncertainty': uncertainty,
        'regime': current_regime,
        'regime_probs': dict(zip(regime_names, current_regime_probs)),
        'hmm_model': hmm,
        'spectral_predictor': spectral_predictor,
        'scalers': {'hmm': scaler_hmm, 'features': scaler_X},
        'selected_features': selected_features
    }


if __name__ == "__main__":
    results = discriminative_spectral_directional_predictor()
