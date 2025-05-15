import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from factor_selection import IterativeFactorSelection
from rolling_window_factor_selection import RollingWindowFactorSelection
from clusters import cluster_mapping

class FactorStrategyBacktester:
    def __init__(self, factor_selector):
        """
        Initialise le backtester

        Args:
            factor_selector: Instance de RollingWindowFactorSelection déjà configurée
        """
        self.selector = factor_selector
        self.portfolio_values = []
        self.weights_history = []
        self.performance_dates = []
        self.factor_weights = {}  # Pour stocker les poids actifs des facteurs

    def backtest_strategy(self, initial_capital=100, rebalance=False, market_weight=0.3, weight_type='t_stat'):
        """
        Exécute le backtest complet de la stratégie

        Args:
            initial_capital: Capital initial à investir
            rebalance: Si True, rebalance à chaque fenêtre. Si False, ajuste seulement lors des premiers achats.
            market_weight: Poids fixe pour le facteur de marché (entre 0 et 1)
            weight_type: Type de pondération pour les autres facteurs ('equal' ou 't_stat')

        Returns:
            DataFrame avec les résultats du backtest
        """
        windows = self.selector.prepare_rolling_windows()
        self.portfolio_values = [initial_capital]
        self.weights_history = []
        self.performance_dates = [windows[0]['end_date']]
        self.factor_weights = {}  # Réinitialiser les poids actifs

        for i in tqdm(range(len(windows) - 1), desc="Backtesting Strategy"):
            # 1. Sélection des facteurs avec t-stats à t_i
            factors_with_t_stats = self._select_factors(windows[i])

            if not factors_with_t_stats:
                continue

            # 2. Calcul des poids basés sur les t-stats (ou conservation des poids actuels si pas de rebalancement)
            if rebalance or not self.factor_weights:
                self.factor_weights = self._calculate_weights(
                    factors_with_t_stats,
                    market_weight=market_weight,
                    weight_type=weight_type
                )
            else:
                # Ajouter uniquement les nouveaux facteurs s'il n'y a pas de rebalancement
                new_factors = set(factors_with_t_stats.keys()) - set(self.factor_weights.keys())
                if new_factors:
                    new_weights = {f: factors_with_t_stats[f] for f in new_factors}
                    new_normalized_weights = self._normalize_weights(new_weights)
                    # Déterminer la proportion à allouer aux nouveaux facteurs (par exemple 10%)
                    new_allocation = 0.1
                    # Réduire les poids existants pour faire place aux nouveaux
                    self.factor_weights = {k: v * (1 - new_allocation) for k, v in self.factor_weights.items()}
                    # Ajouter les nouveaux facteurs avec leurs poids
                    for f, w in new_normalized_weights.items():
                        self.factor_weights[f] = w * new_allocation

            # Enregistrer l'historique des poids
            self.weights_history.append((windows[i]['end_date'], self.factor_weights.copy()))

            # 3. Période de détention [t_i, t_i+1]
            hold_returns = self._get_hold_returns(
                list(self.factor_weights.keys()),
                windows[i]['end_date'],
                windows[i + 1]['end_date']
            )

            if hold_returns.empty:
                continue

            # 4. Correction du drift
            adjusted_returns = self._adjust_for_drift(self.factor_weights, hold_returns)

            # 5. Mise à jour valeur portefeuille
            for date, ret in adjusted_returns.items():
                self.portfolio_values.append(self.portfolio_values[-1] * (1 + ret))
                self.performance_dates.append(date)

        return self._generate_results()

    def _normalize_weights(self, weights):
        """Normalise les poids pour qu'ils somment à 1"""
        total = sum(weights.values())
        if total == 0:
            return {k: 1/len(weights) for k in weights}
        return {k: v/total for k, v in weights.items()}

    def _adjust_for_drift(self, initial_weights, hold_returns):
        """Corrige la dérive des poids"""
        # Version simplifiée : calcul du rendement pondéré pour chaque date
        weighted_returns = {}

        # Pour chaque date, appliquer les poids aux rendements
        for date in hold_returns.index:
            total_return = sum(hold_returns.loc[date, factor] * weight
                               for factor, weight in initial_weights.items()
                               if factor in hold_returns.columns)
            weighted_returns[date] = total_return

        return weighted_returns

    def _select_factors(self, window):
        """Sélectionne les facteurs pour la fenêtre donnée et renvoie les facteurs avec leurs t-stats"""
        # Calculer la t-stat du marché
        market_return = window['market_return'].dropna()
        if len(market_return) >= 50:
            market_mean = market_return.mean()
            market_std = market_return.std()
            market_t_stat = market_mean / (market_std / np.sqrt(len(market_return)))
        else:
            market_t_stat = 0

        # Sélectionner les autres facteurs
        selector = IterativeFactorSelection(
            factors_df=window['factors_df'],
            market_return=window['market_return']
        )
        selector.select_factors_t_std(threshold=3)

        # Combiner marché et autres facteurs sélectionnés
        factors_with_t_stats = {'market': abs(market_t_stat)}

        if not selector.results.empty:
            for _, row in selector.results.iterrows():
                factors_with_t_stats[row['factor']] = abs(row['t_stat'])

        return factors_with_t_stats

    def _calculate_weights(self, factors_with_t_stats, market_weight=0.3, weight_type='t_stat'):
        """
        Calcule la pondération des facteurs

        Args:
            factors_with_t_stats: Dictionnaire {facteur: t_stat}
            market_weight: Poids fixe pour le marché (uniquement utilisé si weight_type='t_stat')
            weight_type: Type de pondération ('equal' ou 't_stat')

        Returns:
            Dictionnaire des poids normalisés
        """
        if not factors_with_t_stats:
            return {}

        # En mode équipondération, tous les facteurs (y compris le marché) ont le même poids
        if weight_type == 'equal':
            total_factors = len(factors_with_t_stats)
            return {factor: 1.0 / total_factors for factor in factors_with_t_stats}

        # Mode t_stat avec poids fixe pour le marché
        else:
            # S'il n'y a que le marché, lui donner tout le poids
            if len(factors_with_t_stats) == 1 and 'market' in factors_with_t_stats:
                return {'market': 1.0}

            # Séparer le marché des autres facteurs
            other_factors = {f: t for f, t in factors_with_t_stats.items() if f != 'market'}

            # Si le marché n'est pas dans les facteurs sélectionnés, ajuster market_weight à 0
            if 'market' not in factors_with_t_stats:
                market_weight = 0

            # Pondération par t-stat pour les autres facteurs
            total_t_stat = sum(other_factors.values())

            if total_t_stat == 0:  # Protection contre division par zéro
                other_weights = {factor: (1 - market_weight) / len(other_factors)
                                 for factor in other_factors}
            else:
                other_weights = {factor: t_stat / total_t_stat * (1 - market_weight)
                                 for factor, t_stat in other_factors.items()}

            # Ajouter le marché avec son poids fixe
            weights = other_weights.copy()
            if 'market' in factors_with_t_stats:
                weights['market'] = market_weight

            return weights

    def _get_hold_returns(self, factors, start_date, end_date):
        """Récupère les rendements des facteurs pour la période de détention"""
        # Vérifie que tous les facteurs nécessaires sont présents
        valid_factors = [f for f in factors if f in self.selector.factors_df.columns]
        if not valid_factors:
            return pd.DataFrame()

        mask = (self.selector.factors_df.index > start_date) & \
               (self.selector.factors_df.index <= end_date)
        return self.selector.factors_df.loc[mask, valid_factors]

    def _generate_results(self):
        """Génère un DataFrame avec les résultats du backtest"""
        results = pd.DataFrame({
            'date': self.performance_dates,
            'portfolio_value': self.portfolio_values
        })

        # Ajouter les poids des facteurs si nécessaire
        if self.weights_history:
            weights_df = pd.DataFrame(self.weights_history, columns=['date', 'weights'])
            results = results.merge(weights_df, on='date', how='left')

        return results

    def plot_performance(self):
        """Visualise la performance de la stratégie"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.performance_dates, self.portfolio_values, 'b-', lw=2)
        plt.title("Drift-Adjusted Strategy Performance")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True, ls=':')
        plt.show()

    def get_performance_metrics(self):
        """
        Calcule les métriques de performance clés

        Returns:
            Dict avec les métriques de performance
        """
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        trading_days = len(returns)
        annual_factor = 12  # Nombre de jours de trading par an

        metrics = {
            'CAGR': (self.portfolio_values[-1] / self.portfolio_values[0]) ** (annual_factor / trading_days) - 1,
            'Volatilité annualisée': returns.std() * np.sqrt(annual_factor),
            'Ratio de Sharpe': returns.mean() / returns.std() * np.sqrt(annual_factor) if returns.std() > 0 else 0,
            'Drawdown maximal': (
                        1 - (pd.Series(self.portfolio_values) / pd.Series(self.portfolio_values).cummax())).max(),
            'Valeur finale': self.portfolio_values[-1]
        }

        return metrics


if __name__ == "__main__":
    # Initialiser le sélecteur de facteurs
    selector = RollingWindowFactorSelection(weighting='VW_cap')

    # Exécuter le backtest avec rebalancement et paramètres de pondération
    backtester = FactorStrategyBacktester(selector)
    results = backtester.backtest_strategy(
        initial_capital=100,
        market_weight=0,
        weight_type='equal'
    )

    # Visualiser les résultats
    backtester.plot_performance()

    # Afficher les métriques
    metrics = backtester.get_performance_metrics()
    print("\nMétriques de performance (marché 30%, autres par t-stat):")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.4f}")

    # Test avec marché à 30% et autres facteurs équipondérés
    backtester2 = FactorStrategyBacktester(selector)
    results2 = backtester2.backtest_strategy(
        initial_capital=100,
        market_weight=0.3,
        weight_type='equal'
    )

    # Visualiser les résultats
    backtester2.plot_performance()

    # Afficher les métriques
    metrics2 = backtester2.get_performance_metrics()
    print("\nMétriques de performance (marché 30%, autres équipondérés):")
    for k, v in metrics2.items():
        print(f"{k:20}: {v:.4f}")