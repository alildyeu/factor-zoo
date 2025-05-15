import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from factor_selection import IterativeFactorSelection


class FactorStrategyBacktester:
    def __init__(self, factor_selector):
        """
        Initialise le backtester

        Args:
            factor_selector: Instance de RollingWindowFactorSelection déjà configurée
        """
        self.selector = factor_selector
        self.portfolio_values = []
        self.factor_exposures = []
        self.hold_period_returns = []

    def backtest_strategy(self, initial_capital=1000):
        """
        Exécute le backtest complet sans frais de transaction

        Args:
            initial_capital: Capital initial à investir (par défaut 1M$)

        Returns:
            DataFrame avec la valeur du portefeuille et les expositions aux facteurs
        """
        windows = self.selector.prepare_rolling_windows()
        portfolio_value = initial_capital
        self.portfolio_values = [portfolio_value]
        self.factor_exposures = []
        self.hold_period_returns = []

        for i in tqdm(range(len(windows) - 1), desc="Backtesting Strategy"):
            # 1. Sélection des facteurs
            current_window = windows[i]
            selector = IterativeFactorSelection(
                factors_df=current_window['factors_df'],
                market_return=current_window['market_return']
            )
            selector.select_factors_t_std(threshold=3)

            if selector.results.empty:
                continue

            selected_factors = selector.results['factor'].tolist()

            # 2. Pondération égale des facteurs
            weights = {factor: 1 / len(selected_factors) for factor in selected_factors}
            self.factor_exposures.append({
                'date': windows[i]['end_date'],
                **weights
            })

            # 3. Période de détention (12 mois suivants)
            hold_start = windows[i]['end_date'] + pd.DateOffset(months=1)
            hold_end = windows[i + 1]['end_date']

            # Calcul des returns mensuels pendant la période de détention
            hold_returns = self._get_hold_period_returns(selected_factors, hold_start, hold_end)
            self.hold_period_returns.extend(hold_returns)

            # 4. Mise à jour de la valeur du portefeuille
            for monthly_return in hold_returns:
                portfolio_value *= (1 + monthly_return)
                self.portfolio_values.append(portfolio_value)

        # Création du DataFrame de résultats
        results = pd.DataFrame({
            'date': pd.date_range(
                start=windows[0]['end_date'] + pd.DateOffset(months=1),
                periods=len(self.portfolio_values) - 1,
                freq='M'
            ),
            'portfolio_value': self.portfolio_values[1:]
        })

        # Ajout des expositions aux facteurs
        exposures_df = pd.DataFrame(self.factor_exposures)
        results = results.merge(exposures_df, on='date', how='left')

        return results

    def _get_hold_period_returns(self, factors, start_date, end_date):
        """
        Calcule les returns mensuels des facteurs pendant la période de détention

        Args:
            factors: Liste des facteurs sélectionnés
            start_date: Date de début de détention
            end_date: Date de fin de détention

        Returns:
            Liste des returns mensuels du portefeuille
        """
        mask = (self.selector.factors_df.index >= start_date) & (self.selector.factors_df.index <= end_date)

        period_returns = self.selector.factors_df.loc[mask, factors]

        # Retourne la moyenne mensuelle (pondérée égale)
        return period_returns.mean(axis=1).tolist()


    def plot_performance(self, benchmark=None):
        """
        Visualise la performance du portefeuille

        Args:
            benchmark: Series pandas avec les valeurs d'un benchmark (optionnel)
        """
        plt.figure(figsize=(12, 6))

        # Plot strategy
        dates = pd.date_range(
            start=self.selector.factors_df.index[0] + pd.DateOffset(months=180),
            periods=len(self.portfolio_values),
            freq='M'
        )
        plt.plot(dates, self.portfolio_values, label='Factor Strategy', linewidth=2)

        # Plot benchmark if provided
        if benchmark is not None:
            plt.plot(benchmark.index, benchmark.values, label='Benchmark', linestyle='--')

        plt.title('Portfolio Performance Over Time', pad=20)
        plt.xlabel('Year')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.show()


    def get_performance_metrics(self):
        """
        Calcule les métriques de performance clés

        Returns:
            Dict avec les métriques de performance
        """
        returns = pd.Series(self.portfolio_values).pct_change().dropna()

        metrics = {
            'CAGR': (self.portfolio_values[-1] / self.portfolio_values[0]) ** (12 / len(returns)) - 1,
            'Annualized Volatility': returns.std() * np.sqrt(12),
            'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(12),
            'Max Drawdown': (returns.cumsum().expanding().max() - returns.cumsum()).max(),
            'Final Value': self.portfolio_values[-1]
        }

        return metrics


# Exemple d'utilisation
if __name__ == "__main__":
    # 1. Initialiser le sélecteur de facteurs
    from extensions.factor_strategy.rolling_window_factor_selection import RollingWindowFactorSelection

    selector = RollingWindowFactorSelection(weighting='VW_cap')

    # 2. Initialiser et exécuter le backtest
    backtester = FactorStrategyBacktester(selector)
    results = backtester.backtest_strategy(initial_capital=1000)

    # 3. Visualiser les résultats
    backtester.plot_performance()

    # 4. Afficher les métriques
    metrics = backtester.get_performance_metrics()
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.4f}")