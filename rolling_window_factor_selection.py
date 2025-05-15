import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from dataclasses import dataclass
from data_loader import DataLoader
from clusters import cluster_mapping
from factor_selection import IterativeFactorSelection


@dataclass
class RollingWindowConfig:
    window_size: int = 180  # 15 ans en mois
    step: int = 12  # Avancer d'un an à chaque itération
    t_thresholds: Tuple[float, float] = (2.00, 3.00)


class RollingWindowFactorSelection:
    def __init__(self, weighting: str = 'VW_cap', config: RollingWindowConfig = None):
        self.weighting = weighting
        self.config = config if config else RollingWindowConfig()
        self.data_loader = DataLoader(weighting=weighting)
        self.factors_df, self.market_return = self.data_loader.load_factor_data('US')
        self.cluster_mapping = cluster_mapping

    def prepare_rolling_windows(self) -> List[pd.DataFrame]:
        """Prépare les fenêtres glissantes pour l'analyse"""
        dates = self.factors_df.index
        windows = []

        start_idx = 0
        end_idx = start_idx + self.config.window_size

        while end_idx <= len(dates):
            window_dates = dates[start_idx:end_idx]
            windows.append({
                'start_date': window_dates[0],
                'end_date': window_dates[-1],
                'factors_df': self.factors_df.loc[window_dates],
                'market_return': self.market_return.loc[window_dates]
            })

            start_idx += self.config.step
            end_idx = start_idx + self.config.window_size

        return windows

    def run_rolling_analysis(self) -> pd.DataFrame:
        """Exécute l'analyse sur toutes les fenêtres glissantes"""
        windows = self.prepare_rolling_windows()
        results = []

        for window in tqdm(windows, desc="Processing rolling windows"):
            for threshold in self.config.t_thresholds:
                selector = IterativeFactorSelection(
                    factors_df=window['factors_df'],
                    market_return=window['market_return']
                )

                # Utilise la méthode de sélection basée sur t-stat
                selector.select_factors_t_std(threshold=threshold)

                if not selector.results.empty:
                    # last_row = selector.results.iloc[-1]
                    n_factors = len(selector.results)

                    # Compter les facteurs par cluster
                    clusters = self.count_factors_by_cluster(selector.results['factor'].tolist())

                    results.append({
                        'end_date': window['end_date'],
                        'threshold': threshold,
                        'n_factors': n_factors,
                        **clusters
                    })

        return pd.DataFrame(results)

    def count_factors_by_cluster(self, factors: List[str]) -> Dict[str, int]:
        """Compte le nombre de facteurs sélectionnés par cluster"""
        cluster_counts = {}

        for factor in factors:
            cluster = self.cluster_mapping.get(factor, 'Other')
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        return cluster_counts

    def plot_exhibit6(self, results_df: pd.DataFrame):
        """Crée un graphique avec des barres séparées par date"""
        # Assurer que la colonne end_date est au format datetime
        results_df['end_date'] = pd.to_datetime(results_df['end_date'])

        # Trier par date pour garantir l'ordre chronologique
        results_df = results_df.sort_values('end_date')

        # Extraire l'année pour les labels
        results_df['year'] = results_df['end_date'].dt.year

        # Préparer les données par threshold
        thresholds = sorted(results_df['threshold'].unique())
        plot_data = {threshold: results_df[results_df['threshold'] == threshold].reset_index(drop=True)
                     for threshold in thresholds}

        # Ordre exact des clusters comme dans l'image
        cluster_order = [
            'Accruals', 'Investment', 'Low Risk', 'Momentum', 'Profitability',
            'Seasonality', 'Size', 'Debt Issuance', 'Low Leverage', 'Market',
            'Profit Growth', 'Quality', 'Short-Term Reversal', 'Value'
        ]

        # Couleurs correspondantes
        cluster_colors = {
            'Accruals': '#1f77b4',
            'Investment': '#ff7f0e',
            'Low Risk': '#2ca02c',
            'Momentum': '#7f7f7f',
            'Profitability': '#aec7e8',
            'Seasonality': '#17becf',
            'Size': '#bcbd22',
            'Debt Issuance': '#ffbb78',
            'Low Leverage': '#98df8a',
            'Market': '#e377c2',
            'Profit Growth': '#d62728',
            'Quality': '#9467bd',
            'Short-Term Reversal': '#8c564b',
            'Value': '#e377c2'
        }

        # Créer la figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Panel supérieur (t > 2)
        self._plot_single_panel(ax1, plot_data[2.0], cluster_order, cluster_colors, "t > 2")

        # Panel inférieur (t > 3)
        self._plot_single_panel(ax2, plot_data[3.0], cluster_order, cluster_colors, "t > 3")

        # Configuration finale
        plt.xlabel('Year')
        plt.tight_layout()
        plt.show()

    def _plot_single_panel(self, ax, data, cluster_order, cluster_colors, title):
        """Helper pour tracer un seul panel avec barres séparées"""
        if data.empty:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return

        # Position des barres
        x_pos = np.arange(len(data))
        bar_width = 0.8  # Largeur des barres

        # Les labels de l'axe x seront les années
        years = data['year'].tolist()

        # Tracer la ligne du total
        '''total = data['n_factors']
        ax.plot(x_pos, total, 'k-', linewidth=2, label='Total Factors', marker='o')'''

        # Barres empilées
        bottom = np.zeros(len(data))
        for cluster in cluster_order:
            if cluster in data.columns:
                counts = data[cluster].fillna(0)
                ax.bar(x_pos, counts, bottom=bottom,
                       color=cluster_colors.get(cluster, '#999999'),  # couleur par défaut si manquante
                       label=cluster, width=bar_width, edgecolor='white', linewidth=0.5)
                bottom += counts

        # Configuration des axes
        ax.set_title(title, pad=20)
        ax.set_ylabel('Number of Factors')
        ax.grid(True, axis='y', alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(years, rotation=90)

        # Définir les graduations selon le threshold
        if "t > 2" in title:
            # Pour le graphique t > 2, graduations par pas de 5
            ymax = np.ceil(data['n_factors'].max() / 5) * 5
            ax.set_yticks(np.arange(0, ymax + 1, 5))
        else:
            # Pour le graphique t > 3, graduations par nombres entiers
            y_ticks = ax.get_yticks()
            y_ticks_int = [int(y) for y in y_ticks]
            ax.set_yticks(y_ticks_int)

        # Définir les limites pour que le graphique soit bien affiché
        ax.set_xlim(-0.5, len(x_pos) - 0.5)

        # Légende seulement pour le panel inférieur
        if ax.get_subplotspec().is_last_row():
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2)

    def run_full_analysis(self):
        """Exécute l'analyse complète"""
        results_df = self.run_rolling_analysis()
        self.plot_exhibit6(results_df)
        return results_df


if __name__ == "__main__":
    analyzer = RollingWindowFactorSelection(weighting='VW_cap')
    df = pd.read_csv('rolling_window_factor_selection.csv')
    analyzer.plot_exhibit6(df)
    #results = analyzer.run_full_analysis()