# corrections_exhibits.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from factor_selection import IterativeFactorSelection
from clusters import create_factor_clusters

class FactorZooExhibitsImproved:
    """Version améliorée des exhibits pour correspondre plus fidèlement à l'article Factor Zoo"""
    
    def __init__(self, start_date="1971-11-01", end_date="2021-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        # Importez le nouveau mapping des clusters (si stocké dans une variable)
        try:
            from clusters import cluster_mapping
            self.cluster_mapping = cluster_mapping
        except ImportError:
            # Fallback au create_factor_clusters si cluster_mapping n'est pas disponible
            self.get_cluster = create_factor_clusters()
            self.cluster_mapping = None
    
    def get_cluster_exact(self, factor):
        """Utilise le mapping des clusters de l'article si disponible, sinon fallback"""
        if self.cluster_mapping is not None:
            return self.cluster_mapping.get(factor, "Other")
        else:
            return self.get_cluster(factor)
    
    def exhibit_1_factor_alphas_article(self):
        """Version qui reproduit exactement l'Exhibit 1 de l'article original"""
        print("\nGénération Exhibit 1 (reproduction exacte): Factor Alphas")
        
        data_loader = DataLoader('VW_cap', self.start_date, self.end_date)
        factors_df, market_return = data_loader.load_factor_data('US')
        
        # Afficher le nombre de facteurs pour diagnostic
        print(f"Nombre de facteurs chargés: {len(factors_df.columns)}")
        
        # Calculer les alphas CAPM pour tous les facteurs
        alphas = {}
        for factor in factors_df.columns:
            y = factors_df[factor]
            X = sm.add_constant(market_return)
            valid_idx = ~(y.isna() | market_return.isna())
            
            if valid_idx.sum() > 30:
                model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
                alphas[factor] = model.params[0] * 12 * 100  # Annualisé en %
        
        # Préparer les données pour le graphique
        alpha_df = pd.DataFrame.from_dict(alphas, orient='index', columns=['alpha'])
        alpha_df['cluster'] = alpha_df.index.map(self.get_cluster_exact)
        
        # Afficher la distribution des clusters pour diagnostic
        print("Distribution des clusters:")
        for cluster in sorted(alpha_df['cluster'].unique()):
            count = len(alpha_df[alpha_df['cluster'] == cluster])
            print(f"  {cluster}: {count} facteurs")
        
        # Définir l'ordre exact des clusters comme dans l'article
        cluster_order = [
            'Accruals', 'Debt Issuance', 'Investment', 'Low Leverage', 
            'Low Risk', 'Momentum', 'Profit Growth', 'Profitability', 
            'Quality', 'Seasonality', 'Short-Term Reversal', 'Size', 'Value'
        ]
        
        # S'assurer que tous les clusters sont présents dans les données
        cluster_order = [c for c in cluster_order if c in alpha_df['cluster'].unique()]
        
        # Créer un nouveau DataFrame pour maintenir l'ordre des clusters
        sorted_df = pd.DataFrame()
        for cluster in cluster_order:
            cluster_data = alpha_df[alpha_df['cluster'] == cluster].sort_values('alpha')
            sorted_df = pd.concat([sorted_df, cluster_data])
        
        # Utiliser le DataFrame trié pour le graphique
        alpha_df = sorted_df
        
        # Créer le graphique avec un fond gris clair comme dans l'article
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_facecolor('#f0f0f0')  # Fond gris clair
        
        # Palette de couleurs exactement comme dans l'article
        cluster_colors = {
            'Accruals': '#1f77b4',        # Bleu
            'Debt Issuance': '#ff7f0e',   # Orange
            'Investment': '#d62728',      # Rouge
            'Low Leverage': '#9467bd',    # Violet
            'Low Risk': '#bcbd22',        # Jaune-vert
            'Momentum': '#8c564b',        # Marron clair
            'Profit Growth': '#17becf',   # Turquoise
            'Profitability': '#e377c2',   # Rose
            'Quality': '#7f7f7f',         # Gris
            'Seasonality': '#2ca02c',     # Vert
            'Short-Term Reversal': '#8c564b', # Marron
            'Size': '#00bfff',            # Bleu ciel
            'Value': '#1f77b4'            # Bleu foncé
        }
        
        # Tracer les barres par cluster
        x_pos = 0
        x_positions = []
        x_labels = []
        
        for cluster in cluster_order:
            cluster_data = alpha_df[alpha_df['cluster'] == cluster]
            
            for idx, row in cluster_data.iterrows():
                color = cluster_colors.get(cluster, '#333333')
                ax.bar(x_pos, row['alpha'], width=0.8, color=color, alpha=0.9)
                x_positions.append(x_pos)
                x_labels.append(idx)
                x_pos += 1
            
            # Ajouter un espace entre les clusters
            x_pos += 1
        
        # Configurer les axes et labels
        ax.set_ylabel('Alphas p.a. [%]', fontsize=12)
        ax.set_title('Factor Alphas', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Définir l'échelle de l'axe Y comme dans l'article
        ax.set_ylim(-6, 10)
        
        # Réduire le nombre d'étiquettes sur l'axe X pour éviter l'encombrement
        step = max(1, len(x_positions) // 40)  # Réduire si nécessaire
        ax.set_xticks(x_positions[::step])
        ax.set_xticklabels(x_labels[::step], rotation=90, fontsize=6)
        
        # Ajouter une légende avec les couleurs exactes comme dans l'article
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=cluster_colors.get(cluster, '#333333'), 
                                        alpha=0.9, label=cluster) 
                          for cluster in cluster_order]
        
        # Placer la légende en bas comme dans l'article
        ax.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=7, frameon=False)
        
        # Ajuster les marges pour accommoder la légende
        plt.subplots_adjust(bottom=0.25)
        
        # Ajouter des lignes verticales entre les clusters
        prev_cluster = None
        for i, (idx, row) in enumerate(alpha_df.iterrows()):
            if row['cluster'] != prev_cluster:
                if prev_cluster is not None:
                    # Ajouter une ligne verticale après chaque cluster
                    ax.axvline(x=i-0.5, color='white', linestyle='-', linewidth=1)
                prev_cluster = row['cluster']
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajuster les marges
        return fig
    
    def exhibit_4_selected_factors_plot_improved(self):
        """Version améliorée de l'Exhibit 4: Selected Alpha Factors avec meilleur contraste"""
        print("\nGénération Exhibit 4 amélioré: Selected Factors Plot")
        
        # Charger les données
        data_loader = DataLoader('VW_cap', self.start_date, self.end_date)
        factors_df, market_return = data_loader.load_factor_data('US')
        
        # Dans ce cas, on sélectionne manuellement les facteurs (les 15 premiers selon l'article)
        # Vous pouvez ajuster cette liste selon vos propres résultats
        selected_factors = ['cop_at', 'noa_gr1a', 'saleq_gr1', 'ival_me', 'resff3_12_1', 
                           'seas_6_10an', 'debt_me', 'seas_6_10na', 'zero_trades_252d',
                           'cowc_gr1a', 'nncoa_gr1a', 'ocf_me', 'zero_trades_21d', 
                           'turnover_126d', 'rmax5_rvol_21d']
        
        # Calculer les alphas CAPM pour tous les facteurs
        alphas = {}
        for factor in factors_df.columns:
            y = factors_df[factor]
            X = sm.add_constant(market_return)
            valid_idx = ~(y.isna() | market_return.isna())
            
            if valid_idx.sum() > 30:
                model = sm.OLS(y[valid_idx], X[valid_idx]).fit()
                alphas[factor] = model.params[0] * 12 * 100  # Annualisé en %
        
        # Préparer les données pour le graphique
        alpha_df = pd.DataFrame.from_dict(alphas, orient='index', columns=['alpha'])
        alpha_df['cluster'] = alpha_df.index.map(self.get_cluster_exact)
        alpha_df['selected'] = alpha_df.index.isin(selected_factors)
        
        # Définir l'ordre des clusters comme dans l'article
        cluster_order = [
            'Accruals', 'Debt Issuance', 'Investment', 'Low Leverage', 
            'Low Risk', 'Momentum', 'Profit Growth', 'Profitability', 
            'Quality', 'Seasonality', 'Short-Term Reversal', 'Size', 'Value'
        ]
        
        # S'assurer que tous les clusters sont présents dans les données
        cluster_order = [c for c in cluster_order if c in alpha_df['cluster'].unique()]
        
        # Créer un nouveau DataFrame pour maintenir l'ordre des clusters
        sorted_df = pd.DataFrame()
        for cluster in cluster_order:
            cluster_data = alpha_df[alpha_df['cluster'] == cluster].sort_values('alpha')
            sorted_df = pd.concat([sorted_df, cluster_data])
        
        alpha_df = sorted_df
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_facecolor('#f8f8f8')  # Fond très légèrement grisé
        
        # Palette de couleurs
        cluster_colors = {
            'Accruals': '#1f77b4',        # Bleu
            'Debt Issuance': '#ff7f0e',   # Orange
            'Investment': '#d62728',      # Rouge
            'Low Leverage': '#9467bd',    # Violet
            'Low Risk': '#bcbd22',        # Jaune-vert
            'Momentum': '#8c564b',        # Marron clair
            'Profit Growth': '#17becf',   # Turquoise
            'Profitability': '#e377c2',   # Rose
            'Quality': '#7f7f7f',         # Gris
            'Seasonality': '#2ca02c',     # Vert
            'Short-Term Reversal': '#8c564b', # Marron
            'Size': '#00bfff',            # Bleu ciel
            'Value': '#1f77b4'            # Bleu foncé
        }
        
        # Tracer les barres par cluster
        x_pos = 0
        x_positions = []
        x_labels = []
        selected_positions = []
        
        for cluster in cluster_order:
            cluster_data = alpha_df[alpha_df['cluster'] == cluster]
            
            for idx, row in cluster_data.iterrows():
                if row['selected']:
                    # Facteurs sélectionnés avec couleur vive
                    ax.bar(x_pos, row['alpha'], width=0.8, 
                           color=cluster_colors.get(cluster, '#333333'), alpha=1.0, 
                           edgecolor='black', linewidth=1.5)
                    selected_positions.append(x_pos)
                else:
                    # Facteurs non sélectionnés en gris très pâle
                    ax.bar(x_pos, row['alpha'], width=0.8, 
                           color='whitesmoke', edgecolor='lightgrey', alpha=0.3)
                
                x_positions.append(x_pos)
                x_labels.append(idx)
                x_pos += 1
            
            # Espace entre clusters
            x_pos += 0.5
        
        # Configurer les axes et labels
        ax.set_ylabel('Alpha [%]', fontsize=12)
        ax.set_title('Selected Alpha Factors', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Supprimer les étiquettes de l'axe x car il y a trop de facteurs
        ax.set_xticks([])
        
        # Ajouter une légende par cluster
        legend_elements = [plt.Rectangle((0,0), 1, 1, color=color, alpha=0.8, label=cluster) 
                          for cluster, color in cluster_colors.items() if cluster in alpha_df['cluster'].values]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 ncol=7, frameon=False)
        
        plt.tight_layout()
        return fig
        
    def save_improved_exhibits(self, output_dir='exhibits_improved'):
        """Sauvegarde les exhibits améliorés"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Exhibit 1: Factor Alphas (version améliorée)
        print("\n=== Sauvegarde Exhibit 1 amélioré ===")
        fig1 = self.exhibit_1_factor_alphas_article()
        fig1.savefig(f'{output_dir}/exhibit_1_factor_alphas_improved.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Exhibit 4: Selected Factors Plot (version améliorée)
        print("\n=== Sauvegarde Exhibit 4 amélioré ===")
        fig4 = self.exhibit_4_selected_factors_plot_improved()
        fig4.savefig(f'{output_dir}/exhibit_4_selected_factors_improved.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print(f"\n=== Les exhibits améliorés ont été sauvegardés dans '{output_dir}' ===")


# Utilisation
if __name__ == "__main__":
    print("=== Génération des exhibits améliorés du Factor Zoo ===")
    exhibits = FactorZooExhibitsImproved()
    exhibits.save_improved_exhibits()