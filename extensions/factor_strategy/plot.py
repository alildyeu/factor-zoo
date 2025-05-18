import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

erc = pd.read_csv('ERC.csv')
max_sh = pd.read_csv('Max_Sharpe.csv')
eq = pd.read_csv('equal.csv')


# Convertir les dates en format datetime pour un meilleur affichage
erc['date'] = pd.to_datetime(erc['date'])
max_sh['date'] = pd.to_datetime(max_sh['date'])
eq['date'] = pd.to_datetime(eq['date'])

# Créer la figure et les axes
plt.figure(figsize=(12, 8))

# Tracer les 3 courbes avec portfolio_value divisé par 100
plt.plot(erc['date'], erc['portfolio_value']/100, label='ERC', linewidth=2)
plt.plot(max_sh['date'], max_sh['portfolio_value']/100, label='Max Sharpe', linewidth=2)
plt.plot(eq['date'], eq['portfolio_value']/100, label='Equally Weighted', linewidth=2)

# Ajouter les éléments de mise en forme
plt.title('Comparaison des stratégies de pondération', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Valeur du portefeuille', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Formater l'axe des abscisses pour une meilleure lisibilité
plt.xticks(rotation=45)
plt.tight_layout()

# Afficher le graphique
plt.show()


# Définir une fonction pour calculer les métriques
def calculate_metrics(portfolio_values):
    returns = portfolio_values.pct_change().dropna()
    annual_factor = 12
    nb_years = len(returns) / annual_factor

    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1 / nb_years) - 1
    vol = returns.std() * np.sqrt(annual_factor)
    sharpe = returns.mean() / returns.std() * np.sqrt(annual_factor) if returns.std() > 0 else 0
    max_dd = (1 - (pd.Series(portfolio_values) / pd.Series(portfolio_values).cummax())).max()
    final_value = portfolio_values.iloc[-1]

    return pd.Series({
        'Rendement moyen annualisé': cagr,
        'Volatilité annualisée': vol,
        'Ratio de Sharpe': sharpe,
        'Drawdown maximal': max_dd,
        'Valeur finale': final_value
    })

erc_metrics = calculate_metrics(erc['portfolio_value'])
max_sh_metrics = calculate_metrics(max_sh['portfolio_value'])
eq_metrics = calculate_metrics(eq['portfolio_value'])

metrics = pd.DataFrame([erc_metrics, max_sh_metrics, eq_metrics]).T.rename(columns={0:'ERC', 1:'Max Sharpe', 2:'Equally Weighted'})
metrics.to_csv('metrics_strat.csv', index=False)