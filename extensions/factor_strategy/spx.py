import pandas as pd
import numpy as np


def calculate_metrics():
    # Importer SPX.csv et le transformer comme les autres séries
    spx = pd.read_csv('SPX.csv', sep=';')
    spx['Date'] = pd.to_datetime(spx['Date'], dayfirst=True)  # Correction du format de date
    spx = spx[(spx['Date'] >= '1986-11-30') & (spx['Date'] <= '2023-10-31')]

    # Définir Date comme index pour faciliter les calculs
    spx.set_index('Date', inplace=True)

    # Conversion des valeurs en nombres
    portfolio_values = spx['PX_LAST'].str.replace(',', '.').astype(float)

    # Calculer les rendements du portefeuille
    returns = portfolio_values.pct_change().dropna()

    # Pour SPX, ce sont les mêmes rendements (pas besoin de calculer séparément)
    spx_returns = returns

    # Pas besoin d'aligner les dates car c'est la même série

    # Facteur d'annualisation (mensuel)
    annual_factor = 12
    nb_years = len(returns) / annual_factor

    # Calcul des métriques standards
    cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1 / nb_years) - 1
    vol = returns.std() * np.sqrt(annual_factor)
    sharpe = returns.mean() / returns.std() * np.sqrt(annual_factor) if returns.std() > 0 else 0

    # Calcul du drawdown maximal
    rolling_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    max_dd = drawdowns.min()

    final_value = portfolio_values.iloc[-1]

    # Calcul des métriques supplémentaires
    # Dans ce cas, beta = 1, alpha = 0 car c'est la même série
    beta = 1.0
    alpha = 0.0
    tracking_error = 0.0
    information_ratio = 0.0

    metrics = pd.Series({
        'Rendement moyen annualisé': cagr,
        'Volatilité annualisée': vol,
        'Ratio de Sharpe': sharpe,
        'Drawdown maximal': max_dd,
        'Valeur finale': final_value,
        'Alpha': alpha,
        'Beta': beta,
        'Tracking Error': tracking_error,
        'Ratio d\'information': information_ratio
    })

    print(metrics)
    return metrics


# Exécuter la fonction
calculate_metrics()