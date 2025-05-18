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

    def backtest_strategy(self, initial_capital=100, market_weight=0.3, weight_type='t_stat'):
        """
        Exécute le backtest complet de la stratégie

        Args:
            initial_capital: Capital initial à investir
            rebalance: Si True, rebalance à chaque fenêtre. Si False, ajuste seulement lors des premiers achats.
            market_weight: Poids fixe pour le facteur de marché (entre 0 et 1)
            weight_type: Type de pondération pour les autres facteurs ('equal', 't_stat', ou 'erc')

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
            #if rebalance or not self.factor_weights:
            self.factor_weights = self._calculate_weights(
                factors_with_t_stats,
                market_weight=market_weight,
                weight_type=weight_type)

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

    '''def _calculate_weights(self, factors_with_t_stats, market_weight=0.3, weight_type='t_stat'):
        """
        Calcule la pondération des facteurs

        Args:
            factors_with_t_stats: Dictionnaire {facteur: t_stat}
            market_weight: Poids fixe pour le marché (uniquement utilisé si weight_type='t_stat')
            weight_type: Type de pondération ('equal', 't_stat' ou 'erc')

        Returns:
            Dictionnaire des poids normalisés
        """
        if not factors_with_t_stats:
            return {}

        # Mode ERC - Equal Risk Contribution
        if weight_type == 'erc':
            # Récupérer la liste des facteurs
            factors_list = list(factors_with_t_stats.keys())

            # S'il n'y a qu'un seul facteur, lui donner tout le poids
            if len(factors_list) == 1:
                return {factors_list[0]: 1.0}

            # Séparer market des autres facteurs
            other_factors = [f for f in factors_list if f != 'market']
            has_market = 'market' in factors_list

            # Réduire la période d'historique pour améliorer la stabilité
            lookback_window = 52 * 15  # 15 ans de données hebdomadaires

            # Récupérer les rendements historiques pour l'optimisation
            latest_date = max(self.selector.factors_df.index)
            history_start = pd.Timestamp(latest_date) - pd.Timedelta(weeks=lookback_window)

            # Collecter les rendements
            returns_data = {}

            # Ajouter les rendements des facteurs standards
            valid_factors = [f for f in other_factors if f in self.selector.factors_df.columns]
            for factor in valid_factors:
                factor_data = self.selector.factors_df.loc[
                    self.selector.factors_df.index >= history_start, factor
                ].dropna()
                if not factor_data.empty:
                    returns_data[factor] = factor_data

            # Ajouter les rendements du marché si nécessaire
            if has_market:
                market_data = self.selector.market_return.loc[
                    self.selector.market_return.index >= history_start
                    ].dropna()
                if not market_data.empty:
                    returns_data['market'] = market_data

            # Si aucun facteur valide, utiliser équipondération
            if not returns_data:
                print("Aucun facteur valide pour l'ERC, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            # Créer un DataFrame des rendements alignés
            returns_df = pd.DataFrame(returns_data)
            returns_history = returns_df.dropna()

            if returns_history.empty or returns_history.shape[0] < 30:
                print("Historique insuffisant pour l'ERC, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            # Calculer la matrice de covariance
            cov_matrix = returns_history.cov()

            # Vérifier que la matrice n'est pas vide
            if cov_matrix.shape[0] == 0:
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            try:
                import scipy.optimize as sco

                # Nombre de facteurs dans la matrice de covariance
                n = len(cov_matrix)

                # Fonction objectif ERC améliorée
                def erc_objective(weights):
                    weights = np.array(weights)

                    # Volatilité du portefeuille
                    portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

                    if portfolio_vol <= 0:
                        return 1e10  # Pénalité élevée

                    # Contributions marginales au risque
                    mcr = cov_matrix.values @ weights / portfolio_vol

                    # Contributions au risque
                    rc = weights * mcr

                    # Objectif: que toutes les contributions au risque soient égales
                    target_risk = portfolio_vol / n
                    return np.sum((rc - target_risk) ** 2)

                # Contraintes: somme des poids = 1
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

                # Bornes: poids entre 1% et 50%
                bounds = tuple((0.01, 0.5) for _ in range(n))

                # Point de départ: équipondération
                x0 = np.ones(n) / n

                # Optimisation
                result = sco.minimize(
                    erc_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success:
                    optimal_weights = result.x

                    # Initialiser le dictionnaire final avec EXACTEMENT les mêmes facteurs
                    weights_dict = {factor: 0.0 for factor in factors_list}

                    # Appliquer les poids optimaux uniquement aux facteurs qui sont dans la matrice de cov
                    for i, factor in enumerate(cov_matrix.index):
                        if factor in weights_dict:
                            weights_dict[factor] = optimal_weights[i]

                    # Si certains facteurs n'ont pas été inclus dans l'optimisation,
                    # leur attribuer un poids équipondéré du reste
                    missing_factors = [f for f in factors_list if f not in cov_matrix.index]
                    if missing_factors:
                        remaining_weight = 1.0 - sum(weights_dict[f] for f in factors_list if f not in missing_factors)
                        remaining_weight = max(0, remaining_weight)  # Éviter les valeurs négatives
                        for factor in missing_factors:
                            weights_dict[factor] = remaining_weight / len(missing_factors)

                    # Normalisation finale pour assurer que la somme est exactement 1
                    total_weight = sum(weights_dict.values())
                    if total_weight > 0:
                        return {k: v / total_weight for k, v in weights_dict.items()}
                    else:
                        return {factor: 1.0 / len(factors_list) for factor in factors_list}
                else:
                    print(f"Optimisation ERC échouée: {result.message}")
                    return {factor: 1.0 / len(factors_list) for factor in factors_list}

            except Exception as e:
                print(f"Erreur dans l'optimisation ERC: {e}")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

        # En mode équipondération, tous les facteurs ont le même poids
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

            return weights'''

    def _calculate_weights(self, factors_with_t_stats, market_weight=0.3, weight_type='t_stat'):
        """
        Calcule la pondération des facteurs

        Args:
            factors_with_t_stats: Dictionnaire {facteur: t_stat}
            market_weight: Poids fixe pour le marché (uniquement utilisé si weight_type='t_stat')
            weight_type: Type de pondération ('equal', 't_stat', 'erc' ou 'max_sharpe')

        Returns:
            Dictionnaire des poids normalisés
        """
        if not factors_with_t_stats:
            return {}

        # Mode Maximisation du Ratio de Sharpe
        if weight_type == 'max_sharpe':
            # Récupérer la liste des facteurs
            factors_list = list(factors_with_t_stats.keys())

            # S'il n'y a qu'un seul facteur, lui donner tout le poids
            if len(factors_list) == 1:
                return {factors_list[0]: 1.0}

            # Séparer market des autres facteurs
            other_factors = [f for f in factors_list if f != 'market']
            has_market = 'market' in factors_list

            # Période d'historique pour l'optimisation
            lookback_window = 52 * 15  # 10 ans de données hebdomadaires

            # Récupérer les rendements historiques pour l'optimisation
            latest_date = max(self.selector.factors_df.index)
            history_start = pd.Timestamp(latest_date) - pd.Timedelta(weeks=lookback_window)

            # Collecter les rendements
            returns_data = {}

            # Ajouter les rendements des facteurs standards
            valid_factors = [f for f in other_factors if f in self.selector.factors_df.columns]
            for factor in valid_factors:
                factor_data = self.selector.factors_df.loc[
                    self.selector.factors_df.index >= history_start, factor
                ].dropna()
                if not factor_data.empty:
                    returns_data[factor] = factor_data

            # Ajouter les rendements du marché si nécessaire
            if has_market:
                market_data = self.selector.market_return.loc[
                    self.selector.market_return.index >= history_start
                    ].dropna()
                if not market_data.empty:
                    returns_data['market'] = market_data

            # Si aucun facteur valide, utiliser équipondération
            if not returns_data:
                print("Aucun facteur valide pour Max Sharpe, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            # Créer un DataFrame des rendements alignés
            returns_df = pd.DataFrame(returns_data)
            returns_history = returns_df.dropna()

            if returns_history.empty or returns_history.shape[0] < 30:
                print("Historique insuffisant pour Max Sharpe, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            try:
                import scipy.optimize as sco

                # Calculer les rendements moyens et la matrice de covariance
                mean_returns = returns_history.mean()
                cov_matrix = returns_history.cov()

                # Vérifier que la matrice n'est pas vide
                if cov_matrix.shape[0] == 0:
                    return {factor: 1.0 / len(factors_list) for factor in factors_list}

                # Fonction objectif: maximiser le ratio de Sharpe
                def sharpe_ratio(weights):
                    weights = np.array(weights)
                    portfolio_return = np.sum(mean_returns * weights)
                    portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

                    if portfolio_vol <= 0:
                        return -1e10  # Pénalité élevée (négatif car on maximise)

                    # Ratio de Sharpe (sans taux sans risque)
                    return -portfolio_return / portfolio_vol  # Négatif car on minimise

                # Contraintes: somme des poids = 1
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

                # Bornes: poids entre 1% et 50%
                bounds = tuple((0.01, 0.5) for _ in range(len(cov_matrix)))

                # Point de départ: équipondération
                x0 = np.ones(len(cov_matrix)) / len(cov_matrix)

                # Optimisation
                result = sco.minimize(
                    sharpe_ratio,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success:
                    optimal_weights = result.x

                    # Initialiser le dictionnaire final avec EXACTEMENT les mêmes facteurs
                    weights_dict = {factor: 0.0 for factor in factors_list}

                    # Appliquer les poids optimaux uniquement aux facteurs qui sont dans la matrice de cov
                    for i, factor in enumerate(cov_matrix.index):
                        if factor in weights_dict:
                            weights_dict[factor] = optimal_weights[i]

                    # Si certains facteurs n'ont pas été inclus dans l'optimisation,
                    # leur attribuer un poids équipondéré du reste
                    missing_factors = [f for f in factors_list if f not in cov_matrix.index]
                    if missing_factors:
                        remaining_weight = 1.0 - sum(weights_dict[f] for f in factors_list if f not in missing_factors)
                        remaining_weight = max(0, remaining_weight)  # Éviter les valeurs négatives
                        for factor in missing_factors:
                            weights_dict[factor] = remaining_weight / len(missing_factors)

                    # Normalisation finale pour assurer que la somme est exactement 1
                    total_weight = sum(weights_dict.values())
                    if total_weight > 0:
                        return {k: v / total_weight for k, v in weights_dict.items()}
                    else:
                        return {factor: 1.0 / len(factors_list) for factor in factors_list}
                else:
                    print(f"Optimisation Max Sharpe échouée: {result.message}")
                    return {factor: 1.0 / len(factors_list) for factor in factors_list}

            except Exception as e:
                print(f"Erreur dans l'optimisation Max Sharpe: {e}")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

        # Mode ERC - Equal Risk Contribution
        elif weight_type == 'erc':
            # Code ERC existant...
            # Récupérer la liste des facteurs
            factors_list = list(factors_with_t_stats.keys())

            # S'il n'y a qu'un seul facteur, lui donner tout le poids
            if len(factors_list) == 1:
                return {factors_list[0]: 1.0}

            # Séparer market des autres facteurs
            other_factors = [f for f in factors_list if f != 'market']
            has_market = 'market' in factors_list

            # Réduire la période d'historique pour améliorer la stabilité
            lookback_window = 52 * 15  # 15 ans de données hebdomadaires

            # Récupérer les rendements historiques pour l'optimisation
            latest_date = max(self.selector.factors_df.index)
            history_start = pd.Timestamp(latest_date) - pd.Timedelta(weeks=lookback_window)

            # Collecter les rendements
            returns_data = {}

            # Ajouter les rendements des facteurs standards
            valid_factors = [f for f in other_factors if f in self.selector.factors_df.columns]
            for factor in valid_factors:
                factor_data = self.selector.factors_df.loc[
                    self.selector.factors_df.index >= history_start, factor
                ].dropna()
                if not factor_data.empty:
                    returns_data[factor] = factor_data

            # Ajouter les rendements du marché si nécessaire
            if has_market:
                market_data = self.selector.market_return.loc[
                    self.selector.market_return.index >= history_start
                    ].dropna()
                if not market_data.empty:
                    returns_data['market'] = market_data

            # Si aucun facteur valide, utiliser équipondération
            if not returns_data:
                print("Aucun facteur valide pour l'ERC, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            # Créer un DataFrame des rendements alignés
            returns_df = pd.DataFrame(returns_data)
            returns_history = returns_df.dropna()

            if returns_history.empty or returns_history.shape[0] < 30:
                print("Historique insuffisant pour l'ERC, utilisation méthode équipondérée")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            # Calculer la matrice de covariance
            cov_matrix = returns_history.cov()

            # Vérifier que la matrice n'est pas vide
            if cov_matrix.shape[0] == 0:
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

            try:
                import scipy.optimize as sco

                # Nombre de facteurs dans la matrice de covariance
                n = len(cov_matrix)

                # Fonction objectif ERC améliorée
                def erc_objective(weights):
                    weights = np.array(weights)

                    # Volatilité du portefeuille
                    portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)

                    if portfolio_vol <= 0:
                        return 1e10  # Pénalité élevée

                    # Contributions marginales au risque
                    mcr = cov_matrix.values @ weights / portfolio_vol

                    # Contributions au risque
                    rc = weights * mcr

                    # Objectif: que toutes les contributions au risque soient égales
                    target_risk = portfolio_vol / n
                    return np.sum((rc - target_risk) ** 2)

                # Contraintes: somme des poids = 1
                constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]

                # Bornes: poids entre 1% et 50%
                bounds = tuple((0.01, 0.5) for _ in range(n))

                # Point de départ: équipondération
                x0 = np.ones(n) / n

                # Optimisation
                result = sco.minimize(
                    erc_objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )

                if result.success:
                    optimal_weights = result.x

                    # Initialiser le dictionnaire final avec EXACTEMENT les mêmes facteurs
                    weights_dict = {factor: 0.0 for factor in factors_list}

                    # Appliquer les poids optimaux uniquement aux facteurs qui sont dans la matrice de cov
                    for i, factor in enumerate(cov_matrix.index):
                        if factor in weights_dict:
                            weights_dict[factor] = optimal_weights[i]

                    # Si certains facteurs n'ont pas été inclus dans l'optimisation,
                    # leur attribuer un poids équipondéré du reste
                    missing_factors = [f for f in factors_list if f not in cov_matrix.index]
                    if missing_factors:
                        remaining_weight = 1.0 - sum(weights_dict[f] for f in factors_list if f not in missing_factors)
                        remaining_weight = max(0, remaining_weight)  # Éviter les valeurs négatives
                        for factor in missing_factors:
                            weights_dict[factor] = remaining_weight / len(missing_factors)

                    # Normalisation finale pour assurer que la somme est exactement 1
                    total_weight = sum(weights_dict.values())
                    if total_weight > 0:
                        return {k: v / total_weight for k, v in weights_dict.items()}
                    else:
                        return {factor: 1.0 / len(factors_list) for factor in factors_list}
                else:
                    print(f"Optimisation ERC échouée: {result.message}")
                    return {factor: 1.0 / len(factors_list) for factor in factors_list}

            except Exception as e:
                print(f"Erreur dans l'optimisation ERC: {e}")
                return {factor: 1.0 / len(factors_list) for factor in factors_list}

        # En mode équipondération, tous les facteurs ont le même poids
        elif weight_type == 'equal':
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

    '''def _get_hold_returns(self, factors, start_date, end_date):
        """Récupère les rendements des facteurs pour la période de détention"""
        # Vérifie que tous les facteurs nécessaires sont présents
        valid_factors = [f for f in factors if f in self.selector.factors_df.columns]
        if not valid_factors:
            return pd.DataFrame()

        mask = (self.selector.factors_df.index > start_date) & \
               (self.selector.factors_df.index <= end_date)
        return self.selector.factors_df.loc[mask, valid_factors]'''

    def _get_hold_returns(self, factors, start_date, end_date):
        """Récupère les rendements des facteurs pour la période de détention"""
        # Créer un DataFrame pour stocker tous les rendements
        all_returns = pd.DataFrame()

        # Filtrer par période
        time_mask = (self.selector.factors_df.index > start_date) & (self.selector.factors_df.index <= end_date)

        # Récupérer les facteurs standards
        standard_factors = [f for f in factors if f in self.selector.factors_df.columns]
        if standard_factors:
            all_returns = self.selector.factors_df.loc[time_mask, standard_factors].copy()

        # Ajouter le rendement du marché si nécessaire
        if 'market' in factors and 'market' not in all_returns.columns:
            market_mask = (self.selector.market_return.index > start_date) & (
                        self.selector.market_return.index <= end_date)
            market_returns = self.selector.market_return.loc[market_mask]

            if not market_returns.empty:
                # Vérifier si all_returns est vide, sinon le joindre
                if all_returns.empty:
                    all_returns = pd.DataFrame(market_returns)
                    all_returns.columns = ['market']
                else:
                    all_returns['market'] = market_returns.reindex(all_returns.index)

        return all_returns

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

    # Exécuter le backtest avec rebalancement et stratégie ERC
    backtester = FactorStrategyBacktester(selector)
    results = backtester.backtest_strategy(
        initial_capital=100,
        weight_type='max_sharpe'
    )

    # Visualiser les résultats
    backtester.plot_performance()

    # Afficher les métriques
    metrics = backtester.get_performance_metrics()
    print("\nMétriques de performance (stratégie ERC):")
    for k, v in metrics.items():
        print(f"{k:20}: {v:.4f}")


'''import numpy as np
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
        print(f"{k:20}: {v:.4f}")'''