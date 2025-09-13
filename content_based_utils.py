"""
Fonctions utilitaires pour le système de recommandation basé sur le contenu (Content-Based Filtering)
Projet STT-3795 | Système de recommandation MovieLens

Ce module contient toutes les fonctions nécessaires pour implémenter un système de
recommandation basé sur les caractéristiques des films (genres, métadonnées).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Dict, Optional, Union
import warnings
from datetime import datetime
import re
from scipy.sparse import csr_matrix
from scipy import stats

warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ContentBasedRecommender:
    """
    Classe principale pour les recommandations basées sur le contenu.
    
    Cette classe implémente plusieurs variantes de recommandation content-based :
    - Basée uniquement sur les genres
    - Pondérée par les notes moyennes
    - Filtrée par popularité
    - Avec normalisation TF-IDF des genres
    """
    
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None):
        """
        Initialise le système de recommandation.
        
        Parameters:
        -----------
        movies_df : pd.DataFrame
            DataFrame des films avec genres et métadonnées
        ratings_df : pd.DataFrame, optional
            DataFrame des évaluations pour calculer les statistiques
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df
        self.genre_columns = self._identify_genre_columns()
        self.movie_stats = None
        self.similarity_matrix = None
        self.feature_matrix = None
        self.scaler = None
        
        # Calcul automatique des statistiques si ratings_df fourni
        if ratings_df is not None:
            self.movie_stats = self._calculate_movie_stats()
            self.movies_df = pd.merge(self.movies_df, self.movie_stats, on='movie_id', how='left')
    
    def _identify_genre_columns(self) -> List[str]:
        """Identifie automatiquement les colonnes de genres."""
        exclude_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
        genre_cols = [col for col in self.movies_df.columns if col not in exclude_cols]
        
        # Vérifier que ce sont bien des colonnes binaires (0/1)
        for col in genre_cols:
            unique_vals = self.movies_df[col].unique()
            if not set(unique_vals).issubset({0, 1, np.nan}):
                genre_cols.remove(col)
        
        return genre_cols
    
    def _calculate_movie_stats(self) -> pd.DataFrame:
        """Calcule les statistiques par film."""
        if self.ratings_df is None:
            raise ValueError("ratings_df requis pour calculer les statistiques")
        
        stats = self.ratings_df.groupby('movie_id').agg({
            'rating': ['count', 'mean', 'std', 'median']
        }).round(3)
        
        stats.columns = ['rating_count', 'rating_mean', 'rating_std', 'rating_median']
        stats['rating_std'] = stats['rating_std'].fillna(0)
        stats.reset_index(inplace=True)
        
        return stats
    
    def create_feature_matrix(self, method: str = 'basic', 
                            rating_weight: float = 1.0,
                            popularity_weight: float = 1.0,
                            min_ratings: int = 1) -> np.ndarray:
        """
        Crée la matrice de caractéristiques pour les films.
        
        Parameters:
        -----------
        method : str
            Méthode de création ('basic', 'weighted', 'tfidf', 'normalized')
        rating_weight : float
            Poids pour la pondération par note moyenne
        popularity_weight : float
            Poids pour la pondération par popularité
        min_ratings : int
            Nombre minimum d'évaluations requis
            
        Returns:
        --------
        np.ndarray : Matrice de caractéristiques
        """
        # Matrice de base des genres
        genre_matrix = self.movies_df[self.genre_columns].fillna(0).values
        
        if method == 'basic':
            self.feature_matrix = genre_matrix
            
        elif method == 'weighted' and self.movie_stats is not None:
            # Pondération par note moyenne et popularité
            ratings = self.movies_df['rating_mean'].fillna(3.0).values
            counts = self.movies_df['rating_count'].fillna(1).values
            
            # Normalisation des poids
            rating_weights = ((ratings - 1) / 4) * rating_weight + 1  # [1, 2.25]
            pop_weights = np.log1p(counts) * popularity_weight / 10 + 1  # Poids logarithmique
            
            # Application des poids
            weighted_matrix = genre_matrix * rating_weights.reshape(-1, 1) * pop_weights.reshape(-1, 1)
            self.feature_matrix = weighted_matrix
            
        elif method == 'tfidf':
            # Normalisation TF-IDF des genres
            tfidf = TfidfVectorizer()
            
            # Conversion des genres en "documents" textuels
            genre_docs = []
            for _, row in self.movies_df.iterrows():
                active_genres = [col for col in self.genre_columns if row[col] == 1]
                genre_docs.append(' '.join(active_genres))
            
            self.feature_matrix = tfidf.fit_transform(genre_docs).toarray()
            
        elif method == 'normalized':
            # Normalisation standard
            scaler = StandardScaler()
            self.feature_matrix = scaler.fit_transform(genre_matrix)
            self.scaler = scaler
            
        else:
            raise ValueError(f"Méthode non reconnue: {method}")
        
        # Filtrage par nombre minimum d'évaluations
        if min_ratings > 1 and self.movie_stats is not None:
            mask = self.movies_df['rating_count'].fillna(0) >= min_ratings
            self.feature_matrix = self.feature_matrix * mask.values.reshape(-1, 1)
        
        return self.feature_matrix
    
    def compute_similarity_matrix(self, metric: str = 'cosine') -> np.ndarray:
        """
        Calcule la matrice de similarité entre les films.
        
        Parameters:
        -----------
        metric : str
            Métrique de similarité ('cosine', 'euclidean', 'manhattan', 'correlation')
            
        Returns:
        --------
        np.ndarray : Matrice de similarité
        """
        if self.feature_matrix is None:
            raise ValueError("Créez d'abord la matrice de caractéristiques avec create_feature_matrix()")
        
        if metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
        elif metric == 'euclidean':
            # Convertir distance en similarité (plus proche de 1 = plus similaire)
            distances = euclidean_distances(self.feature_matrix)
            self.similarity_matrix = 1 / (1 + distances)
        elif metric == 'manhattan':
            distances = manhattan_distances(self.feature_matrix)
            self.similarity_matrix = 1 / (1 + distances)
        elif metric == 'correlation':
            # Corrélation de Pearson
            self.similarity_matrix = np.corrcoef(self.feature_matrix)
            self.similarity_matrix = np.nan_to_num(self.similarity_matrix)
        else:
            raise ValueError(f"Métrique non supportée: {metric}")
        
        return self.similarity_matrix
    
    def get_movie_recommendations(self, movie_title: str, n_recommendations: int = 10,
                                min_similarity: float = 0.0) -> pd.DataFrame:
        """
        Obtient des recommandations pour un film donné.
        
        Parameters:
        -----------
        movie_title : str
            Titre du film de référence
        n_recommendations : int
            Nombre de recommandations à retourner
        min_similarity : float
            Seuil de similarité minimum
            
        Returns:
        --------
        pd.DataFrame : Recommandations avec scores de similarité
        """
        if self.similarity_matrix is None:
            raise ValueError("Calculez d'abord la matrice de similarité avec compute_similarity_matrix()")
        
        # Trouver l'index du film
        movie_matches = self.movies_df[self.movies_df['title'] == movie_title]
        if movie_matches.empty:
            # Recherche approximative
            similar_titles = self.movies_df[self.movies_df['title'].str.contains(
                movie_title, case=False, na=False)]['title'].tolist()
            raise ValueError(f"Film '{movie_title}' non trouvé. Films similaires: {similar_titles[:5]}")
        
        movie_idx = movie_matches.index[0]
        
        # Obtenir les scores de similarité
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        similarity_scores = [(idx, score) for idx, score in similarity_scores 
                           if score >= min_similarity and idx != movie_idx]
        
        # Trier par similarité décroissante
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sélectionner les top N
        top_indices = [idx for idx, _ in similarity_scores[:n_recommendations]]
        top_scores = [score for _, score in similarity_scores[:n_recommendations]]
        
        # Créer le DataFrame de recommandations
        recommendations = self.movies_df.iloc[top_indices].copy()
        recommendations['similarity_score'] = top_scores
        
        # Ajouter les informations de rang
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        # Réorganiser les colonnes
        columns = ['rank', 'movie_id', 'title', 'similarity_score']
        if 'rating_mean' in recommendations.columns:
            columns.append('rating_mean')
        if 'rating_count' in recommendations.columns:
            columns.append('rating_count')
        columns.extend(self.genre_columns)
        
        return recommendations[columns].reset_index(drop=True)
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10,
                               rating_threshold: float = 4.0) -> pd.DataFrame:
        """
        Recommandations basées sur l'historique d'un utilisateur.
        
        Parameters:
        -----------
        user_id : int
            ID de l'utilisateur
        n_recommendations : int
            Nombre de recommandations
        rating_threshold : float
            Seuil pour considérer qu'un film est "aimé"
            
        Returns:
        --------
        pd.DataFrame : Recommandations personnalisées
        """
        if self.ratings_df is None:
            raise ValueError("ratings_df requis pour les recommandations utilisateur")
        
        # Films aimés par l'utilisateur
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        if user_ratings.empty:
            raise ValueError(f"Utilisateur {user_id} non trouvé")
        
        liked_movies = user_ratings[user_ratings['rating'] >= rating_threshold]['movie_id'].tolist()
        if not liked_movies:
            # Prendre les films les mieux notés par l'utilisateur
            liked_movies = user_ratings.nlargest(3, 'rating')['movie_id'].tolist()
        
        # Films déjà vus par l'utilisateur
        seen_movies = set(user_ratings['movie_id'].tolist())
        
        # Calculer le profil utilisateur (moyenne des caractéristiques des films aimés)
        liked_indices = [self.movies_df[self.movies_df['movie_id'] == mid].index[0] 
                        for mid in liked_movies 
                        if not self.movies_df[self.movies_df['movie_id'] == mid].empty]
        
        if not liked_indices:
            raise ValueError("Aucun film aimé trouvé dans la base de données")
        
        user_profile = np.mean(self.feature_matrix[liked_indices], axis=0)
        
        # Calculer la similarité avec tous les films
        similarities = cosine_similarity([user_profile], self.feature_matrix)[0]
        
        # Créer les recommandations
        movie_similarities = list(enumerate(similarities))
        movie_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filtrer les films déjà vus et sélectionner les top N
        recommendations_indices = []
        recommendations_scores = []
        
        for idx, score in movie_similarities:
            movie_id = self.movies_df.iloc[idx]['movie_id']
            if movie_id not in seen_movies:
                recommendations_indices.append(idx)
                recommendations_scores.append(score)
                if len(recommendations_indices) >= n_recommendations:
                    break
        
        # Créer le DataFrame final
        recommendations = self.movies_df.iloc[recommendations_indices].copy()
        recommendations['similarity_score'] = recommendations_scores
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations[['rank', 'movie_id', 'title', 'similarity_score', 
                              'rating_mean', 'rating_count'] + self.genre_columns].reset_index(drop=True)


def analyze_content_features(movies_df: pd.DataFrame, genre_columns: List[str]) -> Dict:
    """
    Analyse les caractéristiques de contenu du dataset.
    
    Parameters:
    -----------
    movies_df : pd.DataFrame
        DataFrame des films
    genre_columns : List[str]
        Liste des colonnes de genres
        
    Returns:
    --------
    Dict : Statistiques des caractéristiques
    """
    analysis = {}
    
    # Statistiques des genres
    genre_counts = movies_df[genre_columns].sum().sort_values(ascending=False)
    analysis['genre_distribution'] = genre_counts.to_dict()
    
    # Films multi-genres
    movies_df['genre_count'] = movies_df[genre_columns].sum(axis=1)
    analysis['avg_genres_per_movie'] = movies_df['genre_count'].mean()
    analysis['max_genres_per_movie'] = movies_df['genre_count'].max()
    analysis['movies_without_genre'] = (movies_df['genre_count'] == 0).sum()
    
    # Combinaisons de genres les plus fréquentes
    genre_combinations = {}
    for _, row in movies_df.iterrows():
        active_genres = [col for col in genre_columns if row[col] == 1]
        if active_genres:
            combo = tuple(sorted(active_genres))
            genre_combinations[combo] = genre_combinations.get(combo, 0) + 1
    
    # Top 10 des combinaisons
    top_combinations = sorted(genre_combinations.items(), key=lambda x: x[1], reverse=True)[:10]
    analysis['top_genre_combinations'] = [('+'.join(combo), count) for combo, count in top_combinations]
    
    return analysis


def plot_genre_analysis(movies_df: pd.DataFrame, genre_columns: List[str], 
                       save_path: str = None) -> None:
    """
    Visualise l'analyse des genres.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribution des genres
    genre_counts = movies_df[genre_columns].sum().sort_values(ascending=False)
    axes[0, 0].bar(range(len(genre_counts)), genre_counts.values, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Distribution des Genres', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(len(genre_counts)))
    axes[0, 0].set_xticklabels(genre_counts.index, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Nombre de films')
    
    # Nombre de genres par film
    movies_df['genre_count'] = movies_df[genre_columns].sum(axis=1)
    genre_count_dist = movies_df['genre_count'].value_counts().sort_index()
    axes[0, 1].bar(genre_count_dist.index, genre_count_dist.values, color='darkgreen', alpha=0.8)
    axes[0, 1].set_title('Distribution du nombre de genres par film', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Nombre de genres')
    axes[0, 1].set_ylabel('Nombre de films')
    
    # Matrice de corrélation des genres
    genre_corr = movies_df[genre_columns].corr()
    im = axes[1, 0].imshow(genre_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_title('Matrice de Corrélation des Genres', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(len(genre_columns)))
    axes[1, 0].set_yticks(range(len(genre_columns)))
    axes[1, 0].set_xticklabels(genre_columns, rotation=45, ha='right')
    axes[1, 0].set_yticklabels(genre_columns)
    plt.colorbar(im, ax=axes[1, 0])
    
    # Heatmap de cooccurrence des genres
    cooccurrence = np.dot(movies_df[genre_columns].T, movies_df[genre_columns])
    sns.heatmap(cooccurrence, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=genre_columns, yticklabels=genre_columns)
    axes[1, 1].set_title('Matrice de Co-occurrence des Genres', fontsize=14, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_content_recommendations(recommender: ContentBasedRecommender,
                                   test_movies: List[str],
                                   n_recommendations: int = 10) -> Dict:
    """
    Évalue la qualité des recommandations content-based.
    
    Parameters:
    -----------
    recommender : ContentBasedRecommender
        Instance du recommandeur
    test_movies : List[str]
        Liste de films de test
    n_recommendations : int
        Nombre de recommandations par film
        
    Returns:
    --------
    Dict : Métriques d'évaluation
    """
    results = {
        'movie_evaluations': [],
        'avg_similarity': 0,
        'genre_diversity': 0,
        'popularity_bias': 0
    }
    
    total_similarity = 0
    total_diversity = 0
    total_popularity = 0
    valid_tests = 0
    
    for movie in test_movies:
        try:
            recs = recommender.get_movie_recommendations(movie, n_recommendations)
            
            if not recs.empty:
                # Similarité moyenne
                avg_sim = recs['similarity_score'].mean()
                total_similarity += avg_sim
                
                # Diversité des genres (nombre de genres uniques dans les recommandations)
                genre_cols = recommender.genre_columns
                unique_genres = recs[genre_cols].sum().astype(bool).sum()
                diversity = unique_genres / len(genre_cols)
                total_diversity += diversity
                
                # Biais de popularité (moyenne des ratings_count)
                if 'rating_count' in recs.columns:
                    avg_popularity = recs['rating_count'].mean()
                    total_popularity += avg_popularity
                
                results['movie_evaluations'].append({
                    'movie': movie,
                    'avg_similarity': avg_sim,
                    'genre_diversity': diversity,
                    'avg_popularity': avg_popularity if 'rating_count' in recs.columns else None
                })
                
                valid_tests += 1
                
        except ValueError:
            continue
    
    if valid_tests > 0:
        results['avg_similarity'] = total_similarity / valid_tests
        results['genre_diversity'] = total_diversity / valid_tests
        results['popularity_bias'] = total_popularity / valid_tests if total_popularity > 0 else 0
    
    return results


def create_movie_profile_visualization(recommender: ContentBasedRecommender,
                                     movie_title: str, 
                                     save_path: str = None) -> None:
    """
    Visualise le profil d'un film et ses recommandations.
    """
    try:
        # Obtenir les recommandations
        recommendations = recommender.get_movie_recommendations(movie_title, 10)
        
        # Trouver le film original
        original_movie = recommender.movies_df[
            recommender.movies_df['title'] == movie_title
        ].iloc[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Profil de genres du film original
        original_genres = [col for col in recommender.genre_columns if original_movie[col] == 1]
        if original_genres:
            axes[0, 0].bar(range(len(original_genres)), [1] * len(original_genres), 
                          color='lightcoral', alpha=0.8)
            axes[0, 0].set_title(f'Profil de Genres: {movie_title}', fontweight='bold')
            axes[0, 0].set_xticks(range(len(original_genres)))
            axes[0, 0].set_xticklabels(original_genres, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Présence')
        
        # 2. Distribution des scores de similarité
        axes[0, 1].hist(recommendations['similarity_score'], bins=10, 
                       color='skyblue', alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Distribution des Scores de Similarité', fontweight='bold')
        axes[0, 1].set_xlabel('Score de similarité')
        axes[0, 1].set_ylabel('Nombre de films')
        
        # 3. Top recommandations
        top_5 = recommendations.head(5)
        y_pos = np.arange(len(top_5))
        axes[1, 0].barh(y_pos, top_5['similarity_score'], color='lightgreen', alpha=0.8)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                                   for title in top_5['title']])
        axes[1, 0].set_title('Top 5 Recommandations', fontweight='bold')
        axes[1, 0].set_xlabel('Score de similarité')
        
        # 4. Analyse des genres dans les recommandations
        if 'rating_mean' in recommendations.columns and 'rating_count' in recommendations.columns:
            scatter = axes[1, 1].scatter(recommendations['rating_count'], 
                                       recommendations['rating_mean'],
                                       c=recommendations['similarity_score'],
                                       cmap='viridis', alpha=0.7, s=60)
            axes[1, 1].set_title('Popularité vs Qualité des Recommandations', fontweight='bold')
            axes[1, 1].set_xlabel('Nombre d\'évaluations')
            axes[1, 1].set_ylabel('Note moyenne')
            plt.colorbar(scatter, ax=axes[1, 1], label='Score de similarité')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Erreur lors de la visualisation: {e}")


def compare_similarity_metrics(movies_df: pd.DataFrame, genre_columns: List[str],
                             test_movie: str, n_recommendations: int = 5) -> pd.DataFrame:
    """
    Compare différentes métriques de similarité pour un film donné.
    """
    metrics = ['cosine', 'euclidean', 'manhattan', 'correlation']
    results = []
    
    for metric in metrics:
        try:
            recommender = ContentBasedRecommender(movies_df)
            recommender.create_feature_matrix(method='basic')
            recommender.compute_similarity_matrix(metric=metric)
            recs = recommender.get_movie_recommendations(test_movie, n_recommendations)
            
            for _, rec in recs.iterrows():
                results.append({
                    'metric': metric,
                    'rank': rec['rank'],
                    'title': rec['title'],
                    'similarity_score': rec['similarity_score']
                })
                
        except Exception as e:
            print(f"Erreur avec la métrique {metric}: {e}")
            continue
    
    comparison_df = pd.DataFrame(results)
    
    if not comparison_df.empty:
        # Pivot pour avoir les métriques en colonnes
        pivot_df = comparison_df.pivot_table(
            index=['rank', 'title'], 
            columns='metric', 
            values='similarity_score'
        ).reset_index()
        
        return pivot_df
    else:
        return pd.DataFrame()


def extract_year_from_title(title: str) -> Optional[int]:
    """
    Extrait l'année de sortie depuis le titre du film.
    
    Parameters:
    -----------
    title : str
        Titre du film (format: "Film Title (YYYY)")
        
    Returns:
    --------
    Optional[int] : Année de sortie ou None
    """
    # Recherche du pattern (YYYY) à la fin du titre
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return int(match.group(1))
    return None


def enhance_movies_with_metadata(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame des films avec des métadonnées additionnelles.
    
    Parameters:
    -----------
    movies_df : pd.DataFrame
        DataFrame original des films
        
    Returns:
    --------
    pd.DataFrame : DataFrame enrichi
    """
    enhanced_df = movies_df.copy()
    
    # Extraction de l'année depuis le titre
    enhanced_df['year'] = enhanced_df['title'].apply(extract_year_from_title)
    
    # Décennie
    enhanced_df['decade'] = (enhanced_df['year'] // 10 * 10).astype('Int64')
    
    # Ère cinématographique
    def get_movie_era(year):
        if pd.isna(year):
            return 'Unknown'
        elif year < 1930:
            return 'Silent Era'
        elif year < 1960:
            return 'Golden Age'
        elif year < 1980:
            return 'New Hollywood'
        elif year < 2000:
            return 'Blockbuster Era'
        else:
            return 'Modern Era'
    
    enhanced_df['era'] = enhanced_df['year'].apply(get_movie_era)
    
    # Nombre de genres par film
    genre_columns = [col for col in movies_df.columns 
                    if col not in ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]
    enhanced_df['genre_count'] = enhanced_df[genre_columns].sum(axis=1)
    
    # Genre principal (le premier genre trouvé)
    def get_primary_genre(row):
        for genre in genre_columns:
            if row[genre] == 1:
                return genre
        return 'Unknown'
    
    enhanced_df['primary_genre'] = enhanced_df.apply(get_primary_genre, axis=1)
    
    return enhanced_df