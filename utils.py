"""
Fonctions utilitaires pour l'analyse du dataset MovieLens 100K
Projet STT-3795 | Système de recommandation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_movielens_data(data_path: str = "ml-100k/") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les données MovieLens 100K depuis les fichiers sources.
    
    Parameters:
    -----------
    data_path : str
        Chemin vers le dossier contenant les données MovieLens
        
    Returns:
    --------
    tuple : (ratings_df, movies_df, users_df)
        Trois DataFrames contenant les données de notation, films et utilisateurs
    """
    try:
        # Chargement des données de notation
        ratings_df = pd.read_csv(
            f"{data_path}u.data", 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            dtype={'user_id': int, 'movie_id': int, 'rating': int, 'timestamp': int}
        )
        
        # Chargement des genres
        genres_df = pd.read_csv(
            f"{data_path}u.genre", 
            sep='|', 
            names=['genre', 'genre_id']
        )
        genre_columns = genres_df['genre'].tolist()
        
        # Chargement des informations sur les films
        movie_columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_columns
        movies_df = pd.read_csv(
            f"{data_path}u.item",
            sep='|',
            names=movie_columns,
            encoding='latin1'
        )
        
        # Chargement des informations utilisateurs
        users_df = pd.read_csv(
            f"{data_path}u.user",
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
        )
        
        print("✅ Données chargées avec succès !")
        print(f"   • {len(ratings_df):,} évaluations")
        print(f"   • {movies_df['movie_id'].nunique():,} films")
        print(f"   • {users_df['user_id'].nunique():,} utilisateurs")
        
        return ratings_df, movies_df, users_df
        
    except FileNotFoundError as e:
        print(f"❌ Erreur : Fichier non trouvé - {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None


def create_movie_stats(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée un DataFrame avec les statistiques par film.
    
    Parameters:
    -----------
    ratings_df : pd.DataFrame
        DataFrame des évaluations
    movies_df : pd.DataFrame
        DataFrame des films
        
    Returns:
    --------
    pd.DataFrame
        DataFrame avec les statistiques par film
    """
    # Statistiques de base par film
    movie_stats = ratings_df.groupby('movie_id').agg({
        'rating': ['count', 'mean', 'std']
    }).round(2)
    
    movie_stats.columns = ['rating_count', 'rating_mean', 'rating_std']
    movie_stats.reset_index(inplace=True)
    
    # Fusion avec les informations des films
    genre_columns = [col for col in movies_df.columns if col not in 
                    ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]
    
    movie_complete = pd.merge(
        movie_stats,
        movies_df[['movie_id', 'title'] + genre_columns],
        on='movie_id',
        how='left'
    )
    
    return movie_complete


def get_data_summary(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    """
    Affiche un résumé complet des données.
    """
    print("=" * 60)
    print("📊 RÉSUMÉ DES DONNÉES MOVIELENS 100K")
    print("=" * 60)
    
    # Statistiques générales
    print("\n🎬 STATISTIQUES GÉNÉRALES")
    print(f"• Nombre d'évaluations : {len(ratings_df):,}")
    print(f"• Nombre de films : {movies_df['movie_id'].nunique():,}")
    print(f"• Nombre d'utilisateurs : {users_df['user_id'].nunique():,}")
    print(f"• Période temporelle : {pd.to_datetime(ratings_df['timestamp'], unit='s').dt.date.min()} - {pd.to_datetime(ratings_df['timestamp'], unit='s').dt.date.max()}")
    
    # Statistiques des évaluations
    print("\n⭐ STATISTIQUES DES ÉVALUATIONS")
    print(f"• Note moyenne : {ratings_df['rating'].mean():.2f}/5")
    print(f"• Écart-type : {ratings_df['rating'].std():.2f}")
    print(f"• Distribution des notes :")
    rating_dist = ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / len(ratings_df)) * 100
        print(f"  - {rating}⭐ : {count:,} ({percentage:.1f}%)")
    
    # Statistiques des utilisateurs
    print("\n👥 STATISTIQUES DES UTILISATEURS")
    print(f"• Âge moyen : {users_df['age'].mean():.1f} ans")
    print(f"• Répartition par genre : {users_df['gender'].value_counts().to_dict()}")
    print(f"• Nombre d'évaluations par utilisateur : {ratings_df.groupby('user_id')['rating'].count().mean():.1f}")
    
    # Top genres
    print("\n🎭 TOP GENRES")
    genre_columns = [col for col in movies_df.columns if col not in 
                    ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]
    genre_counts = movies_df[genre_columns].sum().sort_values(ascending=False)
    for genre, count in genre_counts.head(5).items():
        print(f"• {genre} : {count} films")


def plot_rating_distribution(ratings_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Visualise la distribution des évaluations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution des notes
    ratings_df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution des évaluations', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Note')
    axes[0].set_ylabel('Nombre d\'évaluations')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Histogramme des notes
    axes[1].hist(ratings_df['rating'], bins=5, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1].set_title('Histogramme des évaluations', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Note')
    axes[1].set_ylabel('Fréquence')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_movie_popularity(movie_stats_df: pd.DataFrame, min_ratings: int = 1, save_path: str = None) -> None:
    """
    Visualise la popularité des films.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Distribution du nombre d'évaluations par film
    axes[0, 0].hist(movie_stats_df['rating_count'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 0].set_title('Distribution du nombre d\'évaluations par film', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Nombre d\'évaluations')
    axes[0, 0].set_ylabel('Nombre de films')
    
    # Distribution des notes moyennes
    filtered_data = movie_stats_df[movie_stats_df['rating_count'] >= min_ratings]
    axes[0, 1].hist(filtered_data['rating_mean'], bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 1].set_title(f'Distribution des notes moyennes\n(films avec ≥{min_ratings} évaluations)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Note moyenne')
    axes[0, 1].set_ylabel('Nombre de films')
    
    # Relation entre popularité et note moyenne
    axes[1, 0].scatter(movie_stats_df['rating_mean'], movie_stats_df['rating_count'], 
                      alpha=0.6, color='purple', s=30)
    axes[1, 0].set_title('Relation entre note moyenne et popularité', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Note moyenne')
    axes[1, 0].set_ylabel('Nombre d\'évaluations')
    
    # Top 10 des films les mieux notés (avec un minimum d'évaluations)
    top_movies = movie_stats_df[movie_stats_df['rating_count'] >= 50].nlargest(10, 'rating_mean')
    axes[1, 1].barh(range(len(top_movies)), top_movies['rating_mean'], color='coral')
    axes[1, 1].set_yticks(range(len(top_movies)))
    axes[1, 1].set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                               for title in top_movies['title']], fontsize=10)
    axes[1, 1].set_title('Top 10 des films les mieux notés\n(≥50 évaluations)', 
                        fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Note moyenne')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_genres(movies_df: pd.DataFrame, ratings_df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Analyse les genres de films.
    """
    # Identification des colonnes de genres
    genre_columns = [col for col in movies_df.columns if col not in 
                    ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']]
    
    # Statistiques par genre
    genre_stats = []
    
    for genre in genre_columns:
        genre_movies = movies_df[movies_df[genre] == 1]['movie_id']
        genre_ratings = ratings_df[ratings_df['movie_id'].isin(genre_movies)]
        
        if len(genre_ratings) > 0:
            stats = {
                'genre': genre,
                'film_count': len(genre_movies),
                'total_ratings': len(genre_ratings),
                'avg_rating': genre_ratings['rating'].mean(),
                'std_rating': genre_ratings['rating'].std()
            }
            genre_stats.append(stats)
    
    genre_analysis = pd.DataFrame(genre_stats).round(3)
    genre_analysis = genre_analysis.sort_values('film_count', ascending=False)
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Nombre de films par genre
    axes[0, 0].bar(range(len(genre_analysis)), genre_analysis['film_count'], color='steelblue')
    axes[0, 0].set_title('Nombre de films par genre', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(range(len(genre_analysis)))
    axes[0, 0].set_xticklabels(genre_analysis['genre'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Nombre de films')
    
    # Note moyenne par genre
    sorted_by_rating = genre_analysis.sort_values('avg_rating', ascending=False)
    axes[0, 1].bar(range(len(sorted_by_rating)), sorted_by_rating['avg_rating'], color='darkorange')
    axes[0, 1].set_title('Note moyenne par genre', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(range(len(sorted_by_rating)))
    axes[0, 1].set_xticklabels(sorted_by_rating['genre'], rotation=45, ha='right')
    axes[0, 1].set_ylabel('Note moyenne')
    
    # Nombre total d'évaluations par genre
    sorted_by_ratings = genre_analysis.sort_values('total_ratings', ascending=False)
    axes[1, 0].bar(range(len(sorted_by_ratings)), sorted_by_ratings['total_ratings'], color='green')
    axes[1, 0].set_title('Nombre total d\'évaluations par genre', fontsize=12, fontweight='bold')
    axes[1, 0].set_xticks(range(len(sorted_by_ratings)))
    axes[1, 0].set_xticklabels(sorted_by_ratings['genre'], rotation=45, ha='right')
    axes[1, 0].set_ylabel('Nombre d\'évaluations')
    
    # Écart-type des notes par genre
    sorted_by_std = genre_analysis.sort_values('std_rating', ascending=False)
    axes[1, 1].bar(range(len(sorted_by_std)), sorted_by_std['std_rating'], color='red')
    axes[1, 1].set_title('Écart-type des notes par genre', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(range(len(sorted_by_std)))
    axes[1, 1].set_xticklabels(sorted_by_std['genre'], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Écart-type')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return genre_analysis


def get_movie_recommendations_preview(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, 
                                    user_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Aperçu simple des recommandations basé sur la popularité et les notes moyennes.
    """
    # Films déjà notés par l'utilisateur
    user_movies = set(ratings_df[ratings_df['user_id'] == user_id]['movie_id'])
    
    # Calcul des statistiques des films
    movie_stats = create_movie_stats(ratings_df, movies_df)
    
    # Filtrer les films non vus avec un minimum d'évaluations
    unseen_movies = movie_stats[~movie_stats['movie_id'].isin(user_movies)]
    popular_movies = unseen_movies[unseen_movies['rating_count'] >= 20]
    
    # Recommandations basées sur note moyenne pondérée
    recommendations = popular_movies.nlargest(top_n, 'rating_mean')
    
    return recommendations[['movie_id', 'title', 'rating_mean', 'rating_count']].reset_index(drop=True)


def create_user_movie_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée une matrice utilisateur-film pour les systèmes de recommandation.
    """
    user_movie_matrix = ratings_df.pivot_table(
        index='user_id', 
        columns='movie_id', 
        values='rating',
        fill_value=0
    )
    
    print(f"Matrice créée : {user_movie_matrix.shape[0]} utilisateurs × {user_movie_matrix.shape[1]} films")
    print(f"Densité de la matrice : {(user_movie_matrix != 0).sum().sum() / user_movie_matrix.size:.2%}")
    
    return user_movie_matrix