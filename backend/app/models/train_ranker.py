"""
Train Ranking Models - LightGBM & XGBoost with Visualization
-------------------------------------------------------------
Trains both models, compares performance, generates plots, and saves models.
Uses your actual supplier data from suppliers_clean.csv.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error
from typing import Tuple, Dict
from datetime import datetime

# Set plot style
sns.set_style( "whitegrid" )
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Check for required libraries
try :
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError :
    LGBM_AVAILABLE = False
    print( "⚠️ LightGBM not installed. Run: pip install lightgbm" )

try :
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError :
    XGB_AVAILABLE = False
    print( "⚠️ XGBoost not installed. Run: pip install xgboost" )

if not LGBM_AVAILABLE and not XGB_AVAILABLE :
    print( "❌ Neither LightGBM nor XGBoost is installed. Exiting." )
    exit( 1 )

# Paths
BASE_DIR = "C:/Users/somas/PycharmProjects/SourceUp"
CLEAN_DATA = f"{BASE_DIR}/data/clean/suppliers_clean.csv"
TRAINING_DATA = f"{BASE_DIR}/data/training/ranking_data.csv"
MODELS_DIR = f"{BASE_DIR}/backend/app/models/embeddings"
PLOTS_DIR = f"{BASE_DIR}/data/training/plots"
LGBM_MODEL_PATH = f"{MODELS_DIR}/ranker_lightgbm.pkl"
XGB_MODEL_PATH = f"{MODELS_DIR}/ranker_xgboost.pkl"

# Create plots directory
os.makedirs( PLOTS_DIR, exist_ok=True )


# ============================================================================
# DATA GENERATION FROM YOUR SUPPLIER DATA
# ============================================================================

def generate_training_data_from_suppliers () -> pd.DataFrame :
    """
    Generate synthetic training data from your actual supplier data.

    This simulates user interactions:
    - Creates queries based on product categories
    - Simulates clicks, quotes, and purchases
    - Generates relevance scores

    In production, replace this with real user interaction data.
    """
    if not os.path.exists( CLEAN_DATA ) :
        raise FileNotFoundError(
            f"Supplier data not found at {CLEAN_DATA}\n"
            f"Run: python pipeline/run_all.py"
        )

    print( f"📂 Loading supplier data from {CLEAN_DATA}" )
    suppliers = pd.read_csv( CLEAN_DATA )
    print( f"📊 Loaded {len( suppliers )} suppliers" )

    # Extract unique products to create synthetic queries
    products = suppliers['product name'].dropna().unique()

    # Generate synthetic queries (simulating user searches)
    np.random.seed( 42 )
    n_queries = min( 200, len( products ) )  # 200 unique queries
    query_products = np.random.choice( products, n_queries, replace=False )

    training_records = []
    query_id = 0

    for product in query_products :
        # Find relevant suppliers for this product
        relevant_suppliers = suppliers[
            suppliers['product name'].str.contains( product, case=False, na=False )
        ].head( 15 )  # Top 15 suppliers per query

        if len( relevant_suppliers ) < 2 :
            continue  # Skip if too few suppliers

        # Generate random query constraints
        query_max_price = np.random.choice( [0.30, 0.50, 1.00, 2.00, None] )
        query_location = np.random.choice( ['china', 'india', 'vietnam', None] )
        query_cert = np.random.choice( ['iso', 'fda', 'ce', None] )

        # Extract features for each supplier in this query
        for idx, supplier in relevant_suppliers.iterrows() :
            features = extract_features_from_supplier(
                supplier,
                query_max_price,
                query_location,
                query_cert
            )

            # Generate synthetic relevance score (0-5 scale)
            # Based on feature quality + randomness
            relevance = calculate_synthetic_relevance( features )

            features['query_id'] = query_id
            features['relevance'] = relevance
            training_records.append( features )

        query_id += 1

    df = pd.DataFrame( training_records )
    print( f"✅ Generated {len( df )} training samples from {query_id} queries" )

    return df


def extract_features_from_supplier ( supplier: pd.Series,
                                     query_max_price: float,
                                     query_location: str,
                                     query_cert: str ) -> Dict :
    """Extract features from supplier row."""

    # Parse price
    supplier_price = parse_price( supplier.get( 'price min', supplier.get( 'price', 0 ) ) )

    # Price features
    if query_max_price :
        price_match = 1.0 if (supplier_price > 0 and supplier_price <= query_max_price) else 0.0
        price_ratio = supplier_price / query_max_price if query_max_price > 0 else 0.0
        price_distance = abs( supplier_price - query_max_price ) / query_max_price if query_max_price > 0 else 1.0
    else :
        price_match = 1.0
        price_ratio = 1.0
        price_distance = 0.0

    # Location features
    supplier_location = str( supplier.get( 'supplier location', supplier.get( 'location', '' ) ) ).lower()
    if query_location :
        if query_location in supplier_location :
            location_match = 1.0
        elif supplier_location and query_location :
            location_match = 0.3
        else :
            location_match = 0.0
    else :
        location_match = 0.5

    # Certification features
    supplier_certs = str( supplier.get( 'certifications', '' ) ).lower()
    if query_cert :
        cert_match = 1.0 if query_cert in supplier_certs else 0.0
    else :
        cert_match = 0.5

    # Experience
    years = supplier.get( 'years with gs', 0 )
    try :
        years_normalized = min( float( years ) / 10.0, 1.0 ) if years else 0.0
    except :
        years_normalized = 0.0

    # Business type
    business_type = str( supplier.get( 'business type', '' ) ).lower()
    is_manufacturer = 1.0 if 'manufacturer' in business_type else 0.0
    is_trading_company = 1.0 if 'trading company' in business_type else 0.0

    # Simulate FAISS score (in real system, this comes from retriever)
    faiss_score = np.random.uniform( 0.6, 0.95 )
    faiss_rank = np.random.randint( 1, 50 )

    return {
        'price_match' : price_match,
        'price_ratio' : price_ratio,
        'price_distance' : price_distance,
        'location_match' : location_match,
        'cert_match' : cert_match,
        'years_normalized' : years_normalized,
        'is_manufacturer' : is_manufacturer,
        'is_trading_company' : is_trading_company,
        'faiss_score' : faiss_score,
        'faiss_rank' : faiss_rank,
    }


def parse_price ( price_value ) -> float :
    """Parse price from various formats."""
    if price_value is None or (isinstance( price_value, float ) and np.isnan( price_value )) :
        return 0.0

    try :
        if isinstance( price_value, (int, float) ) :
            return float( price_value )

        price_str = str( price_value ).strip()
        if '-' in price_str :
            parts = price_str.split( '-' )
            return float( parts[0].strip() )

        return float( price_str )
    except (ValueError, AttributeError) :
        return 0.0


def calculate_synthetic_relevance ( features: Dict ) -> float :
    """
    Calculate synthetic relevance score (0-5).

    Simulates user behavior:
    - High score = user clicked, requested quote, or purchased
    - Low score = user ignored or rejected
    """
    base_score = (
            features['price_match'] * 1.5 +
            (1 - features['price_distance']) * 0.8 +
            features['location_match'] * 1.2 +
            features['cert_match'] * 1.0 +
            features['years_normalized'] * 0.5 +
            features['is_manufacturer'] * 0.7 +
            features['faiss_score'] * 0.8
    )

    # Add noise to simulate real-world variance
    noise = np.random.normal( 0, 0.3 )
    score = base_score + noise

    # Clip to 0-5 range
    return np.clip( score, 0, 5 )


def load_or_generate_data () -> pd.DataFrame :
    """Load existing training data or generate from supplier data."""
    if os.path.exists( TRAINING_DATA ) :
        print( f"📂 Loading training data from {TRAINING_DATA}" )
        return pd.read_csv( TRAINING_DATA )
    else :
        print( "⚠️ No training data found. Generating from supplier data..." )
        df = generate_training_data_from_suppliers()

        # Save for future use
        os.makedirs( os.path.dirname( TRAINING_DATA ), exist_ok=True )
        df.to_csv( TRAINING_DATA, index=False )
        print( f"💾 Saved training data to {TRAINING_DATA}" )

        return df


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_data_distribution ( df: pd.DataFrame ) :
    """Plot distribution of training data."""
    fig, axes = plt.subplots( 2, 2, figsize=(15, 10) )

    # Relevance score distribution
    axes[0, 0].hist( df['relevance'], bins=30, color='skyblue', edgecolor='black' )
    axes[0, 0].set_xlabel( 'Relevance Score' )
    axes[0, 0].set_ylabel( 'Frequency' )
    axes[0, 0].set_title( 'Distribution of Relevance Scores' )
    axes[0, 0].axvline( df['relevance'].mean(), color='red', linestyle='--',
                        label=f'Mean: {df["relevance"].mean():.2f}' )
    axes[0, 0].legend()

    # Suppliers per query
    query_counts = df.groupby( 'query_id' ).size()
    axes[0, 1].hist( query_counts, bins=20, color='lightcoral', edgecolor='black' )
    axes[0, 1].set_xlabel( 'Suppliers per Query' )
    axes[0, 1].set_ylabel( 'Frequency' )
    axes[0, 1].set_title( 'Distribution of Suppliers per Query' )
    axes[0, 1].axvline( query_counts.mean(), color='red', linestyle='--', label=f'Mean: {query_counts.mean():.1f}' )
    axes[0, 1].legend()

    # Feature correlations with relevance
    feature_cols = ['price_match', 'location_match', 'cert_match', 'years_normalized',
                    'is_manufacturer', 'faiss_score']
    correlations = df[feature_cols + ['relevance']].corr()['relevance'][:-1].sort_values()
    axes[1, 0].barh( range( len( correlations ) ), correlations.values, color='lightgreen', edgecolor='black' )
    axes[1, 0].set_yticks( range( len( correlations ) ) )
    axes[1, 0].set_yticklabels( correlations.index )
    axes[1, 0].set_xlabel( 'Correlation with Relevance' )
    axes[1, 0].set_title( 'Feature Importance (Correlation)' )
    axes[1, 0].axvline( 0, color='black', linestyle='-', linewidth=0.5 )

    # Price ratio distribution
    axes[1, 1].hist( df['price_ratio'], bins=30, color='plum', edgecolor='black' )
    axes[1, 1].set_xlabel( 'Price Ratio (Supplier Price / Max Price)' )
    axes[1, 1].set_ylabel( 'Frequency' )
    axes[1, 1].set_title( 'Distribution of Price Ratios' )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/01_data_distribution.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/01_data_distribution.png" )
    plt.close()


def plot_feature_importance_comparison ( lgbm_model, xgb_model, feature_names ) :
    """Compare feature importance between models."""
    fig, axes = plt.subplots( 1, 2, figsize=(16, 6) )

    # LightGBM
    if lgbm_model :
        lgbm_importance = pd.DataFrame( {
            'feature' : feature_names,
            'importance' : lgbm_model.feature_importances_
        } ).sort_values( 'importance', ascending=True )

        axes[0].barh( lgbm_importance['feature'], lgbm_importance['importance'], color='lightblue', edgecolor='black' )
        axes[0].set_xlabel( 'Importance' )
        axes[0].set_title( 'LightGBM Feature Importance' )
        axes[0].grid( axis='x', alpha=0.3 )

    # XGBoost
    if xgb_model :
        xgb_importance = pd.DataFrame( {
            'feature' : feature_names,
            'importance' : xgb_model.feature_importances_
        } ).sort_values( 'importance', ascending=True )

        axes[1].barh( xgb_importance['feature'], xgb_importance['importance'], color='lightcoral', edgecolor='black' )
        axes[1].set_xlabel( 'Importance' )
        axes[1].set_title( 'XGBoost Feature Importance' )
        axes[1].grid( axis='x', alpha=0.3 )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/02_feature_importance.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/02_feature_importance.png" )
    plt.close()


def plot_predictions_vs_actual ( models_dict, X_test, y_test, query_test ) :
    """Plot predicted vs actual relevance scores."""
    n_models = len( models_dict )
    fig, axes = plt.subplots( 1, n_models, figsize=(8 * n_models, 6) )

    if n_models == 1 :
        axes = [axes]

    for idx, (name, model) in enumerate( models_dict.items() ) :
        y_pred = model.predict( X_test )

        # Scatter plot
        axes[idx].scatter( y_test, y_pred, alpha=0.5, s=20, color='steelblue' )
        axes[idx].plot( [y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                        'r--', lw=2, label='Perfect Prediction' )
        axes[idx].set_xlabel( 'Actual Relevance' )
        axes[idx].set_ylabel( 'Predicted Relevance' )
        axes[idx].set_title( f'{name}: Predicted vs Actual' )
        axes[idx].legend()
        axes[idx].grid( alpha=0.3 )

        # Add R² score
        from sklearn.metrics import r2_score
        r2 = r2_score( y_test, y_pred )
        axes[idx].text( 0.05, 0.95, f'R² = {r2:.3f}',
                        transform=axes[idx].transAxes,
                        verticalalignment='top',
                        bbox=dict( boxstyle='round', facecolor='wheat', alpha=0.5 ) )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/03_predictions_vs_actual.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/03_predictions_vs_actual.png" )
    plt.close()


def plot_residuals ( models_dict, X_test, y_test ) :
    """Plot residual distribution for each model."""
    n_models = len( models_dict )
    fig, axes = plt.subplots( 1, n_models, figsize=(8 * n_models, 6) )

    if n_models == 1 :
        axes = [axes]

    for idx, (name, model) in enumerate( models_dict.items() ) :
        y_pred = model.predict( X_test )
        residuals = y_test - y_pred

        # Histogram
        axes[idx].hist( residuals, bins=30, color='lightgreen', edgecolor='black', alpha=0.7 )
        axes[idx].axvline( 0, color='red', linestyle='--', linewidth=2 )
        axes[idx].set_xlabel( 'Residuals (Actual - Predicted)' )
        axes[idx].set_ylabel( 'Frequency' )
        axes[idx].set_title( f'{name}: Residual Distribution' )
        axes[idx].grid( alpha=0.3 )

        # Add statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        axes[idx].text( 0.05, 0.95, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}',
                        transform=axes[idx].transAxes,
                        verticalalignment='top',
                        bbox=dict( boxstyle='round', facecolor='wheat', alpha=0.5 ) )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/04_residuals.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/04_residuals.png" )
    plt.close()


def plot_model_comparison ( results: Dict ) :
    """Compare model performance metrics."""
    if not results :
        return

    comparison_df = pd.DataFrame( results ).T

    fig, axes = plt.subplots( 1, 3, figsize=(18, 5) )

    # MSE comparison
    axes[0].bar( comparison_df.index, comparison_df['test_mse'], color=['lightblue', 'lightcoral'], edgecolor='black' )
    axes[0].set_ylabel( 'Mean Squared Error' )
    axes[0].set_title( 'Test MSE Comparison (Lower is Better)' )
    axes[0].grid( axis='y', alpha=0.3 )

    # MAE comparison
    axes[1].bar( comparison_df.index, comparison_df['test_mae'], color=['lightgreen', 'lightyellow'],
                 edgecolor='black' )
    axes[1].set_ylabel( 'Mean Absolute Error' )
    axes[1].set_title( 'Test MAE Comparison (Lower is Better)' )
    axes[1].grid( axis='y', alpha=0.3 )

    # NDCG comparison
    axes[2].bar( comparison_df.index, comparison_df['test_ndcg'], color=['plum', 'peachpuff'], edgecolor='black' )
    axes[2].set_ylabel( 'NDCG@10' )
    axes[2].set_title( 'Test NDCG Comparison (Higher is Better)' )
    axes[2].grid( axis='y', alpha=0.3 )
    axes[2].set_ylim( [0, 1] )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/05_model_comparison.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/05_model_comparison.png" )
    plt.close()


def plot_learning_curves ( lgbm_model, xgb_model ) :
    """Plot learning curves from training history."""
    fig, axes = plt.subplots( 1, 2, figsize=(16, 6) )

    # LightGBM learning curve
    if lgbm_model and hasattr( lgbm_model, 'evals_result_' ) :
        evals_result = lgbm_model.evals_result_
        if 'valid_0' in evals_result and 'ndcg' in evals_result['valid_0'] :
            train_metric = evals_result.get( 'training', {} ).get( 'ndcg', [] )
            valid_metric = evals_result['valid_0']['ndcg']

            epochs = range( 1, len( valid_metric ) + 1 )
            axes[0].plot( epochs, valid_metric, 'b-', label='Validation NDCG', linewidth=2 )
            if train_metric :
                axes[0].plot( epochs, train_metric, 'r--', label='Training NDCG', linewidth=2 )
            axes[0].set_xlabel( 'Iteration' )
            axes[0].set_ylabel( 'NDCG' )
            axes[0].set_title( 'LightGBM Learning Curve' )
            axes[0].legend()
            axes[0].grid( alpha=0.3 )

    # XGBoost learning curve
    if xgb_model and hasattr( xgb_model, 'evals_result' ) :
        evals_result = xgb_model.evals_result()
        if 'validation_0' in evals_result :
            metric_name = list( evals_result['validation_0'].keys() )[0]
            valid_metric = evals_result['validation_0'][metric_name]

            epochs = range( 1, len( valid_metric ) + 1 )
            axes[1].plot( epochs, valid_metric, 'g-', label=f'Validation {metric_name}', linewidth=2 )
            axes[1].set_xlabel( 'Iteration' )
            axes[1].set_ylabel( metric_name )
            axes[1].set_title( 'XGBoost Learning Curve' )
            axes[1].legend()
            axes[1].grid( alpha=0.3 )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/06_learning_curves.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/06_learning_curves.png" )
    plt.close()


def plot_ranking_quality ( models_dict, X_test, y_test, query_test ) :
    """Plot ranking quality by query."""
    fig, axes = plt.subplots( len( models_dict ), 1, figsize=(14, 6 * len( models_dict )) )

    if len( models_dict ) == 1 :
        axes = [axes]

    for idx, (name, model) in enumerate( models_dict.items() ) :
        y_pred = model.predict( X_test )

        # Calculate NDCG per query
        ndcg_per_query = []
        query_ids = []

        for qid in query_test.unique() :
            mask = query_test == qid
            if mask.sum() > 1 :
                true = y_test[mask].values.reshape( 1, -1 )
                pred = y_pred[mask].reshape( 1, -1 )
                ndcg_per_query.append( ndcg_score( true, pred ) )
                query_ids.append( qid )

        # Plot
        axes[idx].bar( range( len( ndcg_per_query ) ), sorted( ndcg_per_query, reverse=True ),
                       color='steelblue', edgecolor='black', alpha=0.7 )
        axes[idx].axhline( np.mean( ndcg_per_query ), color='red', linestyle='--',
                           linewidth=2, label=f'Mean NDCG: {np.mean( ndcg_per_query ):.3f}' )
        axes[idx].set_xlabel( 'Query (sorted by NDCG)' )
        axes[idx].set_ylabel( 'NDCG@10' )
        axes[idx].set_title( f'{name}: Ranking Quality per Query' )
        axes[idx].legend()
        axes[idx].grid( axis='y', alpha=0.3 )
        axes[idx].set_ylim( [0, 1] )

    plt.tight_layout()
    plt.savefig( f"{PLOTS_DIR}/07_ranking_quality.png", dpi=150, bbox_inches='tight' )
    print( f"✅ Saved: {PLOTS_DIR}/07_ranking_quality.png" )
    plt.close()


def generate_training_report ( df, results, feature_names ) :
    """Generate a summary report."""
    report_path = f"{PLOTS_DIR}/training_report.txt"

    with open( report_path, 'w' ) as f :
        f.write( "=" * 70 + "\n" )
        f.write( "SourceUP Supplier Ranking - Training Report\n" )
        f.write( f"Generated: {datetime.now().strftime( '%Y-%m-%d %H:%M:%S' )}\n" )
        f.write( "=" * 70 + "\n\n" )

        # Dataset info
        f.write( "DATASET INFORMATION\n" )
        f.write( "-" * 70 + "\n" )
        f.write( f"Total samples: {len( df )}\n" )
        f.write( f"Number of queries: {df['query_id'].nunique()}\n" )
        f.write( f"Average suppliers per query: {len( df ) / df['query_id'].nunique():.1f}\n" )
        f.write( f"Relevance score range: [{df['relevance'].min():.2f}, {df['relevance'].max():.2f}]\n" )
        f.write( f"Mean relevance: {df['relevance'].mean():.3f}\n" )
        f.write( f"Std relevance: {df['relevance'].std():.3f}\n\n" )

        # Feature statistics
        f.write( "FEATURE STATISTICS\n" )
        f.write( "-" * 70 + "\n" )
        for feat in feature_names :
            f.write( f"{feat:25s}: mean={df[feat].mean():.3f}, std={df[feat].std():.3f}\n" )
        f.write( "\n" )

        # Model performance
        f.write( "MODEL PERFORMANCE\n" )
        f.write( "-" * 70 + "\n" )
        for model_name, metrics in results.items() :
            f.write( f"\n{model_name}:\n" )
            f.write( f"  Test MSE:    {metrics['test_mse']:.4f}\n" )
            f.write( f"  Test MAE:    {metrics['test_mae']:.4f}\n" )
            f.write( f"  Test NDCG:   {metrics['test_ndcg']:.4f}\n" )

        if results :
            best_model = max(results.items(), key=lambda x: x[1]['test_ndcg'])
            f.write( f"\nBest Model: {best_model[0]} (lowest MSE)\n" )

        f.write( "\n" + "=" * 70 + "\n" )
        f.write( "All plots saved in: " + PLOTS_DIR + "\n" )
        f.write( "=" * 70 + "\n" )

    print( f"✅ Saved: {report_path}" )


# ============================================================================
# MODEL TRAINING
# ============================================================================

def prepare_data(df: pd.DataFrame) -> Tuple:
    """Prepare data for ranking models (LightGBM & XGBoost) with aligned indices."""

    feature_cols = [
        'price_match', 'price_ratio', 'price_distance',
        'location_match', 'cert_match', 'years_normalized',
        'is_manufacturer', 'is_trading_company',
        'faiss_score', 'faiss_rank'
    ]

    # Ensure relevance labels are integer 0-5
    df = df.copy()
    df['relevance'] = df['relevance'].round().clip(0, 5).astype(np.int32)

    X = df[feature_cols]
    y = df['relevance']
    query_ids = df['query_id']

    # Split by queries (no leakage)
    unique_queries = query_ids.unique()
    train_queries, test_queries = train_test_split(
        unique_queries,
        test_size=0.2,
        random_state=42
    )

    train_mask = query_ids.isin(train_queries)
    test_mask = query_ids.isin(test_queries)

    # 🔥 RESET INDICES to align y, X, and query_ids
    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)
    query_train = query_ids[train_mask].reset_index(drop=True)

    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask].reset_index(drop=True)
    query_test = query_ids[test_mask].reset_index(drop=True)

    # Group sizes for LightGBM
    train_group = query_train.value_counts().sort_index().values
    test_group = query_test.value_counts().sort_index().values

    # Factorize queries for XGBoost qid
    train_qid = pd.factorize(query_train)[0]
    test_qid = pd.factorize(query_test)[0]

    return (
        X_train, y_train, train_group, query_train, train_qid,
        X_test, y_test, test_group, query_test, test_qid,
        feature_cols
    )


def train_lightgbm(X_train, y_train, train_group,
                   X_test, y_test, test_group):
    """Train LightGBM LambdaRank properly."""

    print("\n" + "=" * 60)
    print("🌳 Training LightGBM Ranker")
    print("=" * 60)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1, 2, 3, 4, 5],  # 🔥 REQUIRED
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        group=train_group,
        eval_set=[(X_test, y_test)],
        eval_group=[test_group],
        eval_metric="ndcg@10",
        callbacks=[
            lgb.early_stopping(15, verbose=False),
            lgb.log_evaluation(20)
        ]
    )

    return model



def train_xgboost(X_train, y_train, train_group,
                  X_test, y_test, test_group):
    """Train XGBoost Ranker properly."""

    print("\n" + "=" * 60)
    print("🚀 Training XGBoost Ranker")
    print("=" * 60)

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42
    )

    # 🔥 Correct qid construction
    train_qid = np.repeat(
        np.arange(len(train_group)), train_group
    )
    test_qid = np.repeat(
        np.arange(len(test_group)), test_group
    )

    model.fit(
        X_train, y_train,
        qid=train_qid,
        eval_set=[(X_test, y_test)],
        eval_qid=[test_qid],
        verbose=20
    )

    return model




def evaluate_model ( model, X_train, y_train, query_train, X_test, y_test, query_test, model_name: str ) :
    """Evaluate model performance."""
    print( f"\n📊 {model_name} Performance:" )
    print( "-" * 40 )

    # Predictions
    y_pred_train = model.predict( X_train )
    y_pred_test = model.predict( X_test )

    # MSE and MAE
    train_mse = mean_squared_error( y_train, y_pred_train )
    test_mse = mean_squared_error( y_test, y_pred_test )
    train_mae = mean_absolute_error( y_train, y_pred_train )
    test_mae = mean_absolute_error( y_test, y_pred_test )

    print( f"Train MSE: {train_mse:.4f} | MAE: {train_mae:.4f}" )
    print( f"Test MSE:  {test_mse:.4f}  | MAE: {test_mae:.4f}" )

    # NDCG per query
    train_ndcg = calculate_ndcg( y_train, y_pred_train, query_train )
    test_ndcg = calculate_ndcg( y_test, y_pred_test, query_test )

    print( f"Train NDCG@10: {train_ndcg:.4f}" )
    print( f"Test NDCG@10:  {test_ndcg:.4f}" )

    # Feature importance
    if hasattr( model, 'feature_importances_' ) :
        print( "\n🎯 Top 5 Features:" )

        feature_names = [
            'price_match', 'price_ratio', 'price_distance',
            'location_match', 'cert_match', 'years_normalized',
            'is_manufacturer', 'is_trading_company',
            'faiss_score', 'faiss_rank'
        ]

        if model_name == "LightGBM" :
            importance = model.feature_importances_
            importance_df = pd.DataFrame( {
                'feature' : feature_names,
                'importance' : importance
            } )

        elif model_name == "XGBoost" :
            booster = model.get_booster()
            score = booster.get_score( importance_type='gain' )

            importance_df = pd.DataFrame( [
                (feature_names[int( k[1 :] )], v)
                for k, v in score.items()
            ], columns=['feature', 'importance'] )

        importance_df = importance_df.sort_values( 'importance', ascending=False )

        for _, row in importance_df.head( 5 ).iterrows() :
            print( f"   {row['feature']:20s}: {row['importance']:.4f}" )

    return {
        'test_mse' : test_mse,
        'test_mae' : test_mae,
        'test_ndcg' : test_ndcg
    }


def calculate_ndcg ( y_true, y_pred, query_ids ) :
    """Calculate average NDCG across queries."""
    ndcg_scores = []
    for qid in query_ids.unique() :
        mask = query_ids == qid
        if mask.sum() > 1 :
            true = y_true[mask].values.reshape( 1, -1 )
            pred = y_pred[mask].reshape( 1, -1 )
            ndcg_scores.append( ndcg_score( true, pred ) )

    return np.mean( ndcg_scores ) if ndcg_scores else 0.0


def save_model ( model, path: str, model_name: str ) :
    """Save trained model."""
    os.makedirs( os.path.dirname( path ), exist_ok=True )
    with open( path, 'wb' ) as f :
        pickle.dump( model, f )
    print( f"✅ {model_name} saved to: {path}" )


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline with visualization (LightGBM & XGBoost)."""
    print("=" * 70)
    print("🤖 SourceUP Supplier Ranking - Model Training with Visualization")
    print("=" * 70)

    # ✅ REQUIRED: initialize containers
    results = {}
    models = {}

    # ------------------------------------------------------------------
    # Load or generate training data
    # ------------------------------------------------------------------
    df = load_or_generate_data()
    print(f"\n📊 Dataset: {len(df)} samples, {df['query_id'].nunique()} queries")

    # Plot data distribution
    print("\n📈 Generating data distribution plots...")
    plot_data_distribution(df)

    # ------------------------------------------------------------------
    # Prepare data for ranking
    # ------------------------------------------------------------------
    (
        X_train, y_train, train_group, query_train, train_qid,
        X_test, y_test, test_group, query_test, test_qid,
        feature_cols
    ) = prepare_data(df)

    print(f"📈 Training: {len(X_train)} samples, {query_train.nunique()} queries")
    print(f"📉 Testing:  {len(X_test)} samples, {query_test.nunique()} queries")

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    if LGBM_AVAILABLE:
        lgbm_model = train_lightgbm(
            X_train, y_train, train_group,
            X_test, y_test, test_group
        )

        results['LightGBM'] = evaluate_model(
            lgbm_model,
            X_train, y_train, query_train,
            X_test, y_test, query_test,
            'LightGBM'
        )
        models['LightGBM'] = lgbm_model

    # ------------------------------------------------------------------
    # Train XGBoost
    # ------------------------------------------------------------------
    if XGB_AVAILABLE:
        xgb_model = train_xgboost(
            X_train, y_train, train_group,
            X_test, y_test, test_group
        )

        results['XGBoost'] = evaluate_model(
            xgb_model,
            X_train, y_train, query_train,
            X_test, y_test, query_test,
            'XGBoost'
        )
        models['XGBoost'] = xgb_model

    # ------------------------------------------------------------------
    # Visualizations & Reporting
    # ------------------------------------------------------------------
    if results:
        print("\n📊 Generating visualization plots...")

        plot_feature_importance_comparison(
            models.get('LightGBM'),
            models.get('XGBoost'),
            feature_cols
        )
        plot_predictions_vs_actual(models, X_test, y_test, query_test)
        plot_residuals(models, X_test, y_test)
        plot_model_comparison(results)
        plot_learning_curves(
            models.get('LightGBM'),
            models.get('XGBoost')
        )
        plot_ranking_quality(models, X_test, y_test, query_test)

        generate_training_report(df, results, feature_cols)

        # Save models
        if 'LightGBM' in models:
            save_model(models['LightGBM'], LGBM_MODEL_PATH, 'LightGBM')
        if 'XGBoost' in models:
            save_model(models['XGBoost'], XGB_MODEL_PATH, 'XGBoost')

        # Final comparison
        print("\n🏆 Model Comparison")
        print("=" * 70)
        comparison_df = pd.DataFrame(results).T
        print(comparison_df.to_string())

        best_model = max(results.items(), key=lambda x: x[1]['test_ndcg'])
        print(
            f"\n🏆 Best Model: {best_model[0]} "
            f"(Test NDCG@10: {best_model[1]['test_ndcg']:.4f})"
        )

        print("\n✅ Training Complete!")
        print(f"\n📊 All plots saved in: {PLOTS_DIR}")

    else:
        print("\n❌ No models were trained. Ensure LightGBM or XGBoost is installed.")



if __name__ == "__main__":
    main()
