"""
Best Strategy to Beat Rule-Based Baseline
Ensemble: LambdaRank + Rule-Based Weighted Combination
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def _find_project_root(marker: str = "config.py") -> Path:
    """Walk up from this file until the folder containing `marker` is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looked for {marker})")


sys.path.insert(0, str(_find_project_root()))
from config import cfg
from rule_baseline import score_rule_based as _canonical_rule_scorer

import lightgbm as lgb
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupShuffleSplit

FEATURE_COLS = [
    "price_match", "price_ratio",
    "location_match", "cert_match",
    "faiss_score",
    # NOTE: faiss_rank removed — measured correlation with relevance label
    # was 0.025 (near-zero) on the full training set, and it is a lossy,
    # redundant derivative of faiss_score (rank position vs. raw similarity
    # magnitude). See dataset_diagnostic.py Experiment 4.
    # NOTE: years_normalized, is_manufacturer, is_trading_company removed —
    # confirmed zero SHAP importance across two independent training runs
    # (near-constant values in current data). Re-add here if richer supplier
    # tenure/business-type data becomes available.
    # NOTE: price_distance removed — for price/max_price <= 2 (the vast
    # majority of rows) it equals abs(price_ratio - 1) exactly, a pure
    # deterministic transform of price_ratio. Keeping both caused the model
    # to split arbitrarily between two copies of the same signal, which is
    # why SHAP rank order for price features flipped between training runs.
]


def train_best_strategy () :
    """Train ensemble strategy: ML + Rule-based weighted combination."""
    print( "=" * 70 )
    print( "🎯 BEST STRATEGY: LambdaRank + Rule-Based Ensemble" )
    print( "=" * 70 )

    # Load data
    df = pd.read_csv( str( cfg.TRAINING_DATA ) )
    df.columns = [c.strip().lower().replace( ' ', '_' ) for c in df.columns]
    df['relevance'] = df['relevance'].round().clip( 0, 5 ).astype( int )

    # ── Drop raw location / tier columns — must never reach the model ───────
    cols_to_drop = [c for c in ["location", "tier", "supplier_name", "query_text"] if c in df.columns]
    if cols_to_drop:
        df = df.drop( columns=cols_to_drop )
        print( f"   ⚠️  Dropped non-feature columns: {cols_to_drop}" )

    print( f"\n📊 Loaded {len( df ):,} rows, {df['query_id'].nunique()} queries" )

    # Split
    splitter = GroupShuffleSplit( n_splits=1, test_size=0.2, random_state=42 )
    train_idx, val_idx = next( splitter.split( df, groups=df['query_id'] ) )

    df_train = df.iloc[train_idx].reset_index( drop=True )
    df_val = df.iloc[val_idx].reset_index( drop=True )

    print( f"   Train: {len( df_train )} rows ({df_train['query_id'].nunique()} queries)" )
    print( f"   Val: {len( df_val )} rows ({df_val['query_id'].nunique()} queries)" )

    # Train LambdaRank
    print( "\n🔧 Training LambdaRank..." )
    train_groups = df_train.groupby( 'query_id' ).size().values
    val_groups = df_val.groupby( 'query_id' ).size().values

    lgb_model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='ndcg',
        ndcg_eval_at=[10],
        num_leaves=63,
        learning_rate=0.03,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        reg_alpha=0.1,
        random_state=42,
        verbose=-1
    )

    lgb_model.fit(
        df_train[FEATURE_COLS].values.astype( np.float32 ),
        df_train['relevance'].values,
        group=train_groups,
        eval_set=[(df_val[FEATURE_COLS].values.astype( np.float32 ), df_val['relevance'].values)],
        eval_group=[val_groups],
        eval_metric='ndcg@10',
        callbacks=[lgb.early_stopping( 30, verbose=False )]
    )

    # Rule-based scores
    # FIX: previously a local formula that used price_distance, years_normalized,
    # and is_manufacturer — none of which are in this file's FEATURE_COLS (they
    # were removed from the 6-feature production set). That caused a KeyError /
    # silent 0.0 substitution at runtime, and even when the columns existed it
    # produced a different formula than every other script in the repo that
    # reports a "Rule-Based" number. Now delegates to the single canonical
    # scorer from rule_baseline.py, matching ablation.py, baselines.py, and
    # ranker.py's fallback path.
    rule_pred = _canonical_rule_scorer(df_val)
    ml_pred = lgb_model.predict( df_val[FEATURE_COLS].values.astype( np.float32 ) )

    best_ndcg = 0
    best_ml_weight = 0.65

    for ml_w in np.arange( 0.3, 0.95, 0.05 ) :
        r_w = 1 - ml_w
        ensemble_pred = r_w * rule_pred + ml_w * ml_pred

        ndcg_scores = []
        for qid in df_val['query_id'].unique() :
            mask = df_val['query_id'] == qid
            if mask.sum() > 1 :
                t = df_val.loc[mask, 'relevance'].values.reshape( 1, -1 )
                p = ensemble_pred[mask].reshape( 1, -1 )
                ndcg_scores.append( ndcg_score( t, p, k=10 ) )

        ndcg = np.mean( ndcg_scores ) if ndcg_scores else 0
        if ndcg > best_ndcg :
            best_ndcg = ndcg
            best_ml_weight = ml_w

    rule_baseline = ndcg_at_k( df_val['relevance'], rule_pred, df_val['query_id'], k=10 )

    print( f"\n{'=' * 50}" )
    print( f"Rule-Based Baseline NDCG@10: {rule_baseline:.4f}" )
    print( f"Optimal ML Weight: {best_ml_weight:.2f}" )
    print( f"Ensemble NDCG@10: {best_ndcg:.4f}" )
    print( f"Improvement: {best_ndcg - rule_baseline:+.4f}" )

    if best_ndcg > rule_baseline :
        print( f"\n✅ SUCCESS! Beat rule-based baseline!" )
    else :
        print( f"\n⚠️ Gap to beat: {rule_baseline - best_ndcg:.4f}" )

    # Save model
    result = {
        'lgb_model' : lgb_model.booster_,
        'ml_weight' : best_ml_weight,
        'rule_weight' : 1 - best_ml_weight
    }

    os.makedirs( str( cfg.MODELS_DIR ), exist_ok=True )
    with open( str( cfg.MODELS_DIR / 'best_strategy.pkl' ), 'wb' ) as f :
        pickle.dump( result, f )

    print( f"\n💾 Model saved: {cfg.MODELS_DIR / 'best_strategy.pkl'}" )

    return result, best_ndcg


def ndcg_at_k ( y_true, y_pred, query_ids, k=10 ) :
    scores = []
    for qid in query_ids.unique() :
        mask = query_ids == qid
        if mask.sum() < 2 :
            continue
        t = y_true[mask].values.reshape( 1, -1 )
        p = y_pred[mask].reshape( 1, -1 )
        scores.append( ndcg_score( t, p, k=k ) )
    return np.mean( scores ) if scores else 0.0


if __name__ == "__main__" :
    train_best_strategy()