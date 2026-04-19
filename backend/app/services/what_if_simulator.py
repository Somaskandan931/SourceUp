"""
What-If Simulator
-----------------
Enables interactive trade-off analysis by simulating ranking changes
when users adjust priorities or constraints.

This is a COMMERCIAL NOVELTY - allows SMEs to understand the impact
of their decisions BEFORE committing.
"""

from typing import Dict, List, Tuple
import pandas as pd
from backend.app.models.ranker import extract_features_batch


class WhatIfSimulator :
    """
    Simulates ranking changes based on priority adjustments.

    Scenarios:
    - "What if I prioritize price over location?"
    - "What if I increase my budget by 10%?"
    - "What if I relax certification requirements?"
    - "What if I accept longer lead times?"
    """

    def __init__ ( self ) :
        self.scenarios = []
        self.base_weights = {
            'price_match' : 0.35,
            'price_competitiveness' : 0.10,
            'location_match' : 0.20,
            'cert_match' : 0.20,
            'years_normalized' : 0.05,
            'is_manufacturer' : 0.05,
            'faiss_score' : 0.05
        }

    def simulate_priority_change (
            self,
            suppliers: List[Dict],
            query: Dict,
            priority_changes: Dict[str, float]
    ) -> Dict :
        """
        Simulate what happens when user changes priorities.

        Args:
            suppliers: List of suppliers to re-rank
            query: Original query
            priority_changes: Dict of factor -> new weight
                             e.g., {'price_match': 0.50, 'location_match': 0.10}

        Returns:
            Comparison between original and new rankings
        """
        # Extract features for all suppliers
        features_df = extract_features_batch( suppliers, query )

        # Calculate original scores
        original_scores = self._calculate_scores( features_df, self.base_weights )

        # Calculate new weights (normalize to sum to 1.0)
        new_weights = self.base_weights.copy()
        new_weights.update( priority_changes )

        # Normalize
        total = sum( new_weights.values() )
        new_weights = {k : v / total for k, v in new_weights.items()}

        # Calculate new scores
        new_scores = self._calculate_scores( features_df, new_weights )

        # Create ranking comparison
        return self._compare_rankings(
            suppliers,
            original_scores,
            new_scores,
            self.base_weights,
            new_weights,
            "Priority Adjustment"
        )

    def simulate_budget_change (
            self,
            suppliers: List[Dict],
            query: Dict,
            budget_multiplier: float
    ) -> Dict :
        """
        Simulate what happens when budget changes.

        Args:
            suppliers: List of suppliers
            query: Original query
            budget_multiplier: e.g., 1.10 for 10% increase, 0.90 for 10% decrease

        Returns:
            Impact analysis
        """
        original_budget = query.get( 'max_price' )

        if not original_budget :
            return {
                'error' : 'No budget constraint in original query',
                'suggestion' : 'Add max_price to query'
            }

        new_budget = original_budget * budget_multiplier
        new_query = query.copy()
        new_query['max_price'] = new_budget

        # Recalculate features with new budget
        original_features = extract_features_batch( suppliers, query )
        new_features = extract_features_batch( suppliers, new_query )

        # Score both
        original_scores = self._calculate_scores( original_features, self.base_weights )
        new_scores = self._calculate_scores( new_features, self.base_weights )

        # Analyze impact
        newly_affordable = []
        for i, supplier in enumerate( suppliers ) :
            orig_price_match = original_features.iloc[i]['price_match']
            new_price_match = new_features.iloc[i]['price_match']

            if orig_price_match == 0 and new_price_match == 1 :
                newly_affordable.append( {
                    'supplier' : self._get_supplier_name( supplier ),
                    'price' : supplier.get( 'price', 'N/A' ),
                    'original_rank' : self._get_rank( original_scores, i ),
                    'new_rank' : self._get_rank( new_scores, i )
                } )

        return {
            'original_budget' : original_budget,
            'new_budget' : new_budget,
            'change_percent' : (budget_multiplier - 1) * 100,
            'newly_affordable_count' : len( newly_affordable ),
            'newly_affordable' : newly_affordable,
            'ranking_comparison' : self._compare_rankings(
                suppliers,
                original_scores,
                new_scores,
                self.base_weights,
                self.base_weights,
                f"Budget {'increase' if budget_multiplier > 1 else 'decrease'}"
            )
        }

    def simulate_constraint_relaxation (
            self,
            suppliers: List[Dict],
            query: Dict,
            constraint_to_relax: str,
            relaxation_amount: float
    ) -> Dict :
        """
        Simulate relaxing a constraint.

        Args:
            suppliers: List of suppliers
            query: Original query
            constraint_to_relax: 'max_lead_time', 'max_price', 'min_years_experience'
            relaxation_amount: Amount to relax (e.g., +7 days, +$10, -1 year)

        Returns:
            Impact of relaxation
        """
        new_query = query.copy()

        if constraint_to_relax == 'max_lead_time' :
            original_value = query.get( 'max_lead_time', float( 'inf' ) )
            new_query['max_lead_time'] = original_value + relaxation_amount
        elif constraint_to_relax == 'max_price' :
            original_value = query.get( 'max_price', float( 'inf' ) )
            new_query['max_price'] = original_value + relaxation_amount
        elif constraint_to_relax == 'min_years_experience' :
            original_value = query.get( 'min_years_experience', 0 )
            new_query['min_years_experience'] = max( 0, original_value + relaxation_amount )
        else :
            return {'error' : f'Unknown constraint: {constraint_to_relax}'}

        # Recalculate
        original_features = extract_features_batch( suppliers, query )
        new_features = extract_features_batch( suppliers, new_query )

        original_scores = self._calculate_scores( original_features, self.base_weights )
        new_scores = self._calculate_scores( new_features, self.base_weights )

        return self._compare_rankings(
            suppliers,
            original_scores,
            new_scores,
            self.base_weights,
            self.base_weights,
            f"Relaxed {constraint_to_relax} by {relaxation_amount}"
        )

    def simulate_trade_off (
            self,
            suppliers: List[Dict],
            query: Dict,
            scenario: str
    ) -> Dict :
        """
        Simulate common trade-off scenarios.

        Args:
            suppliers: List of suppliers
            query: Original query
            scenario: One of:
                - 'price_over_speed': Maximize price savings, ignore lead time
                - 'speed_over_price': Minimize lead time, ignore price
                - 'quality_over_cost': Prioritize certifications and experience
                - 'local_over_cheap': Prioritize location over price

        Returns:
            Trade-off analysis
        """
        scenarios = {
            'price_over_speed' : {
                'price_match' : 0.50,
                'price_competitiveness' : 0.20,
                'location_match' : 0.10,
                'cert_match' : 0.10,
                'years_normalized' : 0.05,
                'is_manufacturer' : 0.05,
                'faiss_score' : 0.00
            },
            'speed_over_price' : {
                'price_match' : 0.10,
                'price_competitiveness' : 0.05,
                'location_match' : 0.40,  # Closer = faster
                'cert_match' : 0.15,
                'years_normalized' : 0.10,
                'is_manufacturer' : 0.10,
                'faiss_score' : 0.10
            },
            'quality_over_cost' : {
                'price_match' : 0.10,
                'price_competitiveness' : 0.05,
                'location_match' : 0.10,
                'cert_match' : 0.40,
                'years_normalized' : 0.20,
                'is_manufacturer' : 0.10,
                'faiss_score' : 0.05
            },
            'local_over_cheap' : {
                'price_match' : 0.10,
                'price_competitiveness' : 0.05,
                'location_match' : 0.50,
                'cert_match' : 0.15,
                'years_normalized' : 0.10,
                'is_manufacturer' : 0.05,
                'faiss_score' : 0.05
            }
        }

        if scenario not in scenarios :
            return {
                'error' : f'Unknown scenario: {scenario}',
                'available_scenarios' : list( scenarios.keys() )
            }

        return self.simulate_priority_change( suppliers, query, scenarios[scenario] )

    def _calculate_scores (
            self,
            features_df: pd.DataFrame,
            weights: Dict[str, float]
    ) -> List[float] :
        """Calculate weighted scores for suppliers."""
        scores = []

        for _, row in features_df.iterrows() :
            score = sum(
                row.get( feature, 0 ) * weight
                for feature, weight in weights.items()
            )
            scores.append( score )

        return scores

    def _compare_rankings (
            self,
            suppliers: List[Dict],
            original_scores: List[float],
            new_scores: List[float],
            original_weights: Dict,
            new_weights: Dict,
            scenario_name: str
    ) -> Dict :
        """Compare two rankings and identify changes."""
        # Create ranked lists
        original_ranking = sorted(
            enumerate( original_scores ),
            key=lambda x : x[1],
            reverse=True
        )
        new_ranking = sorted(
            enumerate( new_scores ),
            key=lambda x : x[1],
            reverse=True
        )

        # Track position changes
        changes = []
        for new_rank, (idx, new_score) in enumerate( new_ranking[:10], 1 ) :
            original_rank = next(
                (i for i, (j, _) in enumerate( original_ranking, 1 ) if j == idx),
                None
            )

            if original_rank != new_rank :
                changes.append( {
                    'supplier' : self._get_supplier_name( suppliers[idx] ),
                    'original_rank' : original_rank,
                    'new_rank' : new_rank,
                    'rank_change' : original_rank - new_rank if original_rank else None,
                    'original_score' : round( original_scores[idx], 4 ),
                    'new_score' : round( new_scores[idx], 4 ),
                    'score_change' : round( new_scores[idx] - original_scores[idx], 4 )
                } )

        # Identify biggest movers
        changes.sort( key=lambda x : abs( x['rank_change'] ) if x['rank_change'] else 0, reverse=True )

        return {
            'scenario' : scenario_name,
            'original_weights' : original_weights,
            'new_weights' : new_weights,
            'top_10_changes' : changes[:10],
            'new_top_supplier' : self._get_supplier_name( suppliers[new_ranking[0][0]] ),
            'original_top_supplier' : self._get_supplier_name( suppliers[original_ranking[0][0]] ),
            'total_rank_shifts' : len( [c for c in changes if c['rank_change'] != 0] )
        }

    def _get_supplier_name ( self, supplier: Dict ) -> str :
        """Extract supplier name safely."""
        return (
                supplier.get( "supplier name" ) or
                supplier.get( "Supplier Name" ) or
                supplier.get( "company name" ) or
                supplier.get( "Company Name" ) or
                "Unknown Supplier"
        )

    def _get_rank ( self, scores: List[float], index: int ) -> int :
        """Get rank of a supplier by index."""
        sorted_scores = sorted( enumerate( scores ), key=lambda x : x[1], reverse=True )
        return next( (i for i, (idx, _) in enumerate( sorted_scores, 1 ) if idx == index), None )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_what_if_simulator = None


def get_what_if_simulator () -> WhatIfSimulator :
    """Get or create what-if simulator singleton."""
    global _what_if_simulator
    if _what_if_simulator is None :
        _what_if_simulator = WhatIfSimulator()
    return _what_if_simulator