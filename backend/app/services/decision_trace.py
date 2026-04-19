"""
Decision Trace Module
---------------------
Generates transparent, auditable explanations for ranking decisions.

This is a CORE DIFFERENTIATOR - provides full transparency into WHY
a supplier was ranked where it was, breaking down all contributing factors.
"""

from typing import Dict, List, Optional
import pandas as pd


class DecisionTrace :
    """
    Creates detailed audit trails for supplier rankings.

    Breaks down:
    - Semantic similarity contribution
    - Price contribution
    - Location contribution
    - Certification contribution
    - Experience contribution
    - Constraint penalties
    - Final weighted score
    """

    def __init__ ( self ) :
        self.traces = {}

    def generate_trace (
            self,
            supplier: Dict,
            query: Dict,
            features: pd.Series,
            final_score: float,
            constraint_results: Optional[Dict] = None
    ) -> Dict :
        """
        Generate a complete decision trace for one supplier.

        Args:
            supplier: Supplier data
            query: Query parameters
            features: Extracted features from ranker
            final_score: Final ranking score
            constraint_results: Results from constraint engine

        Returns:
            Detailed trace dictionary
        """
        trace = {
            'supplier_name' : self._get_supplier_name( supplier ),
            'final_score' : round( final_score, 4 ),
            'contributions' : {},
            'constraints' : {},
            'summary' : []
        }

        # ====================================================================
        # CONTRIBUTION 1: Semantic Similarity
        # ====================================================================
        faiss_score = features.get( 'faiss_score', 0 )
        semantic_contribution = faiss_score * 0.05  # 5% weight in ranker

        trace['contributions']['semantic_match'] = {
            'raw_score' : round( float( faiss_score ), 4 ),
            'weight' : 0.05,
            'contribution' : round( semantic_contribution, 4 ),
            'explanation' : f"Product matches '{query.get( 'product', 'query' )}' with {faiss_score:.2f} similarity"
        }

        # ====================================================================
        # CONTRIBUTION 2: Price Match
        # ====================================================================
        price_match = features.get( 'price_match', 0 )
        price_contribution = price_match * 0.35  # 35% weight

        supplier_price = features.get( 'supplier_price', 0 )
        max_price = query.get( 'max_price' )

        if max_price :
            price_explanation = (
                f"Price ${supplier_price:.2f} is {'within' if price_match else 'outside'} "
                f"budget of ${max_price:.2f}"
            )
        else :
            price_explanation = "No price constraint specified"

        trace['contributions']['price'] = {
            'raw_score' : round( float( price_match ), 4 ),
            'weight' : 0.35,
            'contribution' : round( price_contribution, 4 ),
            'supplier_price' : supplier_price,
            'max_price' : max_price,
            'explanation' : price_explanation
        }

        # ====================================================================
        # CONTRIBUTION 3: Price Competitiveness
        # ====================================================================
        price_distance = features.get( 'price_distance', 0 )
        price_comp_score = 1 - price_distance
        price_comp_contribution = price_comp_score * 0.10  # 10% weight

        trace['contributions']['price_competitiveness'] = {
            'raw_score' : round( float( price_comp_score ), 4 ),
            'weight' : 0.10,
            'contribution' : round( price_comp_contribution, 4 ),
            'explanation' : f"Price is {(1 - price_distance) * 100:.0f}% competitive"
        }

        # ====================================================================
        # CONTRIBUTION 4: Location Match
        # ====================================================================
        location_match = features.get( 'location_match', 0 )
        location_contribution = location_match * 0.20  # 20% weight

        supplier_location = supplier.get( 'supplier location', 'Unknown' )
        query_location = query.get( 'location', '' )

        if query_location :
            if location_match == 1.0 :
                location_explanation = f"Exact match: {supplier_location}"
            elif location_match > 0 :
                location_explanation = f"Partial match: {supplier_location} near {query_location}"
            else :
                location_explanation = f"No match: {supplier_location} ≠ {query_location}"
        else :
            location_explanation = f"Located in {supplier_location}"

        trace['contributions']['location'] = {
            'raw_score' : round( float( location_match ), 4 ),
            'weight' : 0.20,
            'contribution' : round( location_contribution, 4 ),
            'supplier_location' : supplier_location,
            'preferred_location' : query_location,
            'explanation' : location_explanation
        }

        # ====================================================================
        # CONTRIBUTION 5: Certification Match
        # ====================================================================
        cert_match = features.get( 'cert_match', 0 )
        cert_contribution = cert_match * 0.20  # 20% weight

        query_cert = query.get( 'certification', '' )
        supplier_certs = supplier.get( 'certifications', 'None listed' )

        if query_cert :
            cert_explanation = (
                f"{'Has' if cert_match else 'Missing'} required {query_cert} certification"
            )
        else :
            cert_explanation = f"Certifications: {supplier_certs}"

        trace['contributions']['certification'] = {
            'raw_score' : round( float( cert_match ), 4 ),
            'weight' : 0.20,
            'contribution' : round( cert_contribution, 4 ),
            'required' : query_cert,
            'has' : supplier_certs,
            'explanation' : cert_explanation
        }

        # ====================================================================
        # CONTRIBUTION 6: Experience
        # ====================================================================
        years_normalized = features.get( 'years_normalized', 0 )
        experience_contribution = years_normalized * 0.05  # 5% weight

        years = supplier.get( 'years with gs', 0 )
        trace['contributions']['experience'] = {
            'raw_score' : round( float( years_normalized ), 4 ),
            'weight' : 0.05,
            'contribution' : round( experience_contribution, 4 ),
            'years' : years,
            'explanation' : f"{years} years on platform"
        }

        # ====================================================================
        # CONTRIBUTION 7: Business Type
        # ====================================================================
        is_manufacturer = features.get( 'is_manufacturer', 0 )
        business_contribution = is_manufacturer * 0.05  # 5% weight

        business_type = supplier.get( 'business type', 'Unknown' )
        trace['contributions']['business_type'] = {
            'raw_score' : round( float( is_manufacturer ), 4 ),
            'weight' : 0.05,
            'contribution' : round( business_contribution, 4 ),
            'type' : business_type,
            'explanation' : f"Business type: {business_type}"
        }

        # ====================================================================
        # CONSTRAINT EFFECTS
        # ====================================================================
        if constraint_results :
            trace['constraints'] = {
                'passed_all' : all( c.get( 'passed', True ) for c in constraint_results.values() ),
                'details' : constraint_results,
                'penalty' : 0.0  # No penalty if passed, otherwise filtered out
            }

        # ====================================================================
        # SUMMARY
        # ====================================================================
        trace['summary'] = self._generate_summary( trace )

        return trace

    def generate_comparative_trace (
            self,
            supplier_a: Dict,
            supplier_b: Dict,
            trace_a: Dict,
            trace_b: Dict
    ) -> Dict :
        """
        Compare two suppliers to explain ranking difference.

        Args:
            supplier_a: First supplier data
            supplier_b: Second supplier data
            trace_a: Decision trace for supplier A
            trace_b: Decision trace for supplier B

        Returns:
            Comparative analysis
        """
        comparison = {
            'supplier_a' : trace_a['supplier_name'],
            'supplier_b' : trace_b['supplier_name'],
            'score_difference' : round( trace_a['final_score'] - trace_b['final_score'], 4 ),
            'key_differences' : [],
            'winner' : trace_a['supplier_name'] if trace_a['final_score'] > trace_b['final_score'] else trace_b[
                'supplier_name']
        }

        # Compare each contribution
        for key in trace_a['contributions'].keys() :
            contrib_a = trace_a['contributions'][key]['contribution']
            contrib_b = trace_b['contributions'][key]['contribution']
            diff = contrib_a - contrib_b

            if abs( diff ) > 0.01 :  # Significant difference
                comparison['key_differences'].append( {
                    'factor' : key.replace( '_', ' ' ).title(),
                    'difference' : round( diff, 4 ),
                    'explanation' : self._explain_difference( key, contrib_a, contrib_b, trace_a, trace_b )
                } )

        # Sort by absolute difference
        comparison['key_differences'].sort(
            key=lambda x : abs( x['difference'] ),
            reverse=True
        )

        return comparison

    def _get_supplier_name ( self, supplier: Dict ) -> str :
        """Extract supplier name safely."""
        return (
                supplier.get( "supplier name" ) or
                supplier.get( "Supplier Name" ) or
                supplier.get( "company name" ) or
                supplier.get( "Company Name" ) or
                "Unknown Supplier"
        )

    def _generate_summary ( self, trace: Dict ) -> List[str] :
        """Generate human-readable summary of decision."""
        summary = []

        # Top 3 contributions
        contributions = sorted(
            trace['contributions'].items(),
            key=lambda x : x[1]['contribution'],
            reverse=True
        )[:3]

        for factor, data in contributions :
            if data['contribution'] > 0.01 :
                summary.append(
                    f"✓ {factor.replace( '_', ' ' ).title()}: "
                    f"+{data['contribution']:.3f} ({data['explanation']})"
                )

        return summary

    def _explain_difference (
            self,
            factor: str,
            contrib_a: float,
            contrib_b: float,
            trace_a: Dict,
            trace_b: Dict
    ) -> str :
        """Explain why one supplier scored better on a factor."""
        data_a = trace_a['contributions'][factor]
        data_b = trace_b['contributions'][factor]

        if factor == 'price' :
            return (
                f"Supplier A: ${data_a.get( 'supplier_price', 0 ):.2f} vs "
                f"Supplier B: ${data_b.get( 'supplier_price', 0 ):.2f}"
            )
        elif factor == 'location' :
            return (
                f"Supplier A: {data_a.get( 'supplier_location', 'Unknown' )} vs "
                f"Supplier B: {data_b.get( 'supplier_location', 'Unknown' )}"
            )
        elif factor == 'experience' :
            return (
                f"Supplier A: {data_a.get( 'years', 0 )} years vs "
                f"Supplier B: {data_b.get( 'years', 0 )} years"
            )
        else :
            return f"Supplier A scored {data_a['raw_score']:.2f} vs Supplier B: {data_b['raw_score']:.2f}"


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_decision_trace = None


def get_decision_trace () -> DecisionTrace :
    """Get or create decision trace singleton."""
    global _decision_trace
    if _decision_trace is None :
        _decision_trace = DecisionTrace()
    return _decision_trace