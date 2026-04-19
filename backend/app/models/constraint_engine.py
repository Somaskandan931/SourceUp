"""
Constraint Engine - SME-Specific Filtering
-------------------------------------------
Enforces business constraints before ranking to ensure
only viable suppliers are recommended.

This is a COMMERCIAL NOVELTY - most platforms don't do pre-filtering
based on hard business constraints for SMEs.
"""

from typing import List, Dict, Optional
import numpy as np


class ConstraintEngine :
    """
    Filters suppliers based on hard business constraints.

    Constraints:
    - Budget limits (max_price)
    - MOQ thresholds (can SME afford the minimum order?)
    - Delivery urgency (lead_time requirements)
    - Risk tolerance (certification requirements)
    - Location preferences (domestic vs international)
    """

    def __init__ ( self ) :
        self.filters_applied = []
        self.filtered_count = 0

    def apply_constraints (
            self,
            suppliers: List[Dict],
            constraints: Dict
    ) -> List[Dict] :
        """
        Filter suppliers based on SME constraints.

        Args:
            suppliers: List of candidate suppliers
            constraints: Dictionary with constraint parameters

        Returns:
            Filtered list of viable suppliers with constraint metadata
        """
        if not suppliers :
            return []

        self.filters_applied = []
        self.filtered_count = 0
        viable_suppliers = []

        for supplier in suppliers :
            # Track why supplier passed/failed
            constraint_results = {}
            passes_all = True

            # ================================================================
            # CONSTRAINT 1: Budget Limit (Hard constraint)
            # ================================================================
            max_price = constraints.get( "max_price" )
            if max_price is not None :
                supplier_price = self._extract_price( supplier )

                if supplier_price > 0 :
                    within_budget = supplier_price <= max_price
                    constraint_results['budget'] = {
                        'passed' : within_budget,
                        'supplier_price' : supplier_price,
                        'max_allowed' : max_price,
                        'reason' : f"Price ${supplier_price:.2f} {'≤' if within_budget else '>'} ${max_price:.2f}"
                    }

                    if not within_budget :
                        passes_all = False
                        self.filtered_count += 1
                        if 'budget' not in self.filters_applied :
                            self.filters_applied.append( 'budget' )
                else :
                    # No price info - let it pass but flag it
                    constraint_results['budget'] = {
                        'passed' : True,
                        'reason' : 'No price information available'
                    }

            # ================================================================
            # CONSTRAINT 2: MOQ Affordability (SME-specific)
            # ================================================================
            moq_budget = constraints.get( "moq_budget" )  # Total budget for MOQ
            if moq_budget is not None :
                moq = supplier.get( "min order qty" ) or supplier.get( "Min Order Qty" )
                supplier_price = self._extract_price( supplier )

                if moq and supplier_price > 0 :
                    try :
                        moq_value = float( moq )
                        total_moq_cost = moq_value * supplier_price
                        can_afford = total_moq_cost <= moq_budget

                        constraint_results['moq_affordability'] = {
                            'passed' : can_afford,
                            'moq' : moq_value,
                            'total_cost' : total_moq_cost,
                            'budget' : moq_budget,
                            'reason' : f"MOQ cost ${total_moq_cost:.2f} {'≤' if can_afford else '>'} ${moq_budget:.2f}"
                        }

                        if not can_afford :
                            passes_all = False
                            self.filtered_count += 1
                            if 'moq_affordability' not in self.filters_applied :
                                self.filters_applied.append( 'moq_affordability' )
                    except (ValueError, TypeError) :
                        pass

            # ================================================================
            # CONSTRAINT 3: Delivery Urgency (Hard deadline)
            # ================================================================
            max_lead_time = constraints.get( "max_lead_time" )  # days
            if max_lead_time is not None :
                lead_time = supplier.get( "lead time" ) or supplier.get( "Lead Time" )

                if lead_time :
                    try :
                        lead_time_days = float( lead_time )
                        meets_deadline = lead_time_days <= max_lead_time

                        constraint_results['delivery_urgency'] = {
                            'passed' : meets_deadline,
                            'lead_time' : lead_time_days,
                            'max_allowed' : max_lead_time,
                            'reason' : f"{lead_time_days} days {'≤' if meets_deadline else '>'} {max_lead_time} days required"
                        }

                        if not meets_deadline :
                            passes_all = False
                            self.filtered_count += 1
                            if 'delivery_urgency' not in self.filters_applied :
                                self.filters_applied.append( 'delivery_urgency' )
                    except (ValueError, TypeError) :
                        pass

            # ================================================================
            # CONSTRAINT 4: Required Certifications (Risk tolerance)
            # ================================================================
            required_certs = constraints.get( "required_certifications", [] )
            if required_certs :
                supplier_certs = str( supplier.get( "certifications" ) or "" ).lower()

                has_required = all(
                    cert.lower() in supplier_certs
                    for cert in required_certs
                )

                constraint_results['certifications'] = {
                    'passed' : has_required,
                    'required' : required_certs,
                    'has' : supplier_certs,
                    'reason' : f"{'Has' if has_required else 'Missing'} required certifications"
                }

                if not has_required :
                    passes_all = False
                    self.filtered_count += 1
                    if 'certifications' not in self.filters_applied :
                        self.filters_applied.append( 'certifications' )

            # ================================================================
            # CONSTRAINT 5: Location Preference (Soft/Hard based on flag)
            # ================================================================
            preferred_location = constraints.get( "preferred_location" )
            location_mandatory = constraints.get( "location_mandatory", False )

            if preferred_location :
                supplier_location = str(
                    supplier.get( "supplier location" ) or
                    supplier.get( "Supplier Location" ) or ""
                ).lower()

                is_preferred = preferred_location.lower() in supplier_location

                constraint_results['location'] = {
                    'passed' : is_preferred or not location_mandatory,
                    'preferred' : preferred_location,
                    'actual' : supplier_location,
                    'mandatory' : location_mandatory,
                    'reason' : f"Location {'matches' if is_preferred else 'differs from'} preference"
                }

                if location_mandatory and not is_preferred :
                    passes_all = False
                    self.filtered_count += 1
                    if 'location' not in self.filters_applied :
                        self.filters_applied.append( 'location' )

            # ================================================================
            # CONSTRAINT 6: Minimum Experience (Platform trust)
            # ================================================================
            min_years = constraints.get( "min_years_experience" )
            if min_years is not None :
                years = supplier.get( "years with gs" ) or supplier.get( "Years with GS" ) or 0

                try :
                    years_value = float( years )
                    meets_experience = years_value >= min_years

                    constraint_results['experience'] = {
                        'passed' : meets_experience,
                        'years' : years_value,
                        'min_required' : min_years,
                        'reason' : f"{years_value} years {'≥' if meets_experience else '<'} {min_years} years required"
                    }

                    if not meets_experience :
                        passes_all = False
                        self.filtered_count += 1
                        if 'experience' not in self.filters_applied :
                            self.filters_applied.append( 'experience' )
                except (ValueError, TypeError) :
                    pass

            # ================================================================
            # Add constraint metadata to supplier
            # ================================================================
            if passes_all :
                supplier['constraint_results'] = constraint_results
                supplier['passes_constraints'] = True
                supplier['constraint_score'] = self._calculate_constraint_score( constraint_results )
                viable_suppliers.append( supplier )
            else :
                # Optionally keep failed suppliers for explanation
                supplier['constraint_results'] = constraint_results
                supplier['passes_constraints'] = False

        return viable_suppliers

    def _extract_price ( self, supplier: Dict ) -> float :
        """Extract numeric price from supplier data."""
        price = supplier.get( "price min" ) or supplier.get( "Price Min" )

        if price is None :
            # Try parsing the price field
            price_str = supplier.get( "price" ) or supplier.get( "Price" )
            if price_str :
                try :
                    if isinstance( price_str, str ) and '-' in price_str :
                        return float( price_str.split( '-' )[0].strip() )
                    return float( price_str )
                except (ValueError, TypeError) :
                    return 0.0
            return 0.0

        return float( price )

    def _calculate_constraint_score ( self, constraint_results: Dict ) -> float :
        """
        Calculate a 0-1 score based on how well supplier meets constraints.
        Used for soft preferences within viable suppliers.
        """
        if not constraint_results :
            return 1.0

        passed = sum( 1 for c in constraint_results.values() if c.get( 'passed', True ) )
        total = len( constraint_results )

        return passed / total if total > 0 else 1.0

    def get_filter_summary ( self ) -> Dict :
        """Get summary of filters applied."""
        return {
            'filters_applied' : self.filters_applied,
            'total_filtered' : self.filtered_count
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_constraint_engine = None


def get_constraint_engine () -> ConstraintEngine :
    """Get or create constraint engine singleton."""
    global _constraint_engine
    if _constraint_engine is None :
        _constraint_engine = ConstraintEngine()
    return _constraint_engine