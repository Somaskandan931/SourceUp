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
import pandas as pd


class ConstraintEngine:
    """
    Filters suppliers based on hard business constraints.

    Constraints:
    - Budget limits (max_price)
    - MOQ thresholds (can SME afford the minimum order?)
    - Delivery urgency (lead_time requirements)
    - Risk tolerance (certification requirements)
    - Location preferences (domestic vs international)
    """

    def __init__(self):
        self.filters_applied = []
        self.filtered_count = 0
        self._use_vectorized = True  # Use vectorized operations by default

    def apply_constraints(
            self,
            suppliers: List[Dict],
            constraints: Dict
    ) -> List[Dict]:
        """
        Filter suppliers based on SME constraints.

        Uses vectorized operations for large datasets (>500 suppliers)
        and falls back to iterative for smaller datasets.

        Args:
            suppliers: List of candidate suppliers
            constraints: Dictionary with constraint parameters

        Returns:
            Filtered list of viable suppliers with constraint metadata
        """
        if not suppliers:
            return []

        self.filters_applied = []
        self.filtered_count = 0

        # Use vectorized filtering for larger datasets
        if len(suppliers) > 500 and self._use_vectorized:
            return self._apply_constraints_vectorized(suppliers, constraints)
        else:
            return self._apply_constraints_iterative(suppliers, constraints)

    def _apply_constraints_vectorized(
            self,
            suppliers: List[Dict],
            constraints: Dict
    ) -> List[Dict]:
        """
        Apply constraints using vectorized pandas operations.
        Much faster for large supplier lists (>500 items).
        """
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(suppliers)
        original_len = len(df)

        # Track which filters were applied
        active_filters = []

        # Create a mask for filtering (all True initially)
        mask = pd.Series([True] * len(df))

        # Store constraint results for each supplier (will be populated later)
        constraint_results = {i: {} for i in range(len(df))}

        # ================================================================
        # CONSTRAINT 1: Budget Limit (Hard constraint)
        # ================================================================
        max_price = constraints.get("max_price")
        if max_price is not None:
            # Extract prices vectorized
            prices = df.apply(
                lambda row: self._extract_price(row.to_dict()), axis=1
            )
            within_budget = (prices > 0) & (prices <= max_price)

            # Suppliers with no price info pass but are flagged
            no_price_info = (prices == 0)
            within_budget = within_budget | no_price_info

            mask = mask & within_budget
            active_filters.append('budget')

            # Store results for traceability
            for i, (price, passes) in enumerate(zip(prices, within_budget)):
                if max_price is not None:
                    constraint_results[i]['budget'] = {
                        'passed': bool(passes),
                        'supplier_price': float(price),
                        'max_allowed': max_price,
                        'reason': f"Price ${price:.2f} {'≤' if passes else '>'} ${max_price:.2f}" if price > 0 else "No price information available"
                    }

        # ================================================================
        # CONSTRAINT 2: MOQ Affordability (SME-specific)
        # ================================================================
        moq_budget = constraints.get("moq_budget")
        if moq_budget is not None:
            # Extract MOQ and prices vectorized
            moq_values = df.apply(
                lambda row: self._extract_moq(row.to_dict()), axis=1
            )
            prices = df.apply(
                lambda row: self._extract_price(row.to_dict()), axis=1
            )

            total_moq_cost = moq_values * prices
            can_afford = (moq_values > 0) & (prices > 0) & (total_moq_cost <= moq_budget)

            # Suppliers with missing data pass but are flagged
            missing_data = (moq_values == 0) | (prices == 0)
            can_afford = can_afford | missing_data

            mask = mask & can_afford
            active_filters.append('moq_affordability')

            # Store results
            for i, (moq, price, total, passes) in enumerate(zip(moq_values, prices, total_moq_cost, can_afford)):
                if moq_budget is not None and moq > 0 and price > 0:
                    constraint_results[i]['moq_affordability'] = {
                        'passed': bool(passes),
                        'moq': float(moq),
                        'total_cost': float(total),
                        'budget': moq_budget,
                        'reason': f"MOQ cost ${total:.2f} {'≤' if passes else '>'} ${moq_budget:.2f}"
                    }

        # ================================================================
        # CONSTRAINT 3: Delivery Urgency (Hard deadline)
        # ================================================================
        max_lead_time = constraints.get("max_lead_time")
        if max_lead_time is not None:
            lead_times = df.apply(
                lambda row: self._extract_lead_time(row.to_dict()), axis=1
            )
            meets_deadline = (lead_times > 0) & (lead_times <= max_lead_time)

            # Suppliers with missing data pass
            missing_data = (lead_times == 0)
            meets_deadline = meets_deadline | missing_data

            mask = mask & meets_deadline
            active_filters.append('delivery_urgency')

            # Store results
            for i, (lead_time, passes) in enumerate(zip(lead_times, meets_deadline)):
                if max_lead_time is not None and lead_time > 0:
                    constraint_results[i]['delivery_urgency'] = {
                        'passed': bool(passes),
                        'lead_time': float(lead_time),
                        'max_allowed': max_lead_time,
                        'reason': f"{lead_time} days {'≤' if passes else '>'} {max_lead_time} days required"
                    }

        # ================================================================
        # CONSTRAINT 4: Required Certifications (Risk tolerance)
        # ================================================================
        required_certs = constraints.get("required_certifications", [])
        if required_certs:
            cert_strings = df.get("certifications", "").fillna("").astype(str).str.lower()

            # Check each required certification
            for cert in required_certs:
                has_cert = cert_strings.str.contains(cert.lower(), na=False)
                mask = mask & has_cert

            active_filters.append('certifications')

            # Store results
            for i, (cert_str, passes) in enumerate(zip(cert_strings, mask)):
                has_required = all(cert.lower() in cert_str for cert in required_certs)
                constraint_results[i]['certifications'] = {
                    'passed': bool(passes),
                    'required': required_certs,
                    'has': cert_str,
                    'reason': f"{'Has' if has_required else 'Missing'} required certifications"
                }

        # ================================================================
        # CONSTRAINT 5: Location Preference (Soft/Hard based on flag)
        # ================================================================
        preferred_location = constraints.get("preferred_location")
        location_mandatory = constraints.get("location_mandatory", False)

        if preferred_location:
            locations = df.get("supplier location", "").fillna("").astype(str).str.lower()
            is_preferred = locations.str.contains(preferred_location.lower(), na=False)

            if location_mandatory:
                mask = mask & is_preferred
            else:
                # Soft constraint - doesn't filter, but we track it
                pass

            active_filters.append('location')

            # Store results
            for i, (loc, pref) in enumerate(zip(locations, is_preferred)):
                constraint_results[i]['location'] = {
                    'passed': bool(pref or not location_mandatory),
                    'preferred': preferred_location,
                    'actual': loc,
                    'mandatory': location_mandatory,
                    'reason': f"Location {'matches' if pref else 'differs from'} preference"
                }

        # ================================================================
        # CONSTRAINT 6: Minimum Experience (Platform trust)
        # ================================================================
        min_years = constraints.get("min_years_experience")
        if min_years is not None:
            years = df.get("years with gs", 0).fillna(0).astype(float)
            meets_experience = years >= min_years
            mask = mask & meets_experience
            active_filters.append('experience')

            # Store results
            for i, (years_val, passes) in enumerate(zip(years, meets_experience)):
                constraint_results[i]['experience'] = {
                    'passed': bool(passes),
                    'years': float(years_val),
                    'min_required': min_years,
                    'reason': f"{years_val} years {'≥' if passes else '<'} {min_years} years required"
                }

        # Mark violations — do NOT filter suppliers out
        self.filters_applied = active_filters
        self.filtered_count = int((~mask).sum())

        # Add constraint metadata back to all suppliers (including violators)
        all_suppliers = []
        for idx, (_, row) in enumerate(df.iterrows()):
            supplier = row.to_dict()
            original_idx = idx
            violated = not bool(mask.iloc[idx])

            if original_idx in constraint_results:
                supplier['constraint_results'] = constraint_results[original_idx]
                supplier['constraint_score'] = self._calculate_constraint_score(
                    constraint_results[original_idx]
                )
            else:
                supplier['constraint_results'] = {}
                supplier['constraint_score'] = 1.0

            supplier['constraint_violated'] = violated
            supplier['passes_constraints'] = not violated
            all_suppliers.append(supplier)

        return all_suppliers

    def _apply_constraints_iterative(
            self,
            suppliers: List[Dict],
            constraints: Dict
    ) -> List[Dict]:
        """
        Mark suppliers with constraint violations instead of removing them.
        Suppliers stay in the list so the ranker can apply a penalty score,
        keeping CVR meaningful (< 1.0) rather than always 1.0.
        """
        self.filters_applied = []
        self.filtered_count = 0
        all_suppliers = []

        for supplier in suppliers:
            constraint_results = {}
            constraint_violated = False  # Track overall violation flag

            # CONSTRAINT 1: Budget Limit
            max_price = constraints.get("max_price")
            if max_price is not None:
                supplier_price = self._extract_price(supplier)
                if supplier_price > 0:
                    within_budget = supplier_price <= max_price
                    constraint_results['budget'] = {
                        'passed': within_budget,
                        'supplier_price': supplier_price,
                        'max_allowed': max_price,
                        'reason': f"Price ${supplier_price:.2f} {'≤' if within_budget else '>'} ${max_price:.2f}"
                    }
                    if not within_budget:
                        constraint_violated = True
                        if 'budget' not in self.filters_applied:
                            self.filters_applied.append('budget')
                else:
                    constraint_results['budget'] = {
                        'passed': True,
                        'reason': 'No price information available'
                    }

            # CONSTRAINT 2: MOQ Affordability
            moq_budget = constraints.get("moq_budget")
            if moq_budget is not None:
                moq = supplier.get("min order qty") or supplier.get("Min Order Qty")
                supplier_price = self._extract_price(supplier)
                if moq and supplier_price > 0:
                    try:
                        moq_value = float(moq)
                        total_moq_cost = moq_value * supplier_price
                        can_afford = total_moq_cost <= moq_budget
                        constraint_results['moq_affordability'] = {
                            'passed': can_afford,
                            'moq': moq_value,
                            'total_cost': total_moq_cost,
                            'budget': moq_budget,
                            'reason': f"MOQ cost ${total_moq_cost:.2f} {'≤' if can_afford else '>'} ${moq_budget:.2f}"
                        }
                        if not can_afford:
                            constraint_violated = True
                            if 'moq_affordability' not in self.filters_applied:
                                self.filters_applied.append('moq_affordability')
                    except (ValueError, TypeError):
                        pass

            # CONSTRAINT 3: Delivery Urgency
            max_lead_time = constraints.get("max_lead_time")
            if max_lead_time is not None:
                lead_time = supplier.get("lead time") or supplier.get("Lead Time")
                if lead_time:
                    try:
                        lead_time_days = float(lead_time)
                        meets_deadline = lead_time_days <= max_lead_time
                        constraint_results['delivery_urgency'] = {
                            'passed': meets_deadline,
                            'lead_time': lead_time_days,
                            'max_allowed': max_lead_time,
                            'reason': f"{lead_time_days} days {'≤' if meets_deadline else '>'} {max_lead_time} days required"
                        }
                        if not meets_deadline:
                            constraint_violated = True
                            if 'delivery_urgency' not in self.filters_applied:
                                self.filters_applied.append('delivery_urgency')
                    except (ValueError, TypeError):
                        pass

            # CONSTRAINT 4: Required Certifications
            required_certs = constraints.get("required_certifications", [])
            if required_certs:
                supplier_certs = str(supplier.get("certifications") or "").lower()
                has_required = all(cert.lower() in supplier_certs for cert in required_certs)
                constraint_results['certifications'] = {
                    'passed': has_required,
                    'required': required_certs,
                    'has': supplier_certs,
                    'reason': f"{'Has' if has_required else 'Missing'} required certifications"
                }
                if not has_required:
                    constraint_violated = True
                    if 'certifications' not in self.filters_applied:
                        self.filters_applied.append('certifications')

            # CONSTRAINT 5: Location Preference
            preferred_location = constraints.get("preferred_location")
            location_mandatory = constraints.get("location_mandatory", False)
            if preferred_location:
                supplier_location = str(
                    supplier.get("supplier location") or
                    supplier.get("Supplier Location") or ""
                ).lower()
                is_preferred = preferred_location.lower() in supplier_location
                constraint_results['location'] = {
                    'passed': is_preferred or not location_mandatory,
                    'preferred': preferred_location,
                    'actual': supplier_location,
                    'mandatory': location_mandatory,
                    'reason': f"Location {'matches' if is_preferred else 'differs from'} preference"
                }
                if location_mandatory and not is_preferred:
                    constraint_violated = True
                    if 'location' not in self.filters_applied:
                        self.filters_applied.append('location')

            # CONSTRAINT 6: Minimum Experience
            min_years = constraints.get("min_years_experience")
            if min_years is not None:
                years = supplier.get("years with gs") or supplier.get("Years with GS") or 0
                try:
                    years_value = float(years)
                    meets_experience = years_value >= min_years
                    constraint_results['experience'] = {
                        'passed': meets_experience,
                        'years': years_value,
                        'min_required': min_years,
                        'reason': f"{years_value} years {'≥' if meets_experience else '<'} {min_years} years required"
                    }
                    if not meets_experience:
                        constraint_violated = True
                        if 'experience' not in self.filters_applied:
                            self.filters_applied.append('experience')
                except (ValueError, TypeError):
                    pass

            # MARK violation — do NOT remove the supplier
            supplier['constraint_results'] = constraint_results
            supplier['constraint_violated'] = constraint_violated
            supplier['passes_constraints'] = not constraint_violated
            supplier['constraint_score'] = self._calculate_constraint_score(constraint_results)

            if constraint_violated:
                self.filtered_count += 1

            all_suppliers.append(supplier)

        return all_suppliers

    @staticmethod
    def compute_cvr(suppliers: List[Dict]) -> float:
        """
        Constraint Violation Rate: fraction of suppliers that violate at least
        one constraint.  Returns a value in [0, 1] — lower is worse.
        CVR = 1 - (violations / total)
        """
        if not suppliers:
            return 1.0
        violations = sum(1 for s in suppliers if s.get("constraint_violated", False))
        return 1.0 - (violations / len(suppliers))

    def _extract_price(self, supplier: Dict) -> float:
        """Extract numeric price from supplier data."""
        price = supplier.get("price min") or supplier.get("Price Min")

        if price is None:
            price_str = supplier.get("price") or supplier.get("Price")
            if price_str:
                try:
                    if isinstance(price_str, str) and '-' in price_str:
                        return float(price_str.split('-')[0].strip())
                    return float(price_str)
                except (ValueError, TypeError):
                    return 0.0
            return 0.0

        try:
            return float(price)
        except (ValueError, TypeError):
            return 0.0

    def _extract_moq(self, supplier: Dict) -> float:
        """Extract numeric MOQ from supplier data."""
        moq = supplier.get("min order qty") or supplier.get("Min Order Qty")
        if moq:
            try:
                return float(moq)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _extract_lead_time(self, supplier: Dict) -> float:
        """Extract numeric lead time from supplier data."""
        lead_time = supplier.get("lead time") or supplier.get("Lead Time")
        if lead_time:
            try:
                return float(lead_time)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _calculate_constraint_score(self, constraint_results: Dict) -> float:
        """Calculate a 0-1 score based on how well supplier meets constraints."""
        if not constraint_results:
            return 1.0

        passed = sum(1 for c in constraint_results.values() if c.get('passed', True))
        total = len(constraint_results)

        return passed / total if total > 0 else 1.0

    def get_filter_summary(self) -> Dict:
        """Get summary of filters applied."""
        return {
            'filters_applied': self.filters_applied,
            'total_filtered': self.filtered_count
        }


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_constraint_engine = None


def get_constraint_engine() -> ConstraintEngine:
    """Get or create constraint engine singleton."""
    global _constraint_engine
    if _constraint_engine is None:
        _constraint_engine = ConstraintEngine()
    return _constraint_engine