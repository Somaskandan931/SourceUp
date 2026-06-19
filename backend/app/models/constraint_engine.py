"""
Constraint Engine - SME-Specific Filtering
-------------------------------------------
Marks suppliers that violate hard constraints without removing them.
The ranker then applies a penalty score so violators appear at the bottom.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from backend.app.utils.fields import get_field


class ConstraintEngine:

    def __init__(self):
        self.filters_applied = []
        self.filtered_count = 0

    def apply_constraints(self, suppliers: List[Dict], constraints: Dict) -> List[Dict]:
        if not suppliers:
            return []
        self.filters_applied = []
        self.filtered_count = 0
        return self._apply_constraints_iterative(suppliers, constraints)

    def _apply_constraints_iterative(self, suppliers: List[Dict], constraints: Dict) -> List[Dict]:
        all_suppliers = []

        for supplier in suppliers:
            constraint_results = {}
            constraint_violated = False

            # ── CONSTRAINT 1: Budget ──────────────────────────────────────────
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
                    constraint_results['budget'] = {'passed': True, 'reason': 'No price information available'}

            # ── CONSTRAINT 2: MOQ Affordability ──────────────────────────────
            moq_budget = constraints.get("moq_budget")
            if moq_budget is not None:
                moq = self._extract_moq(supplier)
                supplier_price = self._extract_price(supplier)
                if moq > 0 and supplier_price > 0:
                    total_moq_cost = moq * supplier_price
                    can_afford = total_moq_cost <= moq_budget
                    constraint_results['moq_affordability'] = {
                        'passed': can_afford,
                        'moq': moq,
                        'total_cost': total_moq_cost,
                        'budget': moq_budget,
                        'reason': f"MOQ cost ${total_moq_cost:.2f} {'≤' if can_afford else '>'} ${moq_budget:.2f}"
                    }
                    if not can_afford:
                        constraint_violated = True
                        if 'moq_affordability' not in self.filters_applied:
                            self.filters_applied.append('moq_affordability')

            # ── CONSTRAINT 3: Delivery Urgency ───────────────────────────────
            max_lead_time = constraints.get("max_lead_time")
            if max_lead_time is not None:
                lead_time = self._extract_lead_time(supplier)
                if lead_time > 0:
                    meets_deadline = lead_time <= max_lead_time
                    constraint_results['delivery_urgency'] = {
                        'passed': meets_deadline,
                        'lead_time': lead_time,
                        'max_allowed': max_lead_time,
                        'reason': f"{lead_time} days {'≤' if meets_deadline else '>'} {max_lead_time} days required"
                    }
                    if not meets_deadline:
                        constraint_violated = True
                        if 'delivery_urgency' not in self.filters_applied:
                            self.filters_applied.append('delivery_urgency')

            # ── CONSTRAINT 4: Required Certifications ────────────────────────
            required_certs = constraints.get("required_certifications", [])
            if required_certs:
                supplier_certs = str(get_field(supplier, "certifications", default="") or "").lower()
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

            # ── CONSTRAINT 5: Location Preference (RapidFuzz fuzzy match) ────
            preferred_location = constraints.get("preferred_location")
            location_mandatory = constraints.get("location_mandatory", False)
            if preferred_location:
                supplier_location = str(
                    get_field(supplier, "supplier_location") or
                    get_field(supplier, "location", default="") or ""
                ).lower()
                if RAPIDFUZZ_AVAILABLE and supplier_location:
                    fuzzy_score = fuzz.token_set_ratio(preferred_location.lower(), supplier_location)
                else:
                    fuzzy_score = 100.0 if preferred_location.lower() in supplier_location else 0.0
                is_preferred = fuzzy_score >= 70
                constraint_results['location'] = {
                    'passed': is_preferred or not location_mandatory,
                    'preferred': preferred_location,
                    'actual': supplier_location,
                    'mandatory': location_mandatory,
                    'fuzzy_score': fuzzy_score,
                    'reason': f"Location {'matches' if is_preferred else 'differs from'} preference (fuzzy={fuzzy_score:.0f})"
                }
                if location_mandatory and not is_preferred:
                    constraint_violated = True
                    if 'location' not in self.filters_applied:
                        self.filters_applied.append('location')

            # ── CONSTRAINT 6: Minimum Experience ─────────────────────────────
            min_years = constraints.get("min_years_experience")
            if min_years is not None:
                years = get_field(supplier, "years_with_gs", "years_on_platform", default=0) or 0
                try:
                    years_value = float(years)
                    meets_experience = years_value >= min_years
                    constraint_results['experience'] = {
                        'passed': meets_experience,
                        'years': years_value,
                        'min_required': min_years,
                        'reason': f"{years_value} years {'≥' if meets_experience else '<'} {min_years} required"
                    }
                    if not meets_experience:
                        constraint_violated = True
                        if 'experience' not in self.filters_applied:
                            self.filters_applied.append('experience')
                except (ValueError, TypeError):
                    pass

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
        if not suppliers:
            return 1.0
        violations = sum(1 for s in suppliers if s.get("constraint_violated", False))
        return 1.0 - (violations / len(suppliers))

    def _extract_price(self, supplier: Dict) -> float:
        val = get_field(supplier, "price_min") or get_field(supplier, "price")
        if val:
            try:
                s = str(val).strip()
                return float(s.split('-')[0].strip()) if '-' in s else float(s)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _extract_moq(self, supplier: Dict) -> float:
        val = get_field(supplier, "min_order_qty")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _extract_lead_time(self, supplier: Dict) -> float:
        val = get_field(supplier, "lead_time")
        if val:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
        return 0.0

    def _calculate_constraint_score(self, constraint_results: Dict, gamma: float = 0.5) -> float:
        """
        Graduated soft scoring instead of binary passed/failed averaging.
        A supplier slightly over budget/lead-time scores better than one
        wildly over, instead of both being scored identically as 0.
        """
        if not constraint_results:
            return 1.0
        total = 0.0
        count = 0
        for key, c in constraint_results.items():
            if key == 'budget' and 'supplier_price' in c and c.get('max_allowed'):
                penalty = max(0.0, (c['supplier_price'] - c['max_allowed']) / c['max_allowed'])
                total += max(0.0, 1.0 - gamma * penalty)
            elif key == 'moq_affordability' and 'total_cost' in c and c.get('budget'):
                penalty = max(0.0, (c['total_cost'] - c['budget']) / c['budget'])
                total += max(0.0, 1.0 - gamma * penalty)
            elif key == 'delivery_urgency' and 'lead_time' in c and c.get('max_allowed'):
                penalty = max(0.0, (c['lead_time'] - c['max_allowed']) / c['max_allowed'])
                total += max(0.0, 1.0 - gamma * penalty)
            elif key == 'experience' and 'years' in c and c.get('min_required'):
                shortfall = max(0.0, (c['min_required'] - c['years']) / c['min_required'])
                total += max(0.0, 1.0 - gamma * shortfall)
            elif key == 'location' and 'fuzzy_score' in c:
                total += c['fuzzy_score'] / 100.0
            else:
                total += 1.0 if c.get('passed', True) else 0.0
            count += 1
        return total / count if count else 1.0

    def get_filter_summary(self) -> Dict:
        return {
            'filters_applied': self.filters_applied,
            'total_filtered': self.filtered_count
        }


_constraint_engine = None


def get_constraint_engine() -> ConstraintEngine:
    global _constraint_engine
    if _constraint_engine is None:
        _constraint_engine = ConstraintEngine()
    return _constraint_engine