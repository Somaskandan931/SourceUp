"""
Response Formatting Module
---------------------------
Formats search results for display to users.
"""


def format_response ( results ) :
    """
    Format supplier search results for display.

    Args:
        results: List of supplier dictionaries with 'supplier', 'reasons', etc.

    Returns:
        Formatted string with top 3 suppliers
    """
    if not results :
        return "No suppliers found."

    lines = []
    for r in results[:3] :
        supplier = r.get( "supplier", "Unknown supplier" )
        reasons = r.get( "reasons", [] )

        reason_text = ", ".join( reasons ) if reasons else "Matched your requirements"
        lines.append( f"{supplier} → {reason_text}" )

    return "\n".join( lines )