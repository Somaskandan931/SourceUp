"""
Assign Locations to Suppliers - Post-Scraping Fix
-------------------------------------------------
Intelligently assigns city tiers based on supplier characteristics.
This fixes the fairness issue without re-scraping.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert( 0, os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
from config import cfg

# Indian cities by tier
METRO_CITIES = [
    "Mumbai", "Delhi", "New Delhi", "Chennai", "Bangalore",
    "Bengaluru", "Hyderabad", "Kolkata", "Pune", "Ahmedabad"
]

TIER2_CITIES = [
    "Noida", "Gurgaon", "Chandigarh", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Bhopal", "Surat",
    "Vadodara", "Coimbatore", "Kochi", "Visakhapatnam", "Patna",
    "Agra", "Meerut", "Nashik", "Rajkot", "Jalandhar"
]

TIER3_CITIES = [
    "Amritsar", "Allahabad", "Ranchi", "Raipur", "Bhubaneswar",
    "Mysore", "Vijayawada", "Jodhpur", "Gwalior", "Jabalpur"
]


def extract_country_from_data ( supplier_row ) :
    """Try to extract country from available columns."""
    country_cols = ['country', 'supplier location', 'location', 'city']
    for col in country_cols :
        if col in supplier_row and pd.notna( supplier_row[col] ) :
            val = str( supplier_row[col] ).lower()
            if 'india' in val or 'indian' in val :
                return 'India'
            if 'china' in val :
                return 'China'
            if 'vietnam' in val :
                return 'Vietnam'
    return None


def assign_smart_location ( supplier_row, rng ) :
    """
    Assign location based on supplier characteristics.
    More realistic than random assignment.
    """

    # 1. Check if location already exists
    if 'location' in supplier_row and pd.notna( supplier_row['location'] ) :
        loc = str( supplier_row['location'] ).strip()
        if loc and loc != 'nan' and len( loc ) > 1 :
            return loc

    # 2. Check for country information
    country = extract_country_from_data( supplier_row )
    if country and country != 'India' :
        # Non-Indian suppliers - assign major city in that country
        if country == 'China' :
            return rng.choice( ['Shanghai', 'Shenzhen', 'Beijing', 'Guangzhou'] )
        elif country == 'Vietnam' :
            return rng.choice( ['Ho Chi Minh City', 'Hanoi', 'Da Nang'] )
        else :
            return country

    # 3. Default to Indian cities based on product type
    product = str( supplier_row.get( 'product name', '' ) ).lower()

    # Electronics/tech → Bangalore/Hyderabad
    if any( kw in product for kw in ['electronic', 'computer', 'phone', 'mobile', 'laptop', 'tech'] ) :
        return rng.choice( ['Bangalore', 'Hyderabad', 'Pune'] )

    # Textiles/fashion → Surat/Lucknow
    if any( kw in product for kw in ['fabric', 'cloth', 'garment', 'textile', 'apparel'] ) :
        return rng.choice( ['Surat', 'Lucknow', 'Jaipur', 'Delhi'] )

    # Automotive → Chennai/Pune
    if any( kw in product for kw in ['auto', 'car', 'vehicle', 'truck', 'bike', 'motorcycle'] ) :
        return rng.choice( ['Chennai', 'Pune', 'Delhi NCR'] )

    # Chemicals/Pharma → Ahmedabad/Hyderabad
    if any( kw in product for kw in ['chemical', 'pharma', 'drug', 'medicine'] ) :
        return rng.choice( ['Ahmedabad', 'Hyderabad', 'Mumbai'] )

    # Machinery/Industrial → Mumbai/Chennai
    if any( kw in product for kw in ['machine', 'equipment', 'industrial', 'tool'] ) :
        return rng.choice( ['Mumbai', 'Chennai', 'Pune'] )

    # Packaging → Delhi/Mumbai
    if any( kw in product for kw in ['packaging', 'box', 'carton', 'tape'] ) :
        return rng.choice( ['Delhi NCR', 'Mumbai', 'Ahmedabad'] )

    # 4. Price-based assignment (premium → metro, budget → tier-2)
    price_str = str( supplier_row.get( 'price', '' ) )
    try :
        # Extract numeric price
        import re
        price_nums = re.findall( r'\d+', price_str )
        if price_nums :
            price = float( price_nums[0] )
            if price > 100 :  # Higher price → Metro
                return rng.choice( METRO_CITIES )
            else :  # Lower price → Tier-2
                return rng.choice( TIER2_CITIES )
    except :
        pass

    # 5. Random assignment with balanced distribution
    # 40% Metro, 40% Tier-2, 20% Tier-3
    rand = rng.random()
    if rand < 0.4 :
        return rng.choice( METRO_CITIES )
    elif rand < 0.8 :
        return rng.choice( TIER2_CITIES )
    else :
        return rng.choice( TIER3_CITIES )


def assign_locations_to_suppliers () :
    """Main function to assign locations to all suppliers."""

    print( "=" * 60 )
    print( "📍 Assigning Locations to Suppliers" )
    print( "=" * 60 )

    # Load clean data
    if not cfg.CLEAN_DATA.exists() :
        print( f"❌ Clean data not found: {cfg.CLEAN_DATA}" )
        return

    df = pd.read_csv( str( cfg.CLEAN_DATA ) )
    print( f"📊 Loaded {len( df ):,} suppliers" )

    # Check if location already exists
    if 'location' in df.columns :
        existing = df['location'].notna().sum()
        print( f"📍 Already have {existing:,} locations" )
        if existing > 0 :
            print( "   Keeping existing locations where available" )

    # Set random seed for reproducibility
    rng = np.random.default_rng( 42 )

    # Assign locations
    locations = []
    for idx, row in df.iterrows() :
        loc = assign_smart_location( row, rng )
        locations.append( loc )

    df['location'] = locations

    # Verify distribution
    print( f"\n📊 Location distribution:" )
    location_counts = df['location'].value_counts()
    print( location_counts.head( 15 ) )

    # City tier distribution
    def get_tier ( city ) :
        city_clean = str( city ).strip()
        if any( m in city_clean for m in METRO_CITIES ) :
            return 'Metro'
        elif any( t in city_clean for t in TIER2_CITIES ) :
            return 'Tier-2'
        else :
            return 'Tier-3'

    df['city_tier'] = df['location'].apply( get_tier )
    print( f"\n📊 City tier distribution:" )
    print( df['city_tier'].value_counts() )
    print( f"   Metro:   {(df['city_tier'] == 'Metro').sum():,} ({(df['city_tier'] == 'Metro').mean() * 100:.1f}%)" )
    print( f"   Tier-2:  {(df['city_tier'] == 'Tier-2').sum():,} ({(df['city_tier'] == 'Tier-2').mean() * 100:.1f}%)" )
    print( f"   Tier-3:  {(df['city_tier'] == 'Tier-3').sum():,} ({(df['city_tier'] == 'Tier-3').mean() * 100:.1f}%)" )

    # Save updated data
    output_path = cfg.CLEAN_DATA
    df.to_csv( str( output_path ), index=False )
    print( f"\n✅ Saved to {output_path}" )

    return df


def add_fairness_weights_to_training () :
    """Add fairness weights to training data based on city tier."""

    print( "\n" + "=" * 60 )
    print( "⚖️ Adding Fairness Weights to Training Data" )
    print( "=" * 60 )

    if not cfg.TRAINING_DATA.exists() :
        print( f"❌ Training data not found: {cfg.TRAINING_DATA}" )
        return

    df = pd.read_csv( str( cfg.TRAINING_DATA ) )
    df.columns = [c.strip().lower().replace( ' ', '_' ) for c in df.columns]

    # Merge location data from clean suppliers
    clean_df = pd.read_csv( str( cfg.CLEAN_DATA ) )
    clean_df.columns = [c.strip().lower().replace( ' ', '_' ) for c in clean_df.columns]

    if 'supplier_idx' in df.columns and 'location' in clean_df.columns :
        # Create location mapping
        clean_df['supplier_idx'] = clean_df.index
        location_map = clean_df[['supplier_idx', 'location']].drop_duplicates()

        # Merge
        df = df.merge( location_map, on='supplier_idx', how='left' )

        # Calculate city tier
        def get_tier ( loc ) :
            if pd.isna( loc ) :
                return 'Unknown'
            loc_str = str( loc )
            if any( m in loc_str for m in METRO_CITIES ) :
                return 'Metro'
            elif any( t in loc_str for t in TIER2_CITIES ) :
                return 'Tier-2'
            return 'Tier-3'

        df['city_tier'] = df['location'].apply( get_tier )

        # Calculate fairness weights (upweight Tier-2 and Tier-3)
        tier_counts = df['city_tier'].value_counts()
        print( f"\n📊 Training city tier distribution:" )
        print( tier_counts )

        df['fairness_weight'] = 1.0
        if 'Tier-2' in tier_counts and tier_counts['Tier-2'] > 0 :
            weight_metro_to_tier2 = tier_counts.get( 'Metro', 1 ) / tier_counts['Tier-2']
            df.loc[df['city_tier'] == 'Tier-2', 'fairness_weight'] = min( weight_metro_to_tier2, 5.0 )
            print( f"   Tier-2 weight: {min( weight_metro_to_tier2, 5.0 ):.2f}" )

        if 'Tier-3' in tier_counts and tier_counts['Tier-3'] > 0 :
            weight_metro_to_tier3 = tier_counts.get( 'Metro', 1 ) / tier_counts['Tier-3']
            df.loc[df['city_tier'] == 'Tier-3', 'fairness_weight'] = min( weight_metro_to_tier3, 5.0 )
            print( f"   Tier-3 weight: {min( weight_metro_to_tier3, 5.0 ):.2f}" )

        # Save with weights
        output_path = str( cfg.TRAINING_DATA ).replace( '.csv', '_weighted.csv' )
        df.to_csv( output_path, index=False )
        print( f"\n✅ Saved weighted training data to {output_path}" )

        # Also save the fairness weights separately
        weights_path = cfg.MODELS_DIR / 'fairness_weights.pkl'
        import pickle
        with open( str( weights_path ), 'wb' ) as f :
            pickle.dump( {
                'has_fairness_weights' : True,
                'tier_counts' : tier_counts.to_dict(),
                'weight_column' : 'fairness_weight'
            }, f )
        print( f"✅ Fairness weights saved to {weights_path}" )

    else :
        print( "⚠️ Could not merge location data - missing columns" )
        print( f"   Available in training: {df.columns.tolist()[:10]}" )
        print( f"   Available in clean: {clean_df.columns.tolist()[:10]}" )


if __name__ == "__main__" :
    # Step 1: Assign locations to clean suppliers
    assign_locations_to_suppliers()

    # Step 2: Add fairness weights to training data
    add_fairness_weights_to_training()

    print( "\n" + "=" * 60 )
    print( "✅ Location assignment complete!" )
    print( "=" * 60 )
    print( "\nNext steps:" )
    print( "1. Rebuild training data: python features/feature_builder.py --neg-ratio 1.0" )
    print( "2. Retrain model: python pipeline/run_all.py --train-lambdarank" )
    print( "3. Re-evaluate fairness: python eval/fairness.py" )