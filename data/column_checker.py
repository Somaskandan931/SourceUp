import pandas as pd

# Load the CSV and inspect its structure
csv_path = "C:/Users/somas/PycharmProjects/SourceUp/data/clean/suppliers_clean.csv"

print("=" * 70)
print("🔍 CSV Column Inspector")
print("=" * 70)

# Read just the first few rows to inspect
df = pd.read_csv(csv_path, nrows=5)

print(f"\n📊 Total columns found: {len(df.columns)}")
print("\n📋 Column names:")
print("-" * 70)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n" + "=" * 70)
print("📌 First few rows:")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("📈 Data types:")
print("=" * 70)
print(df.dtypes)

print("\n" + "=" * 70)
print("💡 Columns that might contain product information:")
print("=" * 70)
product_related = [col for col in df.columns if any(keyword in col.lower()
                   for keyword in ['product', 'item', 'goods', 'material', 'commodity'])]
if product_related:
    for col in product_related:
        print(f"  • {col}")
else:
    print("  ⚠️ No obvious product-related columns found")
    print("  📝 Showing all columns again for manual inspection:")
    for col in df.columns:
        print(f"     - {col}")
