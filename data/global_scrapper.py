import requests
import csv
import os
import time
import re

# === Configuration ===
SEARCH_CSV_PATH = 'C:/Users/somas/PycharmProjects/ONSOURCE/data/supplier_search_terms.csv'
OUTPUT_CSV_PATH = 'C:/Users/somas/PycharmProjects/ONSOURCE/data/output.csv'

POST_URL = 'https://www.globalsources.com/api/agg-search/DESKTOP/v3/product/search'

HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Referer': 'https://www.globalsources.com/',
    'Origin': 'https://www.globalsources.com',
}

def clean_html(raw_html):
    return re.sub(r'<[^>]+>', '', raw_html)

def read_search_terms(file_path):
    with open(file_path, newline='', encoding='utf-8') as f:
        return [row['query'] for row in csv.DictReader(f)]

def fetch_products(query, page_num):
    payload = {
        "pageNum": page_num,
        "pageSize": 80,
        "query": query,
        "popupFlag": False,
        "options": {"QUERY_PRODUCT_WITH_LLM_CORE_TERM": "w_1"}
    }

    try:
        response = requests.post(POST_URL, headers=HEADERS, json=payload, timeout=15)
        print(f"Request for query '{query}', page {page_num} returned status code: {response.status_code}")
        # Print a snippet of response for debugging
        print(f"Response snippet: {response.text[:500]}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error on query '{query}', page {page_num}: {e}")
        return None

def extract_product_data(item, query):
    supplier = item.get('supplier', {})
    cat = item.get('categoryInfo', {})
    return {
        'Query': query,
        'Product Name': clean_html(item.get('productName', '')),
        'Product ID': item.get('productId'),
        'Model Number': item.get('modelNumber'),
        'Price': item.get('price'),
        'Price Min': item.get('specifyFobPriceRangeLow'),
        'Price Max': item.get('specifyFobPriceRangeUp'),
        'Min Order Qty': item.get('minOrderQuantity'),
        'Unit': item.get('minOrderUnit'),
        'Lead Time': item.get('leadTime'),
        'FOB Port': item.get('fobPort'),
        'Product URL': f"https://www.globalsources.com{item.get('desktopProductDetailUrl')}",
        'Image URL': item.get('primaryImageUrl'),
        'Company Name': item.get('companyName'),
        'Supplier Name': supplier.get('supplierName'),
        'Supplier Location': supplier.get('supplierLocation'),
        'Business Type': ', '.join(supplier.get('businessType', [])),
        'Certifications': supplier.get('companyCerts'),
        'Years with GS': supplier.get('globalSourcesYear'),
        'Category L1': cat.get('l1CategoryVo', {}).get('categoryName', ''),
        'Category L2': cat.get('l2CategoryVo', {}).get('categoryName', ''),
        'Category L3': cat.get('l3CategoryVo', {}).get('categoryName', ''),
        'Category L4': cat.get('l4CategoryVo', {}).get('categoryName', ''),
    }

def write_to_csv(data, output_path):
    if not data:
        print("⚠️ No data to write.")
        return
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Data written to {output_path}")

def scrape_all():
    all_data = []
    queries = read_search_terms(SEARCH_CSV_PATH)

    for query in queries:
        print(f"\n🔍 Query: {query}")
        page = 1
        while True:
            result = fetch_products(query, page)
            if not result or 'data' not in result or 'list' not in result['data']:
                print(f"⚠️ No data found or unexpected response structure for query '{query}', page {page}")
                break

            products = result['data']['list']
            total_pages = result['data'].get('totalPage', 1)

            for item in products:
                all_data.append(extract_product_data(item, query))

            print(f"   ✅ Page {page}/{total_pages} | {len(products)} items")
            page += 1
            if page > total_pages:
                break
            time.sleep(1)  # polite pause

    write_to_csv(all_data, OUTPUT_CSV_PATH)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)
    scrape_all()
