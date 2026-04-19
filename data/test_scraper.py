import os
from global_scrapper import read_search_terms, fetch_products, extract_product_data, write_to_csv

# Adjust these paths as per your environment
SEARCH_CSV_PATH = 'C:/Users/somas/PycharmProjects/ONSOURCE/data/supplier_search_terms.csv'
TEST_OUTPUT_CSV = 'C:/Users/somas/PycharmProjects/ONSOURCE/data/test_output.csv'

def test_read_search_terms():
    queries = read_search_terms(SEARCH_CSV_PATH)
    print("Search terms loaded:", queries)
    assert isinstance(queries, list) and len(queries) > 0, "No queries loaded!"
    print("✅ read_search_terms passed.")

def test_fetch_products():
    sample_query = 'disposable food containers'
    result = fetch_products(sample_query, 1)
    assert result is not None, "fetch_products returned None!"
    assert 'data' in result and 'list' in result['data'], "Unexpected API response structure!"
    print(f"API fetch returned {len(result['data']['list'])} products for query '{sample_query}' page 1")
    print("✅ fetch_products passed.")
    return result

def test_extract_product_data(api_result):
    items = api_result['data']['list']
    sample_item = items[0]
    extracted = extract_product_data(sample_item, 'disposable food containers')
    print("Extracted product data sample:")
    for k, v in extracted.items():
        print(f"  {k}: {v}")
    assert 'Product Name' in extracted, "Missing expected keys in extraction!"
    print("✅ extract_product_data passed.")
    return [extracted]

def test_write_to_csv(data):
    write_to_csv(data, TEST_OUTPUT_CSV)
    assert os.path.exists(TEST_OUTPUT_CSV), "CSV file was not created!"
    file_size = os.path.getsize(TEST_OUTPUT_CSV)
    assert file_size > 0, "CSV file is empty!"
    print(f"CSV file written successfully: {TEST_OUTPUT_CSV} ({file_size} bytes)")
    print("✅ write_to_csv passed.")

if __name__ == "__main__":
    test_read_search_terms()
    api_response = test_fetch_products()
    extracted_data = test_extract_product_data(api_response)
    test_write_to_csv(extracted_data)
