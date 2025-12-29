package com.somas.global_sources_scraper;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;
import io.restassured.RestAssured;
import io.restassured.response.Response;

import java.io.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

import static io.restassured.RestAssured.given;

public class App {

    private static final ObjectMapper mapper = new ObjectMapper();

    static {
        RestAssured.baseURI = "https://www.globalsources.com";
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("❌ Usage: java -jar somasjar.jar <input_csv_path> <output_csv_path>");
            return;
        }

        String inputCsv = args[0];
        String outputCsv = args[1];

        try {
            List<String> queries = readSearchTerms(inputCsv);
            List<Map<String, String>> allData = new ArrayList<>();

            for (String query : queries) {
                System.out.println("\n🔍 Query: " + query);
                int page = 1;

                while (true) {
                    JsonNode result = fetchProducts(query, page);

                    if (result == null || !result.has("data") || !result.get("data").has("list")) {
                        System.out.println("⚠ No data or unexpected structure for query '" + query + "', page " + page);
                        break;
                    }

                    JsonNode products = result.get("data").get("list");
                    int totalPages = result.get("data").has("totalPage") ? result.get("data").get("totalPage").asInt() : 1;

                    for (JsonNode item : products) {
                        allData.add(extractProductData(item, query));
                    }

                    System.out.println("   ✅ Page " + page + "/" + totalPages + " | " + products.size() + " items");

                    page++;
                    if (page > totalPages) break;

                    TimeUnit.SECONDS.sleep(1);
                }
            }

            writeToCSV(allData, outputCsv);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static JsonNode fetchProducts(String query, int pageNum) {
        try {
            Map<String, Object> payload = new LinkedHashMap<>();
            payload.put("pageNum", pageNum);
            payload.put("pageSize", 80);
            payload.put("query", query);
            payload.put("popupFlag", false);
            payload.put("options", Map.of("QUERY_PRODUCT_WITH_LLM_CORE_TERM", "w_1"));

            Response response = given()
                    .header("Content-Type", "application/json")
                    .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
                    .header("Referer", "https://www.globalsources.com/")
                    .header("Origin", "https://www.globalsources.com")
                    .body(mapper.writeValueAsString(payload))
                    .when()
                    .post("/api/agg-search/DESKTOP/v3/product/search")
                    .then()
                    .extract().response();

            if (response.statusCode() != 200) {
                System.out.println("❌ HTTP error for query '" + query + "', page " + pageNum + ": " + response.statusCode());
                return null;
            }

            String responseBody = response.getBody().asString();
            System.out.println("Response snippet: " + responseBody.substring(0, Math.min(500, responseBody.length())));
            return mapper.readTree(responseBody);

        } catch (Exception e) {
            System.out.println("❌ Error for query '" + query + "', page " + pageNum + ": " + e.getMessage());
            return null;
        }
    }

    private static List<String> readSearchTerms(String csvPath) throws IOException {
        List<String> queries = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(csvPath))) {
            String[] line;
            try {
                reader.readNext(); // Skip header
                while ((line = reader.readNext()) != null) {
                    queries.add(line[0]); // Assuming 'query' is the first column
                }
            } catch (CsvValidationException e) {
                e.printStackTrace();
            }
        }
        return queries;
    }

    private static Map<String, String> extractProductData(JsonNode item, String query) {
        Map<String, String> data = new LinkedHashMap<>();
        JsonNode supplier = item.path("supplier");
        JsonNode cat = item.path("categoryInfo");

        data.put("Query", query);
        data.put("Product Name", cleanHtml(item.path("productName").asText("")));
        data.put("Product ID", item.path("productId").asText());
        data.put("Model Number", item.path("modelNumber").asText());
        data.put("Price", item.path("price").asText());
        data.put("Price Min", item.path("specifyFobPriceRangeLow").asText());
        data.put("Price Max", item.path("specifyFobPriceRangeUp").asText());
        data.put("Min Order Qty", item.path("minOrderQuantity").asText());
        data.put("Unit", item.path("minOrderUnit").asText());
        data.put("Lead Time", item.path("leadTime").asText());
        data.put("FOB Port", item.path("fobPort").asText());
        data.put("Product URL", "https://www.globalsources.com" + item.path("desktopProductDetailUrl").asText());
        data.put("Image URL", item.path("primaryImageUrl").asText());
        data.put("Company Name", item.path("companyName").asText());
        data.put("Supplier Name", supplier.path("supplierName").asText());
        data.put("Supplier Location", supplier.path("supplierLocation").asText());
        data.put("Business Type", supplier.path("businessType").toString());
        data.put("Certifications", supplier.path("companyCerts").asText());
        data.put("Years with GS", supplier.path("globalSourcesYear").asText());
        data.put("Category L1", cat.path("l1CategoryVo").path("categoryName").asText());
        data.put("Category L2", cat.path("l2CategoryVo").path("categoryName").asText());
        data.put("Category L3", cat.path("l3CategoryVo").path("categoryName").asText());
        data.put("Category L4", cat.path("l4CategoryVo").path("categoryName").asText());

        return data;
    }

    private static void writeToCSV(List<Map<String, String>> data, String outputPath) throws IOException {
        if (data.isEmpty()) {
            System.out.println("⚠ No data to write.");
            return;
        }

        try (CSVWriter writer = new CSVWriter(new FileWriter(outputPath))) {
            Set<String> headers = data.get(0).keySet();
            writer.writeNext(headers.toArray(new String[0]));

            for (Map<String, String> row : data) {
                writer.writeNext(row.values().toArray(new String[0]));
            }
        }
        System.out.println("✅ Data written to " + outputPath);
    }

    private static String cleanHtml(String rawHtml) {
        return rawHtml.replaceAll("<[^>]*>", "");
    }
}
