# SourceUp Procurement Case Study

## Input

```text
Need: packaging materials supplier
Budget: ₹100000
MOQ budget: 500
Location: Mumbai

```

## Pipeline Trace

- Retrieved (FAISS top-k): 100
- After cross-encoder rerank: 100
- After constraints: 100 (fully matching: 76, location matching: 0)
- Filters applied: moq_affordability
- Ranking method: lightgbm
- Final diversified results: 10

## Top Results

| Rank | Supplier | Product | Price | Location | Score |
|---|---|---|---|---|---|
| 1 | Fujian Shenglin Huanghe I & E Trading Co., Ltd | OEM Recycled Disposable 1000 ml- 32oz paper food packaging box for packing many kinds of food in daily life | 0.09 - 0.10 | China | 0.0469 |
| 2 | Shanghai Forests Packaging Group Co. Ltd. | Printed Corrugated Cardboard Paper Fruit Vegetable Packing Packaging Shipping Tray Carton Box | 0.65 - 0.89 | China | -0.0534 |
| 3 | ZhangZhou Taki Industry And Trade Co.,ltd | Eco Friendly Carton Cylinder Kraft Paper Round Empty Biodegradable Craft Cardboard Boxes Packaging Tubes | 0.29 - 0.86 | China | 0.0381 |
| 4 | Ningbo Rison Houseware Co., Ltd._Packaging | Food Box Kraft Paper Box Lunch Box Disposable Food Containers Takeaway Packaging Box with transparent | 0.03 - 0.04 | China | 0.0381 |
| 5 | Good Seller Co., Ltd | Household sealed glass container plastic storage container storage bottle container preservation | 1.24 - 1.26 | China | 0.0381 |
| 6 | Ganzhou Impetus Packaging Products Co.,Limited | Recycled Paper Cylinder Box Glossy Laminating Tube Box | 0.36 - 0.76 | China | 0.0342 |
| 7 | ASIA POWERFUL GROUP LIMITED | Case packaging box | 0.30 - 0.50 | China | 0.0188 |
| 8 | Liaoning Kunze Industrial Group Co., Ltd | Cardboard gift box packing box rectangular corrugated box packaging lined recyclable paper memory honeycomb | 0.30 | China | 0.0188 |
| 9 | Forests Packaging Shanghai Group Co. Ltd. | 25l Cosmetic custom Boxes corrugated cardboard paper mailer box with strips | 0.43 - 0.89 | China | 0.0188 |
| 10 | Emon Packaging Ltd | Custom printing gift box food packaging box cardboard lid and base box for bread | 0.10 - 0.80 | China | 0.0188 |

## Feature Contribution — Top Result

| Feature | Contribution |
|---|---|
| price_match | +0.0022 |
| price_ratio | -0.0005 |
| location_match | -0.0029 |
| cert_match | -0.0016 |
| faiss_score | +0.0698 |
