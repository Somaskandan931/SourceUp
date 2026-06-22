# SourceUp Procurement Case Study

## Input

```text
Need: food packaging suppliers
Budget: ₹50000
MOQ budget: 500
Location: Chennai
Certification: ISO
```

## Pipeline Trace

- Retrieved (FAISS top-k): 100
- After cross-encoder rerank: 100
- After constraints: 100 (fully matching: 0, location matching: 0)
- Filters applied: certifications, location, moq_affordability
- Ranking method: lightgbm
- Final diversified results: 10

## Top Results

| Rank | Supplier | Product | Price | Location | Score |
|---|---|---|---|---|---|
| 1 | Dalian Huiyou Packaging Co., Ltd. | Fast Food Container Custom Food Grade Burger And Chips Take Away Paper Boxes Fired Chicken Food Packaging | 0.04 - 0.09 | China | 0.06 |
| 2 | Tag Raiser Wooden Products (Dalian) Co., Ltd | Safe Food Grade Kraft Paper Food Container Take Away Lunch Packing Boxes for Carry out Party Resturant | 0.10 | China | -0.0534 |
| 3 | Shenzhen Mellerio Package Co.,Limited | High quality factory price custom 4 side sealed bag for food packaging | 0.05 - 0.10 | China | 0.06 |
| 4 | Dalian Huiyou Packaging Co., Ltd. | Box Free Sample Takeaway Packaging Food Box Takeout Black Fried Chicken Take Out Paper Lunch Box Food Container | 0.04 - 0.09 | China | 0.06 |
| 5 | Huizhou Zhonglifa Industrial Co. Ltd | Net bags of vegetables and firewood onion and potato bags effectively package and print products | 0.04 | China | 0.06 |
| 6 | Fujian Shenglin Huanghe I & E Trading Co., Ltd | OEM Recycled Disposable 1000 ml- 32oz paper food packaging box for packing many kinds of food in daily life | 0.09 - 0.10 | China | 0.0469 |
| 7 | Ningbo Rison Houseware Co., Ltd._Packaging | Food Box Kraft Paper Box Lunch Box Disposable Food Containers Takeaway Packaging Box with transparent | 0.03 - 0.04 | China | 0.0381 |
| 8 | Liaoning Kunze Industrial Group Co., Ltd | Fruit Packaging Boxes Gift Box Packaging Custom Printed Corrugated Cardboard Cherry Fruit Packaging Boxes | 0.75 | China | 0.0333 |
| 9 | YongKang ZhuoTong Houseware Factory | BPA free custom 7 days weekly pill box organizer | 1.65 - 2.00 | China | 0.0333 |
| 10 | Shenzhen Mellerio Package Co.,Limited | Wholesame Factory Price high quality laminated paper and plastic bag for food | 0.05 - 0.20 | China | 0.0333 |

## Feature Contribution — Top Result

| Feature | Contribution |
|---|---|
| price_match | +0.0017 |
| price_ratio | +0.0012 |
| location_match | -0.0039 |
| cert_match | +0.0750 |
| faiss_score | +0.0060 |
