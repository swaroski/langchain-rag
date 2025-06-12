import pandas as pd
import random
from datetime import datetime, timedelta
import uuid

# Define possible values
products = [f"Widget {chr(65+i)}" for i in range(10)] + [f"Gadget {chr(65+i)}" for i in range(10)]
regions = ["North America", "Europe", "Asia", "South America", "Australia"]
descriptions = [
    "High-quality widget for industrial use",
    "Premium gadget with advanced features",
    "Compact device for consumer electronics",
    "Durable widget for heavy machinery",
    "Portable gadget with long battery life",
    "Eco-friendly product for sustainable markets",
    "High-performance device for enterprise use",
    "Cost-effective solution for small businesses",
    "Innovative gadget with AI integration",
    "Reliable widget for everyday applications"
]

# Generate 1000 rows
data = []
start_date = datetime(2024, 1, 1)
for _ in range(1000):
    product = random.choice(products)
    sales = random.randint(50, 200)
    revenue = round(random.uniform(10000, 100000), 2)
    profits = round(revenue * random.uniform(0.2, 0.4), 2)  # 20-40% profit margin
    description = random.choice(descriptions)
    region = random.choice(regions)
    date = (start_date + timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
    data.append({
        "product": product,
        "revenue": revenue,
        "sales": sales,
        "profits": profits,
        "description": description,
        "region": region,
        "date": date
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("sales_data.csv", index=False)
print("Generated sales_data.csv with 1000 rows.")