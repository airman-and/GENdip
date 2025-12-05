import pandas as pd

df = pd.read_csv('/root/workspace/andycho/GenDL-LatentControl/dataset/celebA/list_attr_celeba.csv')
total = len(df)

print(f"Total samples: {total}\n")
print(f"{'Attribute':<25} {'Count':<10} {'Percentage':<10}")
print("-" * 45)

# Sort by count descending for better readability
counts = {}
for col in df.columns:
    if col == 'image_id': continue
    counts[col] = len(df[df[col] == 1])

sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

for col, count in sorted_counts.items():
    percent = (count / total) * 100
    print(f"{col:<25} {count:<10} {percent:.2f}%")
