import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load Data
csv_path = 'dataset/celebA/list_attr_celeba.csv'
print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Select attributes of interest
target_attrs = ['Eyeglasses', 'Male', 'Young', 'Bald', 'Smiling', 'Wearing_Hat', 'Gray_Hair']
df_subset = df[target_attrs]

# Calculate Correlation Matrix
# Attributes are -1 or 1. Pearson correlation works fine.
corr_matrix = df_subset.corr()

print("Correlation Matrix:")
print(corr_matrix)

# Plot Heatmap
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.2)
heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('CelebA Attribute Correlation Analysis')

# Save
output_dir = 'output/analysis'
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'bias_correlation_heatmap.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Heatmap saved to {save_path}")

# Specific stats for report
print("\n--- Specific Bias Analysis ---")
# 1. Eyeglasses vs Male
corr_eye_male = df_subset['Eyeglasses'].corr(df_subset['Male'])
print(f"Correlation (Eyeglasses, Male): {corr_eye_male:.4f}")

# 2. Eyeglasses vs Young
corr_eye_young = df_subset['Eyeglasses'].corr(df_subset['Young'])
print(f"Correlation (Eyeglasses, Young): {corr_eye_young:.4f}")

# 3. Bald vs Male
corr_bald_male = df_subset['Bald'].corr(df_subset['Male'])
print(f"Correlation (Bald, Male): {corr_bald_male:.4f}")

# 4. Conditional Probabilities
# P(Male | Eyeglasses)
p_male_given_eye = len(df[(df['Eyeglasses']==1) & (df['Male']==1)]) / len(df[df['Eyeglasses']==1])
p_male_global = len(df[df['Male']==1]) / len(df)
print(f"P(Male | Eyeglasses) = {p_male_given_eye:.4f} (Global P(Male) = {p_male_global:.4f})")

# P(Young | Eyeglasses) vs P(Young)
p_young_given_eye = len(df[(df['Eyeglasses']==1) & (df['Young']==1)]) / len(df[df['Eyeglasses']==1])
p_young_global = len(df[df['Young']==1]) / len(df)
print(f"P(Young | Eyeglasses) = {p_young_given_eye:.4f} (Global P(Young) = {p_young_global:.4f})")
