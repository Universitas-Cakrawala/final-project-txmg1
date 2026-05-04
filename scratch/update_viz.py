import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the latest balanced results (yang mengandung GloVe 71%)
try:
    df = pd.read_csv('results/balanced_comparison_table.csv')
except:
    df = pd.read_csv('results/comparison_table.csv')

# Create a clear combination name
df['Combination'] = df['feature_extractor'] + " + " + df['model']

# Sort by Weighted F1
df = df.sort_values('weighted_f1', ascending=False)

# Setup plotting
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create Bar Plot
ax = sns.barplot(
    x='weighted_f1', 
    y='Combination', 
    data=df, 
    palette='viridis'
)

# Add labels
plt.title('Perbandingan Performa Model (Balanced Dataset)', fontsize=15, pad=20)
plt.xlabel('Weighted F1-Score', fontsize=12)
plt.ylabel('Kombinasi Fitur + Model', fontsize=12)
plt.xlim(0, 1.0)

# Add values on bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.3f', padding=5)

plt.tight_layout()

# Save updated figure
output_path = 'results/figures/evaluation/grouped_bar_f1.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)

print(f"✅ Visualisasi diperbarui: {output_path}")
