import pandas as pd
from src.experiment_runner import run_experiments

# Load balanced data
df_balanced = pd.read_csv('data/processed/balanced_reviews.csv')

# Run experiments
print("\n🔄 Menjalankan Eksperimen pada Dataset Seimbang (Balanced)...")
results_df = run_experiments(
    df_balanced, 
    subset='priority', 
    save_dir='results',
    n_iter=2, # Cepat saja untuk testing
    cv=2
)

# Rename table to avoid overwriting original results
balanced_table_path = 'results/balanced_comparison_table.csv'
results_df.to_csv(balanced_table_path, index=False)

print(f"\n✅ Eksperimen Selesai! Hasil disimpan di: {balanced_table_path}")
print("\n📊 Hasil Top 3 (Balanced Data):")
print(results_df[['Extractor', 'Model', 'Weighted F1', 'Macro F1', 'Accuracy']].head(3))
