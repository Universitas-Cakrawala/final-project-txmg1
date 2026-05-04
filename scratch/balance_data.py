import pandas as pd
import numpy as np

# Load original prepared data
df = pd.read_csv('data/processed/reviews_prepared.csv')
print(f"Original distribution:\n{df['sentiment'].value_counts()}")

# Identify minority class count
min_samples = df['sentiment'].value_counts().min()
print(f"Balancing all classes to {min_samples} samples each...")

# Resample each class
balanced_df = pd.DataFrame()
for sentiment in df['sentiment'].unique():
    subset = df[df['sentiment'] == sentiment]
    resampled = subset.sample(n=min_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, resampled])

# Shuffle
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
balanced_df.to_csv('data/processed/balanced_reviews.csv', index=False)
print("Saved to data/processed/balanced_reviews.csv")
print(f"New distribution:\n{balanced_df['sentiment'].value_counts()}")
