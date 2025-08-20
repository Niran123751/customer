# chart.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Seaborn styling (presentation-ready) ---
sns.set_style("whitegrid")        # clean, professional background
sns.set_context("talk")          # slightly larger text for presentation

# --- Synthetic realistic data generation ---
rng = np.random.default_rng(seed=42)

n_customers = 1200

# Define segments and their base spending behaviour
segments = {
    "Low value": {"n": 550, "mu": 3.0, "sigma": 0.6},   # lower median, less spread
    "Mid value": {"n": 450, "mu": 4.2, "sigma": 0.7},   # moderate median, moderate spread
    "High value": {"n": 200, "mu": 5.0, "sigma": 0.9},  # higher median, more spread
}

rows = []
for seg, params in segments.items():
    # Using lognormal to create positive-skew typical of spending amounts
    samples = rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=params["n"])
    # Add a small chance of occasional very large purchases (outliers)
    heavy_tail = rng.choice([0, 1], size=params["n"], p=[0.97, 0.03])
    samples = samples * (1 + heavy_tail * rng.uniform(3.0, 10.0, size=params["n"]))
    for amt in samples:
        rows.append({"segment": seg, "purchase_amount": round(amt, 2)})

df = pd.DataFrame(rows)

# --- Plotting ---
plt.figure(figsize=(8, 8))  # 8x8 inches; we'll save at dpi=64 -> 512x512 px
palette = {"Low value": "#7fbf7f", "Mid value": "#4a90e2", "High value": "#d64545"}

ax = sns.boxplot(
    x="segment",
    y="purchase_amount",
    data=df,
    palette=palette,
    showfliers=True,
    width=0.6,
    boxprops={"linewidth": 1.2, "edgecolor": "k"},
    medianprops={"color": "black", "linewidth": 1.6},
)

# Add jittered points for context (lighter, semi-transparent)
sns.stripplot(
    x="segment",
    y="purchase_amount",
    data=df.sample(frac=0.25, random_state=1),  # sample for readability
    color="k",
    size=3,
    jitter=0.25,
    alpha=0.25,
)

# Labels and title (business-friendly)
ax.set_title("Distribution of Purchase Amounts by Customer Segment", fontsize=18, weight="semibold")
ax.set_xlabel("")  # category labels suffice
ax.set_ylabel("Purchase amount (USD)", fontsize=14)

# Improve y-axis tick formatting (round numbers)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylim(0, df["purchase_amount"].quantile(0.995) * 1.05)  # set max to ~99.5th pct to reduce extreme whitespace

# Annotate medians numerically (for executive readers)
medians = df.groupby("segment")["purchase_amount"].median().round(2)
for i, seg in enumerate(medians.index):
    median_val = medians[seg]
    ax.text(i, median_val * 1.02, f"Median: ${median_val:,.2f}", ha="center", fontsize=10, color="black")

plt.tight_layout()

# Save as 512x512 px: figsize 8x8 inches * dpi=64 -> 512 px
plt.savefig("chart.png", dpi=64, bbox_inches="tight")
plt.close()
