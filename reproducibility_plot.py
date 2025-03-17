import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from itertools import combinations
from matplotlib.colors import LinearSegmentedColormap

# Function to compute Pearson correlation confidence interval
def pearsonr_ci(x, y, ci=95, n_boots=100):
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Bootstrap resampling
    boot_indices = np.random.choice(len(x), (n_boots, len(x)), replace=True)
    r_boots = np.array([pearsonr(x[idx], y[idx])[0] for idx in boot_indices])
    
    ci_low = np.percentile(r_boots, (100 - ci) / 2)
    ci_high = np.percentile(r_boots, (ci + 100) / 2)
    return ci_low, ci_high

# Function to clean and format the dataset
def clean_dataset(file_path):
    df = pd.read_csv(file_path, header=None, low_memory=False)
    
    # Convert to list for efficient indexing
    source_plate = df.iloc[0, 2:].tolist()
    condition_names = df.iloc[1, 2:].tolist()
    replicate_labels = df.iloc[2, 2:].tolist()

    new_columns = [f"{cond}_{int(src)}_{rep}" for cond, src, rep in zip(condition_names, source_plate, replicate_labels)]
    
    # Drop first four rows, reset index
    df_cleaned = df.iloc[4:, 2:].reset_index(drop=True)

    # Set new column names
    df_cleaned.columns = new_columns

    # Convert data to numeric
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='coerce')
    return df_cleaned

# Function to generate replicate reproducibility plot
def plot_reproducibility(df):
    df.columns = df.columns.astype(str)
    
    # Extract replicate labels and generate pairs
    replicate_labels = sorted({col.rsplit("_", 1)[-1] for col in df.columns})
    replicate_pairs = [
        (col, col.replace(f"_{label1}", f"_{label2}"))
        for label1, label2 in combinations(replicate_labels, 2)
        for col in df.columns if f"_{label1}" in col and col.replace(f"_{label1}", f"_{label2}") in df.columns
    ]

    if not replicate_pairs:
        print("⚠ No replicate pairs found. Check column names.")
        return

    df_norm = pd.concat(
        [pd.DataFrame({"Replicate_1": df[rep1], "Replicate_2": df[rep2], "Condition": rep1.rsplit("_", 2)[0]})
         for rep1, rep2 in replicate_pairs if rep2 in df.columns], ignore_index=True
    ).dropna()

    if len(df_norm) < 2:
        print("⚠ Not enough data to compute correlation.")
        return

    # Define custom colormap
    white_viridis = LinearSegmentedColormap.from_list(
        'white_viridis', [(0, '#ffffff'), (1e-200, '#440053'), (0.001, '#404388'), (0.01, '#2a788e'), 
                          (0.1, '#21a784'), (0.3, '#fde624'), (1, 'r')], N=65535)

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(df_norm["Replicate_1"], df_norm["Replicate_2"], gridsize=400, cmap=white_viridis, mincnt=1)
    cbar = plt.colorbar(hb, label='Number of points per bin')
    cbar.ax.set_ylabel("Number of points per bin", labelpad=15)  # Properly increase title spacing

    correlation, p_value = pearsonr(df_norm["Replicate_1"], df_norm["Replicate_2"])
    pci_low, pci_high = pearsonr_ci(df_norm["Replicate_1"], df_norm["Replicate_2"], ci=95, n_boots=100)

    # Set axis limits to 10000
    ax.set_xlim(0, 10000)
    ax.set_ylim(0, 10000)
    
    # Move annotation slightly further up and left
    x_pos = 300  # Shift slightly left
    y_pos = 9900  # Shift slightly higher

    # Add text annotations with slightly more spacing
    ax.text(x_pos, y_pos, f"$r = {round(correlation, 3)}$", fontsize=12, color="black", ha="left", va="top")
    ax.text(x_pos, y_pos - 275, f"$p$-value = {format(p_value, '.3g')}", fontsize=12, color="black", ha="left", va="top")
    ax.text(x_pos, y_pos - 600, f"95% CI = ({round(pci_low, 3)}, {round(pci_high, 3)})", fontsize=12, color="black", ha="left", va="top")

    # Adjust spacing around titles and axis labels
    plt.xlabel("Replicate 1", labelpad=20)
    plt.ylabel("Replicate 2", labelpad=20)
    plt.title("Curated Reproducibility Plot", pad=25)
    plt.savefig("reproducibility_plot.tiff", dpi=600, format='tiff', bbox_inches='tight')
    print("✅ Reproducibility plot saved as 'reproducibility_plot.tiff'")
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reproducibility_plot.py <path_to_csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    df_cleaned = clean_dataset(file_path)
    plot_reproducibility(df_cleaned)
