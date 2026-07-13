import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 100})

OUTPUT_DIR = "report_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_latency_comparison(df):
    """Bar chart for Latency per Strategy"""
    if "Strategy" not in df.columns:
        return
    
    plt.figure()
    sns.barplot(data=df, x="Strategy", y="Latency (s)", palette="viridis")
    plt.title("Average Latency by Prompting Strategy (Comprehensive Test)")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_comparison_comprehensive.png")
    print(f"Saved {OUTPUT_DIR}/latency_comparison_comprehensive.png")

def plot_rouge_comparison(df):
    """Bar chart for ROUGE-L per Strategy"""
    if "Strategy" not in df.columns:
        return

    plt.figure()
    sns.barplot(data=df, x="Strategy", y="ROUGE-L", palette="magma")
    plt.title("Answer Quality (ROUGE-L) by Prompting Strategy (Comprehensive Test)")
    plt.ylabel("ROUGE-L Score")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rouge_comparison_comprehensive.png")
    print(f"Saved {OUTPUT_DIR}/rouge_comparison_comprehensive.png")

def main():
    # Load comprehensive data
    comprehensive_file = "evaluation_results_comprehensive.csv"
    
    if os.path.exists(comprehensive_file):
        print(f"Loading {comprehensive_file}...")
        df = pd.read_csv(comprehensive_file)
        plot_latency_comparison(df)
        plot_rouge_comparison(df)
    else:
        print(f"File {comprehensive_file} not found!")

if __name__ == "__main__":
    main()
