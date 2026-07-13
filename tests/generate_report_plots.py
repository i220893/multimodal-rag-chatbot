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
    plt.title("Average Latency by Prompting Strategy")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/latency_comparison.png")
    print(f"Saved {OUTPUT_DIR}/latency_comparison.png")

def plot_rouge_comparison(df):
    """Bar chart for ROUGE-L per Strategy"""
    if "Strategy" not in df.columns:
        return

    plt.figure()
    sns.barplot(data=df, x="Strategy", y="ROUGE-L", palette="magma")
    plt.title("Answer Quality (ROUGE-L) by Prompting Strategy")
    plt.ylabel("ROUGE-L Score")
    plt.xlabel("Strategy")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rouge_comparison.png")
    print(f"Saved {OUTPUT_DIR}/rouge_comparison.png")

def plot_retrieval_hit_rate(df):
    """Pie chart for Retrieval Hit Rate"""
    if "Retrieval Hit" not in df.columns:
        return

    # Normalize values (Yes/No vs ✅ Yes/❌ No)
    df["Retrieval Hit"] = df["Retrieval Hit"].apply(lambda x: "Hit" if "Yes" in x else "Miss")
    
    counts = df["Retrieval Hit"].value_counts()
    
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=90)
    plt.title("Retrieval System Accuracy (Hit Rate)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/retrieval_hit_rate.png")
    print(f"Saved {OUTPUT_DIR}/retrieval_hit_rate.png")

def main():
    # Load data
    detailed_file = "evaluation_results_detailed.csv"
    enhanced_file = "evaluation_results_enhanced.csv"
    
    if os.path.exists(detailed_file):
        print(f"Loading {detailed_file}...")
        df_detailed = pd.read_csv(detailed_file)
        plot_latency_comparison(df_detailed)
        plot_rouge_comparison(df_detailed)
        
    if os.path.exists(enhanced_file):
        print(f"Loading {enhanced_file}...")
        df_enhanced = pd.read_csv(enhanced_file)
        plot_retrieval_hit_rate(df_enhanced)

if __name__ == "__main__":
    main()
