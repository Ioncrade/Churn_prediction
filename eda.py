import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('data/telecom_churn.csv')

# Print summary statistics
def summarize_data(data):
    summary = data.describe()
    print("Summary Statistics:\n", summary)

# Plot distributions and relationships
def plot_data(data):
    sns.pairplot(data, hue='Churn')
    plt.suptitle('Pairplot of Features', y=1.02)
    plt.show()

    data.hist(bins=20, figsize=(14, 9))
    plt.suptitle('Histograms of Features')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

if __name__ == "__main__":
    summarize_data(data)
    plot_data(data)

