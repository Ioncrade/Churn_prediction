import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load data and perform initial exploration"""
    df = pd.read_csv('data/telecom_churn.csv')
    
    print("="*80)
    print("TELECOM CHURN ANALYSIS - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1]}")
    print(f"Number of samples: {df.shape[0]}")
    
    print("\n2. DATA TYPES AND MISSING VALUES")
    print("-" * 40)
    print(df.info())
    
    print("\n3. MISSING VALUES ANALYSIS")
    print("-" * 30)
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0] if missing_data.sum() > 0 else "No missing values found!")
    
    return df

def analyze_target_variable(df):
    """Analyze the target variable (Churn)"""
    print("\n4. TARGET VARIABLE ANALYSIS")
    print("-" * 35)
    
    churn_counts = df['Churn'].value_counts()
    churn_percentage = df['Churn'].value_counts(normalize=True) * 100
    
    print("Churn Distribution:")
    print(f"No Churn (0): {churn_counts[0]} ({churn_percentage[0]:.2f}%)")
    print(f"Churn (1): {churn_counts[1]} ({churn_percentage[1]:.2f}%)")
    
    # Plot churn distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar plot
    churn_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral'])
    ax1.set_title('Churn Distribution')
    ax1.set_xlabel('Churn (0=No, 1=Yes)')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    ax2.pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
            colors=['skyblue', 'lightcoral'])
    ax2.set_title('Churn Percentage')
    
    plt.tight_layout()
    plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_features(df):
    """Analyze individual features"""
    print("\n5. FEATURE ANALYSIS")
    print("-" * 25)
    
    # Numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Churn')  # Remove target variable
    
    print(f"Numerical features: {numerical_features}")
    
    # Statistical summary
    print("\nStatistical Summary:")
    print(df[numerical_features + ['Churn']].describe())
    
    # Distribution plots for numerical features
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        # Histogram with KDE
        axes[idx].hist(df[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Distribution of {feature}')
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel('Frequency')
    
    # Remove empty subplot
    if len(numerical_features) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_relationships(df):
    """Analyze relationships between features and churn"""
    print("\n6. FEATURE RELATIONSHIPS WITH CHURN")
    print("-" * 40)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Churn')
    
    # Box plots for numerical features vs Churn
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()
    
    for idx, feature in enumerate(numerical_features):
        sns.boxplot(x='Churn', y=feature, data=df, ax=axes[idx])
        axes[idx].set_title(f'{feature} vs Churn')
        axes[idx].set_xlabel('Churn (0=No, 1=Yes)')
    
    # Remove empty subplot
    if len(numerical_features) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('features_vs_churn.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df):
    """Perform correlation analysis"""
    print("\n7. CORRELATION ANALYSIS")
    print("-" * 30)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                mask=mask, center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show strongest correlations with Churn
    churn_corr = corr_matrix['Churn'].abs().sort_values(ascending=False)
    print("\nFeatures most correlated with Churn:")
    print(churn_corr[1:])  # Exclude self-correlation

def segment_analysis(df):
    """Analyze churn by segments"""
    print("\n8. SEGMENT ANALYSIS")
    print("-" * 25)
    
    # Create tenure groups
    df['TenureGroup'] = pd.cut(df['AccountWeeks'], 
                               bins=[0, 52, 104, 156, 300], 
                               labels=['0-1 year', '1-2 years', '2-3 years', '3+ years'])
    
    # Churn rate by tenure groups
    tenure_churn = df.groupby('TenureGroup')['Churn'].agg(['count', 'sum', 'mean']).round(3)
    tenure_churn.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
    print("Churn Analysis by Tenure Groups:")
    print(tenure_churn)
    
    # Contract renewal analysis
    contract_churn = df.groupby('ContractRenewal')['Churn'].agg(['count', 'sum', 'mean']).round(3)
    contract_churn.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
    print("\nChurn Analysis by Contract Renewal:")
    print(contract_churn)
    
    # Data plan analysis
    data_plan_churn = df.groupby('DataPlan')['Churn'].agg(['count', 'sum', 'mean']).round(3)
    data_plan_churn.columns = ['Total_Customers', 'Churned_Customers', 'Churn_Rate']
    print("\nChurn Analysis by Data Plan:")
    print(data_plan_churn)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Tenure groups
    tenure_churn['Churn_Rate'].plot(kind='bar', ax=axes[0,0], color='lightcoral')
    axes[0,0].set_title('Churn Rate by Tenure Groups')
    axes[0,0].set_ylabel('Churn Rate')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Contract renewal
    contract_churn['Churn_Rate'].plot(kind='bar', ax=axes[0,1], color='lightblue')
    axes[0,1].set_title('Churn Rate by Contract Renewal')
    axes[0,1].set_ylabel('Churn Rate')
    axes[0,1].set_xlabel('Contract Renewal (0=No, 1=Yes)')
    axes[0,1].tick_params(axis='x', rotation=0)
    
    # Data plan
    data_plan_churn['Churn_Rate'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
    axes[1,0].set_title('Churn Rate by Data Plan')
    axes[1,0].set_ylabel('Churn Rate')
    axes[1,0].set_xlabel('Data Plan (0=No, 1=Yes)')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # Customer service calls distribution
    sns.boxplot(x='Churn', y='CustServCalls', data=df, ax=axes[1,1])
    axes[1,1].set_title('Customer Service Calls vs Churn')
    axes[1,1].set_xlabel('Churn (0=No, 1=Yes)')
    
    plt.tight_layout()
    plt.savefig('segment_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def outlier_analysis(df):
    """Detect and analyze outliers"""
    print("\n9. OUTLIER ANALYSIS")
    print("-" * 25)
    
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Churn')
    
    # Calculate IQR for outlier detection
    outlier_summary = []
    for feature in numerical_features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_summary.append({
            'Feature': feature,
            'Outlier_Count': len(outliers),
            'Outlier_Percentage': (len(outliers) / len(df)) * 100
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier Summary:")
    print(outlier_df.round(2))

def main():
    """Main function to run all analyses"""
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze target variable
    analyze_target_variable(df)
    
    # Analyze features
    analyze_features(df)
    
    # Analyze feature relationships
    analyze_feature_relationships(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Segment analysis
    segment_analysis(df)
    
    # Outlier analysis
    outlier_analysis(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE! Check the generated PNG files for visualizations.")
    print("="*80)

if __name__ == "__main__":
    main()
