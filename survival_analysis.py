import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from lifelines.plotting import plot_lifetimes
import warnings
warnings.filterwarnings('ignore')

class ChurnSurvivalAnalyzer:
    def __init__(self):
        self.cox_model = CoxPHFitter()
        self.kmf = KaplanMeierFitter()
        self.survival_data = None
        self.cox_results = None
        
    def load_and_prepare_data(self):
        """Load data and prepare for survival analysis"""
        print("="*80)
        print("CHURN SURVIVAL ANALYSIS")
        print("="*80)
        
        # Load the engineered dataset
        df = pd.read_csv('data/telecom_churn_engineered.csv')
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # For survival analysis, we need:
        # - Duration: Time until event (AccountWeeks)
        # - Event: Whether churn occurred (Churn)
        
        # Create survival dataset
        self.survival_data = df.copy()
        
        # Rename columns for clarity in survival analysis
        self.survival_data['duration'] = self.survival_data['AccountWeeks']
        self.survival_data['event'] = self.survival_data['Churn']
        
        print(f"Survival data prepared:")
        print(f"- Total customers: {len(self.survival_data)}")
        print(f"- Churned customers (events): {self.survival_data['event'].sum()}")
        print(f"- Censored customers: {len(self.survival_data) - self.survival_data['event'].sum()}")
        print(f"- Average duration: {self.survival_data['duration'].mean():.2f} weeks")
        print(f"- Duration range: {self.survival_data['duration'].min():.0f} - {self.survival_data['duration'].max():.0f} weeks")
        
        return self.survival_data
    
    def kaplan_meier_analysis(self):
        """Perform Kaplan-Meier survival analysis"""
        print("\n1. KAPLAN-MEIER SURVIVAL ANALYSIS")
        print("-" * 45)
        
        # Overall survival curve
        self.kmf.fit(durations=self.survival_data['duration'], 
                     event_observed=self.survival_data['event'], 
                     label='All Customers')
        
        # Plot overall survival curve
        plt.figure(figsize=(12, 8))
        self.kmf.plot_survival_function()
        plt.title('Kaplan-Meier Survival Curve - Overall Customer Base')
        plt.xlabel('Time (Weeks)')
        plt.ylabel('Survival Probability (1 - Churn Rate)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('kaplan_meier_overall.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print survival statistics
        print(f"Median survival time: {self.kmf.median_survival_time_:.2f} weeks")
        print(f"Survival probability at 52 weeks (1 year): {self.kmf.survival_function_at_times(52).values[0]:.4f}")
        print(f"Survival probability at 104 weeks (2 years): {self.kmf.survival_function_at_times(104).values[0]:.4f}")
        
        return self.kmf
    
    def kaplan_meier_by_groups(self):
        """Analyze survival by different customer segments"""
        print("\n2. SURVIVAL ANALYSIS BY CUSTOMER SEGMENTS")
        print("-" * 50)
        
        # Create figure with subplots for different groupings
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Group by Contract Renewal
        ax1 = axes[0, 0]
        groups = self.survival_data.groupby('ContractRenewal')
        for name, group in groups:
            kmf_temp = KaplanMeierFitter()
            label = f"Contract Renewal: {'Yes' if name == 1 else 'No'}"
            kmf_temp.fit(group['duration'], group['event'], label=label)
            kmf_temp.plot_survival_function(ax=ax1)
        
        ax1.set_title('Survival by Contract Renewal Status')
        ax1.set_xlabel('Time (Weeks)')
        ax1.set_ylabel('Survival Probability')
        ax1.grid(True, alpha=0.3)
        
        # Group by Data Plan
        ax2 = axes[0, 1]
        groups = self.survival_data.groupby('DataPlan')
        for name, group in groups:
            kmf_temp = KaplanMeierFitter()
            label = f"Data Plan: {'Yes' if name == 1 else 'No'}"
            kmf_temp.fit(group['duration'], group['event'], label=label)
            kmf_temp.plot_survival_function(ax=ax2)
        
        ax2.set_title('Survival by Data Plan')
        ax2.set_xlabel('Time (Weeks)')
        ax2.set_ylabel('Survival Probability')
        ax2.grid(True, alpha=0.3)
        
        # Group by High Service Calls
        ax3 = axes[1, 0]
        groups = self.survival_data.groupby('HighServiceCalls')
        for name, group in groups:
            kmf_temp = KaplanMeierFitter()
            label = f"High Service Calls: {'Yes' if name == 1 else 'No'}"
            kmf_temp.fit(group['duration'], group['event'], label=label)
            kmf_temp.plot_survival_function(ax=ax3)
        
        ax3.set_title('Survival by Service Call Frequency')
        ax3.set_xlabel('Time (Weeks)')
        ax3.set_ylabel('Survival Probability')
        ax3.grid(True, alpha=0.3)
        
        # Group by Tenure Groups
        ax4 = axes[1, 1]
        # Create tenure groups for visualization
        tenure_bins = pd.cut(self.survival_data['duration'], 
                           bins=[0, 52, 104, 156, 300], 
                           labels=['0-1y', '1-2y', '2-3y', '3y+'])
        
        for name in tenure_bins.unique():
            if pd.notna(name):
                mask = tenure_bins == name
                group = self.survival_data[mask]
                if len(group) > 0:
                    kmf_temp = KaplanMeierFitter()
                    kmf_temp.fit(group['duration'], group['event'], label=f"Tenure: {name}")
                    kmf_temp.plot_survival_function(ax=ax4)
        
        ax4.set_title('Survival by Tenure Groups')
        ax4.set_xlabel('Time (Weeks)')
        ax4.set_ylabel('Survival Probability')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kaplan_meier_segments.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Perform log-rank tests
        self.perform_logrank_tests()
    
    def perform_logrank_tests(self):
        """Perform log-rank tests to compare survival curves"""
        print("\n3. LOG-RANK TESTS (Statistical Significance)")
        print("-" * 50)
        
        # Test Contract Renewal groups
        group_0 = self.survival_data[self.survival_data['ContractRenewal'] == 0]
        group_1 = self.survival_data[self.survival_data['ContractRenewal'] == 1]
        
        results = logrank_test(group_0['duration'], group_1['duration'], 
                              group_0['event'], group_1['event'])
        
        print("Contract Renewal Comparison:")
        print(f"  Test statistic: {results.test_statistic:.4f}")
        print(f"  p-value: {results.p_value:.6f}")
        print(f"  Significant difference: {'Yes' if results.p_value < 0.05 else 'No'}")
        
        # Test Data Plan groups
        group_0 = self.survival_data[self.survival_data['DataPlan'] == 0]
        group_1 = self.survival_data[self.survival_data['DataPlan'] == 1]
        
        results = logrank_test(group_0['duration'], group_1['duration'], 
                              group_0['event'], group_1['event'])
        
        print("\nData Plan Comparison:")
        print(f"  Test statistic: {results.test_statistic:.4f}")
        print(f"  p-value: {results.p_value:.6f}")
        print(f"  Significant difference: {'Yes' if results.p_value < 0.05 else 'No'}")
        
        # Test High Service Calls groups
        group_0 = self.survival_data[self.survival_data['HighServiceCalls'] == 0]
        group_1 = self.survival_data[self.survival_data['HighServiceCalls'] == 1]
        
        results = logrank_test(group_0['duration'], group_1['duration'], 
                              group_0['event'], group_1['event'])
        
        print("\nHigh Service Calls Comparison:")
        print(f"  Test statistic: {results.test_statistic:.4f}")
        print(f"  p-value: {results.p_value:.6f}")
        print(f"  Significant difference: {'Yes' if results.p_value < 0.05 else 'No'}")
    
    def cox_proportional_hazards_analysis(self):
        """Perform Cox Proportional Hazards regression"""
        print("\n4. COX PROPORTIONAL HAZARDS MODEL")
        print("-" * 45)
        
        # Select key features for Cox model (avoid multicollinearity)
        key_features = [
            'ContractRenewal', 'DataPlan', 'DataUsage', 'CustServCalls',
            'DayMins', 'MonthlyCharge', 'OverageFee', 'RoamMins',
            'HighValueCustomer', 'HighServiceCalls', 'HasOverageFee'
        ]
        
        # Prepare data for Cox model
        cox_data = self.survival_data[['duration', 'event'] + key_features].copy()
        
        # Remove any rows with missing values
        cox_data = cox_data.dropna()
        
        # Check for infinite values and replace them
        cox_data = cox_data.replace([np.inf, -np.inf], np.nan).dropna()
        
        print(f"Cox model data shape: {cox_data.shape}")
        
        # Fit Cox model with penalty to avoid convergence issues
        print("Fitting Cox Proportional Hazards model...")
        try:
            self.cox_model.fit(cox_data, duration_col='duration', event_col='event', 
                              robust=True, step_size=0.5)
        except Exception as e:
            print(f"Error with robust fitting, trying penalized regression: {e}")
            # Try with penalized regression
            self.cox_model = CoxPHFitter(penalizer=0.1)
            self.cox_model.fit(cox_data, duration_col='duration', event_col='event')
        
        # Print model summary
        print("\nCox Model Summary:")
        print(self.cox_model.summary)
        
        # Get hazard ratios
        hazard_ratios = np.exp(self.cox_model.params_)
        
        # Create hazard ratios dataframe
        hr_df = pd.DataFrame({
            'Feature': hazard_ratios.index,
            'Hazard_Ratio': hazard_ratios.values,
            'P_Value': self.cox_model.summary['p'].values,
            'Significant': self.cox_model.summary['p'].values < 0.05
        }).sort_values('Hazard_Ratio', ascending=False)
        
        print("\nTop Features by Hazard Ratio (Risk Factors):")
        print(hr_df.head(10).to_string(index=False))
        
        # Plot hazard ratios
        self.plot_hazard_ratios(hr_df)
        
        # Model performance
        concordance = self.cox_model.concordance_index_
        print(f"\nModel Performance:")
        print(f"  Concordance Index: {concordance:.4f}")
        print(f"  (0.5 = random, 1.0 = perfect discrimination)")
        
        self.cox_results = {
            'model': self.cox_model,
            'hazard_ratios': hr_df,
            'concordance': concordance
        }
        
        return self.cox_results
    
    def plot_hazard_ratios(self, hr_df):
        """Plot hazard ratios"""
        # Filter significant features
        significant_features = hr_df[hr_df['Significant'] == True].head(15)
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if hr > 1 else 'blue' for hr in significant_features['Hazard_Ratio']]
        
        plt.barh(range(len(significant_features)), significant_features['Hazard_Ratio'], 
                color=colors, alpha=0.7)
        plt.yticks(range(len(significant_features)), significant_features['Feature'])
        plt.xlabel('Hazard Ratio')
        plt.title('Significant Risk Factors (Hazard Ratios)\nRed = Increases Risk, Blue = Decreases Risk')
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        # Add text annotations
        for i, (hr, p_val) in enumerate(zip(significant_features['Hazard_Ratio'], 
                                           significant_features['P_Value'])):
            plt.text(hr + 0.05, i, f'HR: {hr:.3f}\np: {p_val:.4f}', 
                    va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('cox_hazard_ratios.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_survival_probability(self, customer_profile=None):
        """Predict survival probability for a customer profile"""
        print("\n5. SURVIVAL PROBABILITY PREDICTIONS")
        print("-" * 45)
        
        if customer_profile is None:
            # Use a sample customer profile
            customer_profile = self.survival_data.iloc[0:1].copy()
            customer_profile = customer_profile.drop(['duration', 'event'], axis=1)
            print("Using sample customer profile for demonstration...")
        
        # Predict survival function
        survival_func = self.cox_model.predict_survival_function(customer_profile)
        
        # Plot survival probability over time
        plt.figure(figsize=(12, 6))
        survival_func.plot()
        plt.title('Predicted Survival Probability Over Time')
        plt.xlabel('Time (Weeks)')
        plt.ylabel('Survival Probability')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predicted_survival.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate risk scores
        risk_scores = self.cox_model.predict_partial_hazard(customer_profile)
        print(f"Risk Score: {risk_scores.values[0]:.4f}")
        print(f"(Higher scores indicate higher churn risk)")
        
        return survival_func, risk_scores
    
    def create_survival_summary(self):
        """Create summary of survival analysis results"""
        print("\n6. SURVIVAL ANALYSIS SUMMARY")
        print("-" * 40)
        
        summary = {
            'total_customers': len(self.survival_data),
            'churned_customers': self.survival_data['event'].sum(),
            'median_survival_time': self.kmf.median_survival_time_,
            'survival_at_1_year': self.kmf.survival_function_at_times(52).values[0],
            'survival_at_2_years': self.kmf.survival_function_at_times(104).values[0],
            'cox_concordance': self.cox_results['concordance'] if self.cox_results else None
        }
        
        print("Key Insights:")
        print(f"  • Total customers analyzed: {summary['total_customers']:,}")
        print(f"  • Churn events observed: {summary['churned_customers']:,}")
        print(f"  • Median survival time: {summary['median_survival_time']:.1f} weeks")
        print(f"  • 1-year retention rate: {summary['survival_at_1_year']:.1%}")
        print(f"  • 2-year retention rate: {summary['survival_at_2_years']:.1%}")
        
        if summary['cox_concordance']:
            print(f"  • Cox model discrimination: {summary['cox_concordance']:.3f}")
        
        # Save summary
        import json
        with open('survival_analysis_summary.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_summary = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                          for k, v in summary.items() if v is not None}
            json.dump(json_summary, f, indent=2)
        
        print("\nSurvival analysis summary saved to 'survival_analysis_summary.json'")
        return summary

def main():
    """Main execution function"""
    print("Starting Churn Survival Analysis...")
    
    # Initialize analyzer
    analyzer = ChurnSurvivalAnalyzer()
    
    # Load and prepare data
    survival_data = analyzer.load_and_prepare_data()
    
    # Kaplan-Meier analysis
    kmf = analyzer.kaplan_meier_analysis()
    
    # Segment analysis
    analyzer.kaplan_meier_by_groups()
    
    # Cox proportional hazards analysis
    cox_results = analyzer.cox_proportional_hazards_analysis()
    
    # Predict survival for sample customer
    survival_func, risk_scores = analyzer.predict_survival_probability()
    
    # Create summary
    summary = analyzer.create_survival_summary()
    
    print("\n" + "="*80)
    print("SURVIVAL ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("- kaplan_meier_overall.png")
    print("- kaplan_meier_segments.png")
    print("- cox_hazard_ratios.png")
    print("- predicted_survival.png")
    print("- survival_analysis_summary.json")
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()
