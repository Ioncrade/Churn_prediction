import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load the dataset"""
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_behavioral_features(self, df):
        """Create behavioral features from existing data"""
        print("Creating behavioral features...")
        
        # Average monthly charges (considering account weeks)
        df['AvgMonthlyCharge'] = df['MonthlyCharge'] / (df['AccountWeeks'] / 4.33)  # weeks to months
        
        # Usage intensity features
        df['DayMinsPerCall'] = df['DayMins'] / (df['DayCalls'] + 1)  # Add 1 to avoid division by zero
        df['DataUsagePerWeek'] = df['DataUsage'] / (df['AccountWeeks'] + 1)
        
        # High-value customer indicator
        df['HighValueCustomer'] = (df['MonthlyCharge'] > df['MonthlyCharge'].quantile(0.75)).astype(int)
        
        # Heavy user indicators
        df['HighDayMinutes'] = (df['DayMins'] > df['DayMins'].quantile(0.75)).astype(int)
        df['HighDataUser'] = (df['DataUsage'] > df['DataUsage'].quantile(0.75)).astype(int)
        
        # Customer service interaction indicator
        df['HighServiceCalls'] = (df['CustServCalls'] >= 4).astype(int)
        
        # Overage fee indicator
        df['HasOverageFee'] = (df['OverageFee'] > 0).astype(int)
        
        # Roaming user
        df['RoamingUser'] = (df['RoamMins'] > 0).astype(int)
        
        print(f"Created behavioral features. New shape: {df.shape}")
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features"""
        print("Creating temporal features...")
        
        # Tenure buckets
        df['TenureGroup'] = pd.cut(df['AccountWeeks'], 
                                   bins=[0, 26, 52, 104, 156, 300], 
                                   labels=['0-6m', '6-12m', '1-2y', '2-3y', '3y+'])
        
        # Convert to numeric for modeling
        tenure_mapping = {'0-6m': 1, '6-12m': 2, '1-2y': 3, '2-3y': 4, '3y+': 5}
        df['TenureGroupNumeric'] = df['TenureGroup'].map(tenure_mapping)
        
        # Cyclic features (assuming account weeks represent some seasonality)
        # Convert weeks to a yearly cycle (52 weeks = 1 year)
        df['TenureCycleSin'] = np.sin(2 * np.pi * (df['AccountWeeks'] % 52) / 52)
        df['TenureCycleCos'] = np.cos(2 * np.pi * (df['AccountWeeks'] % 52) / 52)
        
        print(f"Created temporal features. New shape: {df.shape}")
        return df
    
    def handle_categorical_variables(self, df):
        """Handle categorical variables with appropriate encoding"""
        print("Encoding categorical variables...")
        
        # Binary categorical variables - already encoded as 0/1
        binary_features = ['ContractRenewal', 'DataPlan']
        print(f"Binary features: {binary_features}")
        
        # For TenureGroup, we'll use label encoding (already done with TenureGroupNumeric)
        # Drop the original categorical version
        if 'TenureGroup' in df.columns:
            df = df.drop('TenureGroup', axis=1)
        
        print(f"Categorical encoding complete. Final shape: {df.shape}")
        return df
    
    def detect_and_handle_outliers(self, df, method='iqr'):
        """Detect and handle outliers"""
        print("Detecting and handling outliers...")
        
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude target variable and binary features
        exclude_features = ['Churn', 'ContractRenewal', 'DataPlan', 'HighValueCustomer', 
                           'HighDayMinutes', 'HighDataUser', 'HighServiceCalls', 
                           'HasOverageFee', 'RoamingUser', 'TenureGroupNumeric']
        
        features_to_check = [f for f in numerical_features if f not in exclude_features]
        
        outlier_counts = {}
        
        for feature in features_to_check:
            if method == 'iqr':
                Q1 = df[feature].quantile(0.25)
                Q3 = df[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                outliers_mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
                outlier_counts[feature] = outliers_mask.sum()
                
                df[feature] = np.where(df[feature] < lower_bound, lower_bound, df[feature])
                df[feature] = np.where(df[feature] > upper_bound, upper_bound, df[feature])
        
        print("Outliers handled (capped to bounds):")
        for feature, count in outlier_counts.items():
            if count > 0:
                print(f"  {feature}: {count} outliers capped")
        
        return df
    
    def scale_features(self, df, target_column='Churn'):
        """Scale numerical features"""
        print("Scaling features...")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Identify features to scale (exclude binary features)
        binary_features = ['ContractRenewal', 'DataPlan', 'HighValueCustomer', 
                          'HighDayMinutes', 'HighDataUser', 'HighServiceCalls', 
                          'HasOverageFee', 'RoamingUser', 'TenureGroupNumeric']
        
        features_to_scale = [f for f in X.columns if f not in binary_features]
        features_not_to_scale = [f for f in X.columns if f in binary_features]
        
        print(f"Features to scale: {features_to_scale}")
        print(f"Features not to scale: {features_not_to_scale}")
        
        # Scale continuous features
        X_scaled = X.copy()
        if features_to_scale:
            X_scaled[features_to_scale] = self.scaler.fit_transform(X[features_to_scale])
        
        return X_scaled, y
    
    def create_interaction_features(self, df):
        """Create interaction features"""
        print("Creating interaction features...")
        
        # Key interaction features based on domain knowledge
        df['DataPlan_DataUsage'] = df['DataPlan'] * df['DataUsage']
        df['HighServiceCalls_ContractRenewal'] = df['HighServiceCalls'] * df['ContractRenewal']
        df['MonthlyCharge_AccountWeeks'] = df['MonthlyCharge'] * df['AccountWeeks']
        df['OverageFee_DataUsage'] = df['OverageFee'] * df['DataUsage']
        
        print(f"Created interaction features. New shape: {df.shape}")
        return df
    
    def engineer_features(self, file_path):
        """Main feature engineering pipeline"""
        print("="*60)
        print("STARTING FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Load data
        df = self.load_data(file_path)
        
        # Create behavioral features
        df = self.create_behavioral_features(df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Handle categorical variables
        df = self.handle_categorical_variables(df)
        
        # Handle outliers
        df = self.detect_and_handle_outliers(df)
        
        # Scale features and separate X, y
        X, y = self.scale_features(df)
        
        print("="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"Final dataset shape: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Feature names: {list(X.columns)}")
        
        return X, y, df
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets with stratification"""
        print(f"\nSplitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training set churn rate: {y_train.mean():.3f}")
        print(f"Test set churn rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test

def main():
    """Main execution function"""
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Engineer features
    X, y, df_engineered = fe.engineer_features('data/telecom_churn.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = fe.split_data(X, y)
    
    # Save engineered data
    df_engineered.to_csv('data/telecom_churn_engineered.csv', index=False)
    print(f"\nEngineered dataset saved to 'data/telecom_churn_engineered.csv'")
    
    # Save train/test splits
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    
    print("Train and test data saved to 'data/train_data.csv' and 'data/test_data.csv'")
    
    return X_train, X_test, y_train, y_test, fe

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_engineer = main()
