import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, f1_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_score = 0
        
    def load_data(self):
        """Load preprocessed training and test data"""
        print("Loading preprocessed data...")
        train_data = pd.read_csv('data/train_data.csv')
        test_data = pd.read_csv('data/test_data.csv')
        
        # Separate features and target
        X_train = train_data.drop('Churn', axis=1)
        y_train = train_data['Churn']
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Training churn rate: {y_train.mean():.3f}")
        print(f"Test churn rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_class_imbalance(self, X_train, y_train, method='class_weight'):
        """Handle class imbalance using various techniques"""
        print(f"Handling class imbalance using {method}...")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {X_train_balanced.shape[0]} samples after resampling")
            return X_train_balanced, y_train_balanced
        
        elif method == 'undersampling':
            undersampler = RandomUnderSampler(random_state=42)
            X_train_balanced, y_train_balanced = undersampler.fit_resample(X_train, y_train)
            print(f"Undersampling applied: {X_train_balanced.shape[0]} samples after resampling")
            return X_train_balanced, y_train_balanced
        
        elif method == 'smoteenn':
            smoteenn = SMOTEENN(random_state=42)
            X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)
            print(f"SMOTEENN applied: {X_train_balanced.shape[0]} samples after resampling")
            return X_train_balanced, y_train_balanced
        
        else:  # class_weight
            print("Using class weights in model parameters")
            return X_train, y_train
    
    def initialize_models(self, class_weight_method=True):
        """Initialize baseline models"""
        print("Initializing baseline models...")
        
        # Calculate class weights
        if class_weight_method:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique([0, 1]), 
                y=[0]*2850 + [1]*483  # Approximate class distribution
            )
            class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        else:
            class_weight_dict = None
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                class_weight='balanced' if class_weight_method else None,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced' if class_weight_method else None,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=class_weights[1]/class_weights[0] if class_weight_method else 1,
                n_jobs=-1,
                verbosity=0
            ),
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                class_weight='balanced' if class_weight_method else None,
                n_jobs=-1,
                verbosity=-1
            ),
            'AdaBoost': AdaBoostClassifier(
                random_state=42,
                n_estimators=100
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a single model and return metrics"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'precision_0': class_report['0']['precision'],
            'recall_0': class_report['0']['recall'],
            'precision_1': class_report['1']['precision'],
            'recall_1': class_report['1']['recall'],
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train and evaluate all baseline models"""
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING BASELINE MODELS")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                results = self.evaluate_model(model, model_name, X_test, y_test)
                self.model_results[model_name] = results
                
                # Print results
                print(f"{model_name} Results:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  F1-Score: {results['f1_score']:.4f}")
                print(f"  AUC-ROC: {results['auc_roc']:.4f}")
                print(f"  Precision (Class 1): {results['precision_1']:.4f}")
                print(f"  Recall (Class 1): {results['recall_1']:.4f}")
                
                # Track best model based on F1-score
                if results['f1_score'] > self.best_score:
                    self.best_score = results['f1_score']
                    self.best_model = model
                    self.best_model_name = model_name
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        print(f"\nBest Model: {self.best_model_name} (F1-Score: {self.best_score:.4f})")
    
    def plot_model_comparison(self):
        """Plot comparison of model performance"""
        print("\nCreating model comparison plots...")
        
        # Prepare data for plotting
        model_names = list(self.model_results.keys())
        accuracies = [self.model_results[name]['accuracy'] for name in model_names]
        f1_scores = [self.model_results[name]['f1_score'] for name in model_names]
        auc_scores = [self.model_results[name]['auc_roc'] for name in model_names]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Accuracy comparison
        axes[0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim([0, 1])
        
        # F1-score comparison
        axes[1].bar(model_names, f1_scores, color='lightgreen', alpha=0.7)
        axes[1].set_title('Model F1-Score Comparison')
        axes[1].set_ylabel('F1-Score')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim([0, 1])
        
        # AUC-ROC comparison
        axes[2].bar(model_names, auc_scores, color='lightcoral', alpha=0.7)
        axes[2].set_title('Model AUC-ROC Comparison')
        axes[2].set_ylabel('AUC-ROC')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("Creating confusion matrix plots...")
        
        n_models = len(self.model_results)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            if idx < len(axes):
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{model_name}\nConfusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
        
        # Remove empty subplots
        for idx in range(n_models, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        print("Creating ROC curve plots...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.model_results.items():
            y_pred_proba = results['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = results['auc_roc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_test):
        """Plot Precision-Recall curves for all models"""
        print("Creating Precision-Recall curve plots...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, results in self.model_results.items():
            y_pred_proba = results['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{model_name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_results_summary(self):
        """Create a summary table of all model results"""
        print("Creating results summary...")
        
        summary_data = []
        for model_name, results in self.model_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'AUC-ROC': f"{results['auc_roc']:.4f}",
                'Precision (Churn)': f"{results['precision_1']:.4f}",
                'Recall (Churn)': f"{results['recall_1']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print("\nModel Performance Summary:")
        print(summary_df.to_string(index=False))
        
        # Save to CSV
        summary_df.to_csv('model_performance_summary.csv', index=False)
        print("\nSummary saved to 'model_performance_summary.csv'")
        
        return summary_df
    
    def explain_best_model(self, X_train, feature_names):
        """Use SHAP to explain the best model"""
        print(f"\nCreating SHAP explanations for {self.best_model_name}...")
        
        try:
            # Create SHAP explainer
            if self.best_model_name in ['XGBoost', 'LightGBM']:
                explainer = shap.TreeExplainer(self.best_model)
            else:
                explainer = shap.Explainer(self.best_model, X_train)
            
            # Calculate SHAP values on a subset for efficiency
            sample_size = min(1000, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42)
            shap_values = explainer.shap_values(X_sample)
            
            # If binary classification with 2 output classes, use positive class
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                            plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating SHAP explanations: {str(e)}")

def main():
    """Main execution function"""
    print("="*80)
    print("CHURN PREDICTION - BASELINE MODEL DEVELOPMENT")
    print("="*80)
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Initialize models
    trainer.initialize_models(class_weight_method=True)
    
    # Train and evaluate models
    trainer.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    trainer.plot_model_comparison()
    trainer.plot_confusion_matrices()
    trainer.plot_roc_curves(y_test)
    trainer.plot_precision_recall_curves(y_test)
    
    # Create results summary
    summary_df = trainer.create_results_summary()
    
    # Explain best model using SHAP
    feature_names = X_train.columns.tolist()
    trainer.explain_best_model(X_train, feature_names)
    
    print("\n" + "="*80)
    print("BASELINE MODEL DEVELOPMENT COMPLETE!")
    print("="*80)
    print(f"Best performing model: {trainer.best_model_name}")
    print(f"Best F1-Score: {trainer.best_score:.4f}")
    print("\nGenerated files:")
    print("- model_comparison.png")
    print("- confusion_matrices.png") 
    print("- roc_curves.png")
    print("- precision_recall_curves.png")
    print("- model_performance_summary.csv")
    print("- shap_summary.png")
    print("- shap_feature_importance.png")
    
    return trainer

if __name__ == "__main__":
    trainer = main()
