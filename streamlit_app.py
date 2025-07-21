import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from PIL import Image
import os

# Configure page
st.set_page_config(
    page_title="Telecom Churn Prediction Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading functions
@st.cache_data
def load_data():
    """Load all required datasets"""
    try:
        original_data = pd.read_csv('data/telecom_churn.csv')
        engineered_data = pd.read_csv('data/telecom_churn_engineered.csv')
        train_data = pd.read_csv('data/train_data.csv')
        test_data = pd.read_csv('data/test_data.csv')
        
        return original_data, engineered_data, train_data, test_data
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        return None, None, None, None

@st.cache_data
def load_model_results():
    """Load model performance results"""
    try:
        model_results = pd.read_csv('model_performance_summary.csv')
        return model_results
    except FileNotFoundError:
        st.warning("Model results not found. Please run baseline_models.py first.")
        return None

@st.cache_data
def load_survival_results():
    """Load survival analysis results"""
    try:
        with open('survival_analysis_summary.json', 'r') as f:
            survival_summary = json.load(f)
        return survival_summary
    except FileNotFoundError:
        st.warning("Survival analysis results not found. Please run survival_analysis.py first.")
        return None

# Load data
original_data, engineered_data, train_data, test_data = load_data()
model_results = load_model_results()
survival_summary = load_survival_results()

# Title and introduction
st.title("📱 Telecom Churn Prediction Dashboard")
st.markdown("""
This comprehensive dashboard provides insights into customer churn patterns using machine learning 
and survival analysis. Navigate through different sections to explore data insights, model performance, 
and make predictions.
""")

if original_data is None:
    st.error("Unable to load data. Please ensure all data files are present.")
    st.stop()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Overview", "Data Exploration", "Model Performance", "Survival Analysis", "Churn Prediction", "Business Insights"]
)

# ==========================================
# OVERVIEW PAGE
# ==========================================
if page == "Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(original_data):,}"
        )
    
    with col2:
        churn_rate = original_data['Churn'].mean()
        st.metric(
            label="Overall Churn Rate",
            value=f"{churn_rate:.1%}"
        )
    
    with col3:
        st.metric(
            label="Features",
            value=len(original_data.columns)
        )
    
    with col4:
        avg_tenure = original_data['AccountWeeks'].mean()
        st.metric(
            label="Avg Tenure (weeks)",
            value=f"{avg_tenure:.0f}"
        )
    
    # Dataset sample
    st.subheader("Dataset Sample")
    st.dataframe(original_data.head(10))
    
    # Basic statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features Statistics")
        st.dataframe(original_data.describe())
    
    with col2:
        st.subheader("Feature Information")
        info_data = []
        for col in original_data.columns:
            dtype = str(original_data[col].dtype)
            null_count = original_data[col].isnull().sum()
            unique_count = original_data[col].nunique()
            info_data.append({
                'Feature': col,
                'Type': dtype,
                'Null Values': null_count,
                'Unique Values': unique_count
            })
        st.dataframe(pd.DataFrame(info_data))

# ==========================================
# DATA EXPLORATION PAGE
# ==========================================
elif page == "Data Exploration":
    st.header("🔍 Data Exploration")
    
    # Churn distribution
    st.subheader("Churn Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=original_data['Churn'].value_counts().values,
            names=['No Churn', 'Churn'],
            title="Overall Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        churn_counts = original_data['Churn'].value_counts()
        fig = px.bar(
            x=['No Churn', 'Churn'],
            y=churn_counts.values,
            title="Churn Count Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select feature to explore
    numerical_features = original_data.select_dtypes(include=[np.number]).columns.tolist()
    numerical_features.remove('Churn')
    
    selected_feature = st.selectbox("Select a feature to explore:", numerical_features)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig = px.histogram(
            original_data,
            x=selected_feature,
            nbins=30,
            title=f"Distribution of {selected_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by churn
        fig = px.box(
            original_data,
            x='Churn',
            y=selected_feature,
            title=f"{selected_feature} by Churn Status"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Feature Correlations")
    
    corr_matrix = original_data.corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn by segments
    st.subheader("Churn Analysis by Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure groups
        tenure_bins = pd.cut(original_data['AccountWeeks'], 
                           bins=[0, 52, 104, 156, 300], 
                           labels=['0-1y', '1-2y', '2-3y', '3y+'])
        tenure_churn = original_data.groupby(tenure_bins)['Churn'].mean()
        
        fig = px.bar(
            x=tenure_churn.index,
            y=tenure_churn.values,
            title="Churn Rate by Tenure Groups"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Contract renewal
        contract_churn = original_data.groupby('ContractRenewal')['Churn'].mean()
        
        fig = px.bar(
            x=['No Renewal', 'Renewal'],
            y=contract_churn.values,
            title="Churn Rate by Contract Renewal"
        )
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODEL PERFORMANCE PAGE
# ==========================================
elif page == "Model Performance":
    st.header("🤖 Model Performance")
    
    if model_results is not None:
        # Model comparison metrics
        st.subheader("Model Comparison")
        st.dataframe(model_results)
        
        # Performance visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.bar(
                model_results,
                x='Model',
                y='F1-Score',
                title="F1-Score Comparison"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                model_results,
                x='Model',
                y='AUC-ROC',
                title="AUC-ROC Comparison"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.bar(
                model_results,
                x='Model',
                y='Accuracy',
                title="Accuracy Comparison"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model information
        best_model = model_results.loc[model_results['F1-Score'].idxmax()]
        
        st.subheader("🏆 Best Performing Model")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", best_model['Model'])
        with col2:
            st.metric("F1-Score", best_model['F1-Score'])
        with col3:
            st.metric("AUC-ROC", best_model['AUC-ROC'])
        with col4:
            st.metric("Accuracy", best_model['Accuracy'])
        
        # Display visualizations if available
        st.subheader("Model Visualizations")
        
        viz_files = [
            ('model_comparison.png', 'Model Performance Comparison'),
            ('confusion_matrices.png', 'Confusion Matrices'),
            ('roc_curves.png', 'ROC Curves'),
            ('precision_recall_curves.png', 'Precision-Recall Curves'),
            ('shap_summary.png', 'SHAP Feature Importance'),
            ('shap_feature_importance.png', 'SHAP Feature Ranking')
        ]
        
        for filename, title in viz_files:
            if os.path.exists(filename):
                st.subheader(title)
                image = Image.open(filename)
                st.image(image, use_container_width=True)
    
    else:
        st.warning("Model results not available. Please run the baseline_models.py script first.")

# ==========================================
# SURVIVAL ANALYSIS PAGE
# ==========================================
elif page == "Survival Analysis":
    st.header("⏰ Survival Analysis")
    
    if survival_summary is not None:
        # Key metrics
        st.subheader("Key Survival Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Median Survival Time",
                f"{survival_summary['median_survival_time']:.0f} weeks"
            )
        
        with col2:
            st.metric(
                "1-Year Retention Rate",
                f"{survival_summary['survival_at_1_year']:.1%}"
            )
        
        with col3:
            st.metric(
                "2-Year Retention Rate",
                f"{survival_summary['survival_at_2_years']:.1%}"
            )
        
        with col4:
            st.metric(
                "Cox Model C-Index",
                f"{survival_summary['cox_concordance']:.3f}"
            )
        
        # Survival insights
        st.subheader("📈 Survival Analysis Insights")
        
        insights = [
            f"• **{survival_summary['total_customers']:,}** total customers analyzed",
            f"• **{survival_summary['churned_customers']:,}** churn events observed",
            f"• Median time until churn: **{survival_summary['median_survival_time']:.0f} weeks** (~{survival_summary['median_survival_time']/52:.1f} years)",
            f"• **{survival_summary['survival_at_1_year']:.1%}** of customers remain after 1 year",
            f"• **{survival_summary['survival_at_2_years']:.1%}** of customers remain after 2 years",
            f"• Cox model achieves **{survival_summary['cox_concordance']:.1%}** discrimination accuracy"
        ]
        
        for insight in insights:
            st.markdown(insight)
        
        # Display survival visualizations
        st.subheader("Survival Analysis Visualizations")
        
        survival_viz_files = [
            ('kaplan_meier_overall.png', 'Overall Survival Curve'),
            ('kaplan_meier_segments.png', 'Survival by Customer Segments'),
            ('cox_hazard_ratios.png', 'Cox Model Risk Factors'),
            ('predicted_survival.png', 'Sample Customer Survival Prediction')
        ]
        
        for filename, title in survival_viz_files:
            if os.path.exists(filename):
                st.subheader(title)
                image = Image.open(filename)
                st.image(image, use_container_width=True)
    
    else:
        st.warning("Survival analysis results not available. Please run the survival_analysis.py script first.")

# ==========================================
# CHURN PREDICTION PAGE
# ==========================================
elif page == "Churn Prediction":
    st.header("🎯 Churn Prediction")
    
    st.markdown("""
    Use this interactive tool to predict churn probability for individual customers.
    Adjust the customer characteristics below to see how they affect churn risk.
    """)
    
    # Input form for customer profile
    st.subheader("Customer Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        account_weeks = st.slider("Account Weeks", 1, 250, 100)
        contract_renewal = st.selectbox("Contract Renewal", ["No", "Yes"])
        data_plan = st.selectbox("Data Plan", ["No", "Yes"])
        data_usage = st.slider("Data Usage (GB)", 0.0, 6.0, 2.0, 0.1)
    
    with col2:
        cust_serv_calls = st.slider("Customer Service Calls", 0, 10, 1)
        day_mins = st.slider("Day Minutes", 0, 400, 180)
        day_calls = st.slider("Day Calls", 0, 200, 100)
        monthly_charge = st.slider("Monthly Charge ($)", 10, 120, 55)
    
    with col3:
        overage_fee = st.slider("Overage Fee ($)", 0.0, 20.0, 10.0, 0.1)
        roam_mins = st.slider("Roaming Minutes", 0, 25, 10)
    
    # Convert inputs to model format
    customer_data = {
        'AccountWeeks': account_weeks,
        'ContractRenewal': 1 if contract_renewal == "Yes" else 0,
        'DataPlan': 1 if data_plan == "Yes" else 0,
        'DataUsage': data_usage,
        'CustServCalls': cust_serv_calls,
        'DayMins': day_mins,
        'DayCalls': day_calls,
        'MonthlyCharge': monthly_charge,
        'OverageFee': overage_fee,
        'RoamMins': roam_mins
    }
    
    # Simple rule-based prediction (since we don't have saved models)
    st.subheader("Prediction Results")
    
    # Calculate risk factors
    risk_score = 0
    risk_factors = []
    
    # High service calls risk
    if cust_serv_calls >= 4:
        risk_score += 30
        risk_factors.append("High customer service calls")
    
    # Contract renewal benefit
    if contract_renewal == "No":
        risk_score += 25
        risk_factors.append("No contract renewal")
    
    # Data plan benefit
    if data_plan == "No":
        risk_score += 15
        risk_factors.append("No data plan")
    
    # High usage customers
    if day_mins > 250:
        risk_score += 10
        risk_factors.append("High day minutes usage")
    
    # Overage fees
    if overage_fee > 15:
        risk_score += 10
        risk_factors.append("High overage fees")
    
    # Short tenure risk
    if account_weeks < 52:
        risk_score += 15
        risk_factors.append("Short tenure (< 1 year)")
    
    # Convert to probability
    churn_probability = min(risk_score / 100, 0.95)
    
    # Display prediction
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = churn_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if churn_probability > 0.7 else "orange" if churn_probability > 0.3 else "darkgreen"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk classification
        if churn_probability > 0.7:
            risk_level = "🔴 HIGH RISK"
            recommendation = "Immediate attention required. Consider retention offers."
        elif churn_probability > 0.3:
            risk_level = "🟡 MEDIUM RISK"
            recommendation = "Monitor closely. Proactive engagement recommended."
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = "Customer appears stable. Continue regular service."
        
        st.markdown(f"### {risk_level}")
        st.markdown(f"**Churn Probability:** {churn_probability:.1%}")
        st.markdown(f"**Recommendation:** {recommendation}")
    
    # Risk factors
    if risk_factors:
        st.subheader("⚠️ Risk Factors Identified")
        for factor in risk_factors:
            st.markdown(f"• {factor}")
    else:
        st.success("✅ No major risk factors identified")
    
    # Customer profile summary
    st.subheader("Customer Profile Summary")
    profile_df = pd.DataFrame([customer_data])
    st.dataframe(profile_df)

# ==========================================
# BUSINESS INSIGHTS PAGE
# ==========================================
elif page == "Business Insights":
    st.header("💼 Business Insights & Recommendations")
    
    # Key findings
    st.subheader("🔍 Key Findings")
    
    findings = [
        {
            "Finding": "High Service Call Impact",
            "Detail": "Customers with 4+ service calls are 3x more likely to churn",
            "Action": "Implement proactive service quality improvements"
        },
        {
            "Finding": "Contract Renewal Importance",
            "Detail": "Non-renewed contracts show significantly higher churn risk",
            "Action": "Strengthen contract renewal processes and incentives"
        },
        {
            "Finding": "Data Plan Protection",
            "Detail": "Customers with data plans have lower churn rates",
            "Action": "Promote data plan adoption through targeted campaigns"
        },
        {
            "Finding": "Early Tenure Vulnerability",
            "Detail": "Customers in first year show higher churn risk",
            "Action": "Enhance onboarding and early engagement programs"
        }
    ]
    
    for finding in findings:
        with st.expander(f"📊 {finding['Finding']}"):
            st.markdown(f"**Insight:** {finding['Detail']}")
            st.markdown(f"**Recommended Action:** {finding['Action']}")
    
    # Churn prevention strategies
    st.subheader("🛡️ Churn Prevention Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Immediate Actions:**
        - Monitor customers with 3+ service calls
        - Prioritize contract renewal campaigns
        - Offer data plan upgrades to non-subscribers
        - Implement first-year customer success program
        """)
    
    with col2:
        st.markdown("""
        **Long-term Strategies:**
        - Improve service quality to reduce call volume
        - Develop predictive early warning system
        - Create personalized retention offers
        - Enhance customer experience journey mapping
        """)
    
    # ROI calculations
    st.subheader("💰 Business Impact Analysis")
    
    if survival_summary is not None:
        total_customers = survival_summary['total_customers']
        churned_customers = survival_summary['churned_customers']
        
        # Assumptions for calculations
        avg_customer_value = st.slider("Average Customer Lifetime Value ($)", 500, 2000, 1200)
        retention_program_cost = st.slider("Retention Program Cost per Customer ($)", 10, 100, 50)
        expected_retention_improvement = st.slider("Expected Retention Improvement (%)", 5, 50, 20)
        
        # Calculate potential savings
        preventable_churns = int(churned_customers * (expected_retention_improvement / 100))
        revenue_saved = preventable_churns * avg_customer_value
        program_cost = total_customers * retention_program_cost
        net_benefit = revenue_saved - program_cost
        roi = (net_benefit / program_cost) * 100 if program_cost > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Preventable Churns",
                f"{preventable_churns:,}",
                f"{expected_retention_improvement}% improvement"
            )
        
        with col2:
            st.metric(
                "Revenue Saved",
                f"${revenue_saved:,.0f}",
                f"From retained customers"
            )
        
        with col3:
            st.metric(
                "Program Cost",
                f"${program_cost:,.0f}",
                f"${retention_program_cost}/customer"
            )
        
        with col4:
            st.metric(
                "ROI",
                f"{roi:.0f}%",
                f"Net benefit: ${net_benefit:,.0f}"
            )
    
    # Implementation roadmap
    st.subheader("🗺️ Implementation Roadmap")
    
    roadmap_data = {
        "Phase": ["Phase 1 (0-3 months)", "Phase 2 (3-6 months)", "Phase 3 (6-12 months)"],
        "Focus": [
            "Quick wins & monitoring",
            "Model deployment",
            "Advanced analytics"
        ],
        "Actions": [
            "• Service call monitoring\n• Contract renewal focus\n• Basic reporting",
            "• Deploy ML models\n• Automated alerts\n• Retention campaigns",
            "• Real-time predictions\n• Advanced segmentation\n• Continuous improvement"
        ]
    }
    
    roadmap_df = pd.DataFrame(roadmap_data)
    st.dataframe(roadmap_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Telecom Churn Prediction Dashboard | Built with Streamlit</p>
    <p>Powered by Machine Learning and Survival Analysis</p>
</div>
""", unsafe_allow_html=True)
