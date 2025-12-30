"""
Streamlit Application for Automated Feature Engineering Engine
A user-friendly interface for automatic feature generation and selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io

from auto_feature_engineer import AutoFeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Automated Feature Engineering Engine",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'afe' not in st.session_state:
    st.session_state.afe = None
if 'X_train_fe' not in st.session_state:
    st.session_state.X_train_fe = None
if 'X_test_fe' not in st.session_state:
    st.session_state.X_test_fe = None

def load_sample_data():
    """Load the breast cancer sample dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y, data.target_names


# Header
st.markdown('<h1 class="main-header">🔧 Automated Feature Engineering Engine</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Dataset Selection
st.sidebar.subheader("📊 Dataset")
dataset_option = st.sidebar.radio(
    "Choose dataset:",
    ["Sample Dataset (Breast Cancer)", "Upload CSV File"],
    index=0
)

# Feature Engineering Parameters
st.sidebar.subheader("🔧 Feature Engineering Parameters")
k_features = st.sidebar.slider(
    "Number of features to select (k):",
    min_value=5,
    max_value=100,
    value=25,
    step=5,
    help="Select top-k features based on mutual information score"
)

task_type = st.sidebar.selectbox(
    "Task Type:",
    ["classification", "regression"],
    index=0,
    help="Select classification or regression task"
)

poly_degree = st.sidebar.slider(
    "Polynomial Degree:",
    min_value=2,
    max_value=3,
    value=2,
    help="Degree of polynomial features (x², x³, interactions)"
)

n_bins = st.sidebar.slider(
    "Number of Bins:",
    min_value=3,
    max_value=10,
    value=5,
    help="Number of bins for feature discretization"
)

# Model Parameters
st.sidebar.subheader("🤖 Model Parameters")
test_size = st.sidebar.slider(
    "Test Size:",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
    help="Proportion of dataset for testing"
)

run_benchmark_option = st.sidebar.checkbox(
    "Run Model Benchmark",
    value=True,
    help="Compare model performance with raw vs engineered features"
)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📖 About")
    st.markdown("""
    This tool automatically generates, scores, and selects the best features for your tabular dataset.
    
    **Features:**
    - 🎯 **Target Encoding**: Encodes categorical variables using target statistics
    - 📈 **Polynomial Features**: Generates polynomial and interaction features
    - 📊 **Feature Binning**: Discretizes continuous features into bins
    - 🎯 **Intelligent Selection**: Uses mutual information to rank and select top-k features
    - 🤖 **Model Integration**: Seamlessly works with scikit-learn models
    """)

with col2:
    st.markdown("### 📋 Project Structure")
    st.code("""
    auto_feature_engineer.py
    ├── AutoFeatureEngineer
    ├── FeatureGenerator
    └── FeatureScorer
    """, language="text")

st.markdown("---")

# Dataset Loading Section
st.markdown("### 📥 Dataset Loading")

if dataset_option == "Sample Dataset (Breast Cancer)":
    if st.button("Load Sample Dataset", type="primary"):
        with st.spinner("Loading sample dataset..."):
            X, y, target_names = load_sample_data()
            st.session_state.X = X
            st.session_state.y = y
            st.session_state.target_names = target_names
            st.session_state.data_loaded = True
            st.success(f"✅ Dataset loaded! Shape: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Display dataset info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Samples", X.shape[0])
            with col2:
                st.metric("Features", X.shape[1])
            with col3:
                st.metric("Target Classes", len(target_names))
            
            # Show sample data
            with st.expander("📊 View Dataset Sample"):
                st.dataframe(pd.concat([X.head(10), y.head(10)], axis=1), use_container_width=True)
            
            # Show target distribution
            target_counts = y.value_counts().sort_index()
            fig = px.bar(
                x=[target_names[int(i)] for i in target_counts.index],
                y=target_counts.values,
                title="Target Distribution",
                labels={"x": "Class", "y": "Count"},
                color=target_counts.values,
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your dataset as a CSV file. Make sure it has a target column."
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Shape: {df.shape[0]} samples, {df.shape[1]} columns")
            
            # Target column selection
            target_col = st.selectbox(
                "Select target column:",
                df.columns.tolist(),
                help="Select the column containing target values"
            )
            
            if st.button("Load Dataset", type="primary"):
                X = df.drop(columns=[target_col])
                y = df[target_col]
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.data_loaded = True
                st.success("✅ Dataset loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Samples", X.shape[0])
                with col2:
                    st.metric("Features", X.shape[1])
                with col3:
                    st.metric("Target Classes", len(y.unique()) if task_type == "classification" else "Regression")
                
                with st.expander("📊 View Dataset Sample"):
                    st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

st.markdown("---")

# Feature Engineering Section
if st.session_state.data_loaded:
    st.markdown("### 🚀 Feature Engineering")
    
    if st.button("Run Feature Engineering", type="primary", use_container_width=True):
        X = st.session_state.X
        y = st.session_state.y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize feature engineer
        status_text.text("Initializing AutoFeatureEngineer...")
        progress_bar.progress(10)
        
        # Create feature engineer with custom parameters
        from feature_generation import FeatureGenerator
        from feature_scoring import FeatureScorer
        
        generator = FeatureGenerator(poly_degree=poly_degree, n_bins=n_bins)
        scorer = FeatureScorer(task=task_type)
        afe = AutoFeatureEngineer(k=k_features, task=task_type)
        afe.generator = generator
        afe.scorer = scorer
        
        # Generate features
        status_text.text("Generating features (Target Encoding, Polynomial, Binning)...")
        progress_bar.progress(30)
        
        X_train_fe = afe.fit_transform(X_train, y_train)
        
        status_text.text("Transforming test data...")
        progress_bar.progress(60)
        
        X_test_fe = afe.transform(X_test)
        
        status_text.text("Feature engineering complete!")
        progress_bar.progress(100)
        
        # Store in session state
        st.session_state.afe = afe
        st.session_state.X_train_fe = X_train_fe
        st.session_state.X_test_fe = X_test_fe
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        
        st.success("✅ Feature engineering completed successfully!")
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Original Features", X_train.shape[1])
        with col2:
            st.metric("Generated Features", "Many")
        with col3:
            st.metric("Selected Features", len(afe.selected_features))
        with col4:
            reduction = ((X_train.shape[1] - len(afe.selected_features)) / X_train.shape[1]) * 100
            st.metric("Feature Reduction", f"{reduction:.1f}%")
        
        # Show selected features
        with st.expander("📋 View Selected Features"):
            selected_df = pd.DataFrame({
                'Feature Name': afe.selected_features,
                'Index': range(1, len(afe.selected_features) + 1)
            })
            st.dataframe(selected_df, use_container_width=True, hide_index=True)
        
        # Feature comparison chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Original', 'Selected'],
            y=[X_train.shape[1], len(afe.selected_features)],
            marker_color=['#1f77b4', '#2ca02c'],
            text=[X_train.shape[1], len(afe.selected_features)],
            textposition='auto'
        ))
        fig.update_layout(
            title="Feature Count Comparison",
            xaxis_title="Dataset Type",
            yaxis_title="Number of Features",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Model Benchmarking Section
if st.session_state.data_loaded and st.session_state.afe is not None and run_benchmark_option:
    st.markdown("### 📊 Model Benchmarking")
    
    if st.button("Run Model Comparison", type="primary", use_container_width=True):
        afe = st.session_state.afe
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        X_train_fe = st.session_state.X_train_fe
        X_test_fe = st.session_state.X_test_fe
        
        with st.spinner("Training models and comparing performance..."):
            # Raw model
            raw_model = RandomForestClassifier(random_state=42, n_estimators=100)
            raw_model.fit(X_train, y_train)
            raw_pred = raw_model.predict(X_test)
            raw_acc = accuracy_score(y_test, raw_pred)
            
            # Engineered model
            fe_model = RandomForestClassifier(random_state=42, n_estimators=100)
            fe_model.fit(X_train_fe, y_train)
            fe_pred = fe_model.predict(X_test_fe)
            fe_acc = accuracy_score(y_test, fe_pred)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw Features Accuracy", f"{raw_acc:.4f}")
            with col2:
                st.metric("Engineered Features Accuracy", f"{fe_acc:.4f}")
            with col3:
                improvement = (fe_acc - raw_acc) * 100
                st.metric("Improvement", f"{improvement:+.2f}%", 
                         delta=f"{improvement:+.2f}%")
            
            # Comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Raw Features', 'Engineered Features'],
                y=[raw_acc, fe_acc],
                marker_color=['#ff7f0e', '#2ca02c'],
                text=[f"{raw_acc:.4f}", f"{fe_acc:.4f}"],
                textposition='auto'
            ))
            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Feature Set",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrices
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Raw Features Confusion Matrix")
                cm_raw = confusion_matrix(y_test, raw_pred)
                fig_raw = px.imshow(
                    cm_raw,
                    labels=dict(x="Predicted", y="Actual"),
                    title="Raw Features",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_raw, use_container_width=True)
            
            with col2:
                st.markdown("#### Engineered Features Confusion Matrix")
                cm_fe = confusion_matrix(y_test, fe_pred)
                fig_fe = px.imshow(
                    cm_fe,
                    labels=dict(x="Predicted", y="Actual"),
                    title="Engineered Features",
                    color_continuous_scale="Greens"
                )
                st.plotly_chart(fig_fe, use_container_width=True)
            
            # Classification reports
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Raw Features Classification Report")
                report_raw = classification_report(y_test, raw_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_raw).transpose(), use_container_width=True)
            
            with col2:
                st.markdown("#### Engineered Features Classification Report")
                report_fe = classification_report(y_test, fe_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report_fe).transpose(), use_container_width=True)

st.markdown("---")

# Download Section
if st.session_state.data_loaded and st.session_state.X_train_fe is not None:
    st.markdown("### 💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download engineered training data
        X_train_fe = st.session_state.X_train_fe
        y_train = st.session_state.y_train
        train_df = pd.concat([X_train_fe, y_train], axis=1)
        csv_train = train_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Engineered Training Data (CSV)",
            data=csv_train,
            file_name="engineered_train_data.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download engineered test data
        X_test_fe = st.session_state.X_test_fe
        y_test = st.session_state.y_test
        test_df = pd.concat([X_test_fe, y_test], axis=1)
        csv_test = test_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Engineered Test Data (CSV)",
            data=csv_test,
            file_name="engineered_test_data.csv",
            mime="text/csv"
        )
    
    # Download selected features list
    if st.session_state.afe is not None:
        features_df = pd.DataFrame({
            'Selected Features': st.session_state.afe.selected_features
        })
        csv_features = features_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Selected Features List (CSV)",
            data=csv_features,
            file_name="selected_features.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>🔧 <strong>Automated Feature Engineering Engine</strong></p>
    <p>Built with Streamlit, scikit-learn, and pandas</p>
</div>
""", unsafe_allow_html=True)

