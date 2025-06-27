import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 🎨 Enhanced Page Configuration
st.set_page_config(
    page_title="Human Resources Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "#Human Resources  Analytics Platform\nBuilt with by Neeraj Shah"
    }
)

# 🌈 Enhanced Styling with CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        color: #2c3e50 !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .nav-item {
        padding: 12px 24px;
        border-radius: 25px;
        border: none;
        text-decoration: none;
        font-weight: 600;
        color: black;
        background: rgba(255, 255, 255, 0.9);
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 1);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        color: black;
    }
    
    .active {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24) !important;
        color: black !important;
    }
    
    .insight-card {
        background: rgba(255, 255, 255, 0.1);
        border-left: 4px solid #ff6b6b;
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .prediction-result {
        background: linear-gradient(45deg, #11998e, #38ef7d);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin: 20px 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* DataFrame styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 🚀 Advanced Utility Functions
@st.cache_data
def generate_sample_hr_data(n_employees=1000):
    """Generate realistic HR sample data"""
    np.random.seed(42)
    
    departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations', 'R&D']
    positions = ['Junior', 'Mid-Level', 'Senior', 'Lead', 'Manager', 'Director']
    education_levels = ['Bachelor', 'Master', 'PhD', 'High School']
    
    data = {
        'employee_id': range(1, n_employees + 1),
        'age': np.random.normal(35, 8, n_employees).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.6, 0.4]),
        'department': np.random.choice(departments, n_employees),
        'position': np.random.choice(positions, n_employees),
        'education': np.random.choice(education_levels, n_employees),
        'years_at_company': np.random.exponential(3, n_employees),
        'monthly_income': np.random.normal(75000, 25000, n_employees),
        'performance_rating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.1, 0.3, 0.45, 0.1]),
        'job_satisfaction': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.1, 0.15, 0.3, 0.35, 0.1]),
        'work_life_balance': np.random.choice([1, 2, 3, 4], n_employees, p=[0.2, 0.3, 0.4, 0.1]),
        'distance_from_home': np.random.exponential(10, n_employees),
        'overtime': np.random.choice(['Yes', 'No'], n_employees, p=[0.3, 0.7]),
        'promotion_last_5years': np.random.choice([0, 1], n_employees, p=[0.7, 0.3]),
        'training_times_last_year': np.random.poisson(3, n_employees),
    }
    
    # Calculate attrition based on realistic factors
    attrition_prob = (
        0.1 +  # Base rate
        0.15 * (data['job_satisfaction'] == 1) +
        0.1 * (data['work_life_balance'] == 1) +
        0.08 * (data['years_at_company'] < 1) +
        0.05 * (data['overtime'] == 'Yes') -
        0.1 * (data['performance_rating'] >= 4) -
        0.05 * (data['promotion_last_5years'] == 1)
    )
    
    data['attrition'] = np.random.binomial(1, np.clip(attrition_prob, 0, 1), n_employees)
    
    df = pd.DataFrame(data)
    df['monthly_income'] = df['monthly_income'].clip(lower=30000)
    df['age'] = df['age'].clip(lower=18, upper=65)
    df['years_at_company'] = df['years_at_company'].clip(lower=0, upper=40)
    
    return df


# 🎯 Enhanced Header Section
def create_advanced_header():
    st.markdown("""
        <div class="main-header">
            <div style="display: flex; align-items: center; justify-content: center;">
                <div style="text-align: center;">
                    <h1 style="margin: 0; font-size: 42px; color: #2c3e50;">
                        🚀 Human Resources Analytics Platform
                    </h1>
                    <p style="margin: 10px 0 0 0; font-size: 18px; color: #666;">
                        AI-Powered People Analytics & Workforce Intelligence
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# 🌐 Enhanced Navigation
def create_navigation():
    query_params = st.query_params
    current_page = query_params.get("page", "Dashboard")
    
    pages = ["Dashboard", "Analytics", "ML Predictions", "Employee Management", "Reports"]
    
    nav_html = '<div class="nav-container">'
    for page in pages:
        active_class = "active" if current_page == page else ""
        nav_html += f'<a href="/?page={page}" class="nav-item {active_class}">{page}</a>'
    nav_html += '</div>'
    
    st.markdown(nav_html, unsafe_allow_html=True)
    return current_page

# 📊 Advanced Dashboard Page
def show_dashboard(df):
    st.markdown("## 📈 Executive Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <h3>👥 Total Employees</h3>
                <h2>{len(df):,}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        attrition_rate = (df['attrition'].sum() / len(df)) * 100
        st.markdown(f"""
            <div class="metric-card">
                <h3>📉 Attrition Rate</h3>
                <h2>{attrition_rate:.1f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_salary = df['monthly_income'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Avg Salary</h3>
                <h2>₹{avg_salary:,.0f}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_tenure = df['years_at_company'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <h3>⏱️ Avg Tenure</h3>
                <h2>{avg_tenure:.1f} years</h2>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        high_performers = (df['performance_rating'] >= 4).sum()
        st.markdown(f"""
            <div class="metric-card">
                <h3>⭐ High Performers</h3>
                <h2>{high_performers}</h2>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    
    # Insights Section
    st.markdown("## 🎯 Key Insights")
    
    insights = [
        f"🔍 {df['department'].value_counts().index[0]} department has the highest headcount ({df['department'].value_counts().iloc[0]} employees)",
        f"⚠️ Employees with job satisfaction rating 1-2 have {((df[df['job_satisfaction'] <= 2]['attrition'].mean() * 100)):.1f}% attrition rate",
        f"💡 High performers (rating 4-5) have {((df[df['performance_rating'] >= 4]['attrition'].mean() * 100)):.1f}% attrition rate",
        f"📊 Average tenure for retained employees is {df[df['attrition'] == 0]['years_at_company'].mean():.1f} years"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

# 🔬 Cleaned-up Advanced Analytics Page
def show_analytics(df):
    st.markdown("## 🔬 Advanced Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🔍 Segmentation", "📈 Trends"])
    
    # Tab 1: Department Deep Dive only (chart code removed)
    with tab1:
        st.markdown("### Department Deep Dive")
        dept_analysis = df.groupby('department').agg({
            'monthly_income': ['mean', 'median'],
            'performance_rating': 'mean',
            'job_satisfaction': 'mean',
            'attrition': 'mean',
            'years_at_company': 'mean'
        }).round(2)

        st.dataframe(dept_analysis, use_container_width=True)
    
    # Tab 2: Risk Segmentation
    with tab2:
        st.markdown("### Employee Segmentation")
        
        # Risk Segmentation
        df['risk_score'] = (
            (5 - df['job_satisfaction']) * 0.3 +
            (5 - df['performance_rating']) * 0.2 +
            (df['years_at_company'] < 2) * 0.3 +
            (df['overtime'] == 'Yes') * 0.2
        )
        
        df['risk_category'] = pd.cut(df['risk_score'], 
                                     bins=[0, 1, 2, 5], 
                                     labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        risk_dist = df['risk_category'].value_counts()
        fig = px.pie(values=risk_dist.values, names=risk_dist.index,
                     title="Employee Risk Distribution")
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Trends
    with tab3:
        st.markdown("### Trend Analysis")
        
        # Simulated time series data
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        trend_data = pd.DataFrame({
            'date': dates,
            'headcount': np.random.normal(1000, 50, len(dates)).cumsum() % 200 + 800,
            'attrition_rate': np.random.normal(15, 3, len(dates))
        })
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['headcount'], 
                                 name="Headcount"), secondary_y=False)
        fig.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['attrition_rate'], 
                                 name="Attrition Rate"), secondary_y=True)
        
        fig.update_layout(title="Workforce Trends Over Time", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

# 🤖 Advanced ML Predictions Page
def show_ml_predictions(df):
    st.markdown("## 🤖 Advanced ML Predictions")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Attrition Prediction", "📈 Performance Prediction", "💰 Salary Prediction"])
    
    with tab1:
        st.markdown("### Employee Attrition Prediction")
        
        # Prepare data for ML
        ml_df = df.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['gender', 'department', 'position', 'education', 'overtime']
        for col in categorical_cols:
            ml_df[col + '_encoded'] = le.fit_transform(ml_df[col])
        
        # Features for prediction
        feature_cols = ['age', 'years_at_company', 'monthly_income', 'performance_rating',
                       'job_satisfaction', 'work_life_balance', 'distance_from_home',
                       'promotion_last_5years', 'training_times_last_year'] + \
                      [col + '_encoded' for col in categorical_cols]
        
        X = ml_df[feature_cols]
        y = ml_df['attrition']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Try multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        model_results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_results[name] = {'model': model, 'accuracy': accuracy}
        
        # Select best model
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
        best_model = model_results[best_model_name]['model']
        
        st.success(f"🏆 Best Model: {best_model_name} (Accuracy: {model_results[best_model_name]['accuracy']:.3f})")
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                        title='Top 10 Feature Importance for Attrition Prediction')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        # Prediction interface
        st.markdown("### 🔮 Predict Individual Employee Attrition")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 65, 30)
            tenure = st.slider("Years at Company", 0, 40, 5)
            salary = st.slider("Monthly Income", 30000, 200000, 75000, step=5000)
        
        with col2:
            performance = st.select_slider("Performance Rating", options=[1, 2, 3, 4, 5], value=3)
            satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4, 5], value=3)
            work_balance = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=2)
        
        with col3:
            department = st.selectbox("Department", df['department'].unique())
            overtime = st.selectbox("Overtime", ['Yes', 'No'])
            promotions = st.slider("Promotions in Last 5 Years", 0, 5, 1)
        
        if st.button("🎯 Predict Attrition Risk", type="primary"):
            # Create prediction input
            input_data = pd.DataFrame({
                'age': [age],
                'years_at_company': [tenure],
                'monthly_income': [salary],
                'performance_rating': [performance],
                'job_satisfaction': [satisfaction],
                'work_life_balance': [work_balance],
                'distance_from_home': [10],  # Default value
                'promotion_last_5years': [promotions],
                'training_times_last_year': [3],  # Default value
            })
            
            # Add encoded categorical features (simplified)
            for col in categorical_cols:
                input_data[col + '_encoded'] = [0]  # Simplified encoding
            
            prediction = best_model.predict(input_data[feature_cols])[0]
            probability = best_model.predict_proba(input_data[feature_cols])[0]
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="prediction-result" style="background: linear-gradient(45deg, #ff6b6b, #ee5a24);">
                        ⚠️ HIGH ATTRITION RISK<br>
                        Probability: {probability[1]:.1%}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="prediction-result" style="background: linear-gradient(45deg, #11998e, #38ef7d);">
                        ✅ LOW ATTRITION RISK<br>
                        Probability: {probability[0]:.1%}
                    </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Performance Prediction Model")
        st.info("🚧 Feature coming soon - Performance prediction based on multiple factors")
    
    with tab3:
        st.markdown("### Salary Prediction Model")
        st.info("🚧 Feature coming soon - Salary benchmarking and prediction")

# 👥 Employee Management Page
def show_employee_management(df):
    st.markdown("## 👥 Employee Management")
    
    tab1, tab2, tab3 = st.tabs(["📋 Employee Directory", "➕ Add Employee", "📊 Bulk Operations"])
    
    with tab1:
        st.markdown("### Employee Directory")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            dept_filter = st.multiselect("Filter by Department", df['department'].unique())
        with col2:
            position_filter = st.multiselect("Filter by Position", df['position'].unique())
        with col3:
            risk_filter = st.selectbox("Filter by Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])
        
        # Apply filters
        filtered_df = df.copy()
        if dept_filter:
            filtered_df = filtered_df[filtered_df['department'].isin(dept_filter)]
        if position_filter:
            filtered_df = filtered_df[filtered_df['position'].isin(position_filter)]
        
        # Search functionality
        search_term = st.text_input("🔍 Search employees...")
        if search_term:
            filtered_df = filtered_df[
                filtered_df['department'].str.contains(search_term, case=False) |
                filtered_df['position'].str.contains(search_term, case=False)
            ]
        
        # Display results
        st.markdown(f"**Showing {len(filtered_df)} employees**")
        
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display",
            df.columns.tolist(),
            default=['employee_id', 'age', 'gender', 'department', 'position', 'monthly_income', 'performance_rating']
        )
        
        if display_cols:
            st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
    
    with tab2:
        st.markdown("### Add New Employee")
        
        with st.form("add_employee_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_age = st.number_input("Age", min_value=18, max_value=65, value=30)
                new_gender = st.selectbox("Gender", ["Male", "Female"])
                new_dept = st.selectbox("Department", df['department'].unique())
                new_position = st.selectbox("Position", df['position'].unique())
                new_education = st.selectbox("Education", df['education'].unique())
            
            with col2:
                new_salary = st.number_input("Monthly Income", min_value=30000, value=50000, step=5000)
                new_performance = st.select_slider("Performance Rating", options=[1, 2, 3, 4, 5], value=3)
                new_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4, 5], value=3)
                new_work_balance = st.select_slider("Work Life Balance", options=[1, 2, 3, 4], value=2)
                new_overtime = st.selectbox("Overtime", ["Yes", "No"])
            
            submitted = st.form_submit_button("➕ Add Employee", type="primary")
            
            if submitted:
                st.success("✅ Employee added successfully! (In a real system, this would save to database)")
    
    with tab3:
        st.markdown("### Bulk Operations")
        
        st.markdown("#### 📤 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export to CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"hr_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export to Excel"):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Employee_Data', index=False)
                    
                st.download_button(
                    label="Download Excel",
                    data=output.getvalue(),
                    file_name=f"hr_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        st.markdown("#### 📤 Upload Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_df = pd.read_csv(uploaded_file)
                else:
                    new_df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(new_df)} records found.")
                st.dataframe(new_df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

# 📋 Reports Page
def show_reports():
    st.markdown("## 📋 Advanced Reports")
    
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "📈 Departmental Reports", "🎯 Custom Reports"])
    
    with tab1:
        st.markdown("### Executive Summary Report")
        
        report_date = datetime.now().strftime("%B %Y")
        
        st.markdown(f"""
        ### HR Analytics Executive Summary - {report_date}
        
        #### Key Highlights:
        - **Workforce Growth**: 12% increase in headcount year-over-year
        - **Retention Rate**: 87% overall retention with improvements in Engineering
        - **Performance**: 68% of employees rated as high performers (4+ rating)
        - **Engagement**: Average job satisfaction increased to 3.8/5
        - **Diversity**: 40% female representation across all levels
        
        #### Action Items:
        1. Focus retention efforts on Sales and Marketing departments
        2. Implement targeted development programs for mid-level employees
        3. Review compensation structure for high-risk attrition segments
        4. Expand remote work policies to improve work-life balance
        """)
        
        # Generate PDF Report Button
        if st.button("📄 Generate PDF Report", type="primary"):
            st.success("📄 PDF report generation initiated! (Feature would be implemented with reportlab)")
    
    with tab2:
        st.markdown("### Departmental Deep Dive Reports")
        
        # Sample departmental data
        dept_data = {
            'Department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'Operations'],
            'Headcount': [245, 189, 156, 45, 78, 123],
            'Attrition_Rate': [8.2, 15.7, 12.3, 6.7, 9.1, 11.5],
            'Avg_Salary': [95000, 78000, 72000, 85000, 88000, 65000],
            'Satisfaction': [4.1, 3.6, 3.8, 4.2, 3.9, 3.7],
            'Performance': [4.2, 3.8, 3.9, 4.0, 4.1, 3.8]
        }
        
        dept_df = pd.DataFrame(dept_data)
        
        selected_dept = st.selectbox("Select Department for Detailed Analysis", dept_df['Department'])
        
        if selected_dept:
            dept_row = dept_df[dept_df['Department'] == selected_dept].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Headcount", dept_row['Headcount'])
            with col2:
                st.metric("Attrition Rate", f"{dept_row['Attrition_Rate']}%", delta="-2.1%")
            with col3:
                st.metric("Avg Salary", f"₹{dept_row['Avg_Salary']:,}")
            with col4:
                st.metric("Satisfaction", f"{dept_row['Satisfaction']}/5", delta="0.3")
            
            # Department-specific insights
            st.markdown(f"""
            #### {selected_dept} Department Analysis:
            
            **Strengths:**
            - Above-average performance ratings
            - Strong team collaboration
            - Good technical skills development
            
            **Areas for Improvement:**
            - Work-life balance concerns
            - Career progression clarity needed
            - Cross-training opportunities
            
            **Recommendations:**
            - Implement flexible working arrangements
            - Create clear career development paths
            - Increase training budget by 15%
            """)
    
    with tab3:
        st.markdown("### Custom Report Builder")
        
        st.markdown("#### Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_type = st.selectbox(
                "Report Type",
                ["Attrition Analysis", "Performance Review", "Compensation Analysis", "Diversity Report"]
            )
            
            date_range = st.date_input(
                "Date Range",
                value=[datetime.now() - timedelta(days=30), datetime.now()],
                max_value=datetime.now()
            )
        
        with col2:
            departments = st.multiselect(
                "Include Departments",
                ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"],
                default=["Engineering", "Sales"]
            )
            
            metrics = st.multiselect(
                "Include Metrics",
                ["Headcount", "Attrition", "Performance", "Satisfaction", "Salary"],
                default=["Attrition", "Performance"]
            )
        
        report_format = st.radio("Output Format", ["Dashboard View", "PDF Download", "Excel Export"])
        
        if st.button("🚀 Generate Custom Report", type="primary"):
            st.success(f"✅ {report_type} report generated successfully for {len(departments)} departments!")
            
            # Sample custom report visualization
            if report_type == "Attrition Analysis":
                sample_data = pd.DataFrame({
                    'Month': pd.date_range('2024-01-01', periods=12, freq='M'),
                    'Attrition_Rate': np.random.normal(12, 3, 12),
                    'Voluntary': np.random.normal(8, 2, 12),
                    'Involuntary': np.random.normal(4, 1, 12)
                })
                
                fig = px.line(sample_data, x='Month', y=['Attrition_Rate', 'Voluntary', 'Involuntary'],
                             title=f"{report_type} - Custom Report")
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

# 🏠 Main Application Logic
def main():
    create_advanced_header()
    
    # Load or generate sample data
    if 'hr_data' not in st.session_state:
        with st.spinner("🔄 Loading HR data..."):
            st.session_state.hr_data = generate_sample_hr_data(1000)
    
    df = st.session_state.hr_data
    
    # Navigation
    current_page = create_navigation()
    
    # Sidebar with additional controls
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.info(f"👥 Total Employees: {len(df)}")
        st.info(f"📉 Attrition Rate: {(df['attrition'].sum()/len(df)*100):.1f}%")
        st.info(f"⭐ Avg Performance: {df['performance_rating'].mean():.1f}/5")
        
        st.markdown("---")
        
        # Data refresh
        if st.button("🔄 Refresh Data"):
            st.session_state.hr_data = generate_sample_hr_data(1000)
            st.rerun()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        show_advanced_charts = st.checkbox("Show Advanced Charts", True)
        real_time_updates = st.checkbox("Real-time Updates", False)
        
        if real_time_updates:
            st.success("🔴 Live mode active")
        
        st.markdown("---")
        
        # Help section
        with st.expander("❓ Help & Support"):
            st.markdown("""
            **Quick Guide:**
            - 📈 Dashboard: Overview and key metrics
            - 🔬 Analytics: Deep dive analysis
            - 🤖 ML Predictions: AI-powered insights
            - 👥 Employee Mgmt: Manage employee data
            - 📋 Reports: Generate custom reports
            
            **Need Help?**
            - 📧 support@hranalytics.com
            - 📞 +91-98765-43210
            """)
    
    # Page routing
    if current_page == "Dashboard":
        show_dashboard(df)
    elif current_page == "Analytics":
        show_analytics(df)
    elif current_page == "ML Predictions":
        show_ml_predictions(df)
    elif current_page == "Employee Management":
        show_employee_management(df)
    elif current_page == "Reports":
        show_reports()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.7);">
            <p>🚀 Advanced HR Analytics Platform v2.0 | Built by Neeraj Shah</p>
            <p>Powered by Streamlit, Plotly, and Scikit-learn | © 2024 All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
