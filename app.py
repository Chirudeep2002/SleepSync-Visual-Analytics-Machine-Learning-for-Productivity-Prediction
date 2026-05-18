# app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
from xgboost import XGBRegressor

plotly_template = "plotly_dark"
# -----------------------------------
# Plotly Dark Theme Function
# -----------------------------------

def apply_dark_theme(fig, height=500):

    fig.update_layout(

        template="plotly_dark",

        height=height,

        plot_bgcolor="rgba(0,0,0,0)",

        paper_bgcolor="rgba(0,0,0,0)",

        font=dict(
            color="white",
            size=14
        ),

        title_font=dict(
            size=24,
            color="white"
        ),

        legend=dict(
            font=dict(
                color="white",
                size=12
            )
        ),

        xaxis=dict(
            showgrid=False,
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        ),

        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            title_font=dict(color="white"),
            tickfont=dict(color="white")
        )
    )

    return fig


# --- Streamlit Config ---
st.set_page_config(
    page_title="SleepSync AI",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Main selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Selected text inside box */
.stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}

/* Dropdown popup container */
div[data-baseweb="popover"] {
    background-color: #111827 !important;
}

/* Actual dropdown list */
ul {
    background-color: #111827 !important;
}

/* Dropdown options */
li {
    background-color: #111827 !important;
    color: white !important;
    font-size: 16px !important;
}

/* Hovered option */
li:hover {
    background-color: #2563EB !important;
    color: white !important;
}

/* Selected option */
li[aria-selected="true"] {
    background-color: #1D4ED8 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Main App */
.stApp {
    background: linear-gradient(to bottom right, #0B1120, #111827);
    color: #F9FAFB;
}

/* Main Container */
.main {
    background-color: transparent;
}

/* Headers */
h1, h2, h3, h4, h5 {
    color: #F9FAFB !important;
    font-family: 'Segoe UI', sans-serif;
}

/* Paragraph Text */
p, li, label, div {
    color: #D1D5DB;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #374151;
}

/* Metric Cards */
[data-testid="metric-container"] {

    background: rgba(31, 41, 55, 0.45);

    border: 1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(12px);

    padding: 18px;

    border-radius: 18px;

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);
}

/* Buttons */
.stButton > button {
    background-color: #3B82F6;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #2563EB;
    color: white;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #1F2937;
    border-radius: 10px;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #374151;
    border-radius: 12px;
    overflow: hidden;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    color: #D1D5DB;
}

.stTabs [aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 3px solid #60A5FA;
}

/* Sliders */
.stSlider > div > div > div {
    color: #60A5FA;
}

/* Success Messages */
.stAlert {
    border-radius: 12px;
}

/* Footer */
footer {
    visibility: hidden;
}
/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent;
    border-radius: 10px 10px 0px 0px;
    padding-left: 20px;
    padding-right: 20px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(59,130,246,0.15);
}
            .stTabs [data-baseweb="tab-list"] {
    gap: 18px;
}

.stTabs [data-baseweb="tab"] {

    height: 50px;

    background: rgba(255,255,255,0.03);

    border-radius: 12px 12px 0px 0px;

    padding-left: 20px;
    padding-right: 20px;

    font-weight: 600;

    transition: 0.3s;
}

.stTabs [aria-selected="true"] {

    background: rgba(59,130,246,0.18);

    border-bottom: 3px solid #60A5FA;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Reduce excessive side padding */
.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 100%;
}
/* Plotly Chart Containers */
[data-testid="stPlotlyChart"] {

    background: rgba(255,255,255,0.03);

    border: 1px solid rgba(255,255,255,0.06);

    padding: 15px;

    border-radius: 18px;

    margin-bottom: 25px;

    backdrop-filter: blur(10px);
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Dark Selectbox */

.stSelectbox div[data-baseweb="select"] > div {

    background-color: #111827 !important;

    color: white !important;

    border: 1px solid rgba(255,255,255,0.08) !important;

    border-radius: 10px !important;
}

/* Dropdown menu */

div[role="listbox"] {

    background-color: #111827 !important;

    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Options */

div[role="option"] {

    background-color: #111827 !important;

    color: white !important;
}

/* Hover */

div[role="option"]:hover {

    background-color: #1F2937 !important;
}

/* Selected */

div[aria-selected="true"] {

    background-color: #2563EB !important;

    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# --- Title + Tabs ---
st.markdown("""
# 😴 SleepSync AI
#### AI-Powered Sleep & Productivity Analytics
""")
st.divider()
tabs = st.tabs([
    "Introduction", 
    "Dataset Analysis", 
    "Filtered Data", 
    "Correlation Heatmap", 
    "Modeling", 
    "Predict Your Productivity", 
    "Conclusion"
])

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("sleep_productivity_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()
# --- Feature Engineering ---

df["Sleep Efficiency"] = (
    df["Sleep Duration (hrs)"] /
    df["Work Hours"]
)

df["Recovery Score"] = (
    df["Sleep Duration (hrs)"] -
    df["Sleep Debt (hrs)"]
)

df["Work-Life Balance"] = (
    df["Sleep Duration (hrs)"] /
    (df["Work Hours"] + 1)
)
# --- Tab 1: Introduction ---
with tabs[0]:
    st.header("📘 Introduction")
    st.markdown("""
    Sleep plays a crucial role in maintaining cognitive functions, emotional well-being, and overall productivity.
    Especially in professional settings, variations in sleep duration can have significant impacts on an individual's efficiency,
    decision-making capabilities, and stress management.

    In this project,the aim is to explore the relationship between sleep patterns and productivity levels across various professions.
    Using real-world sleep tracker datasets combined with productivity surveys, I uncover hidden trends and generate insights that
    could benefit individuals and organizations alike.

                
    ### 🎯 Objective
    - Analyze how sleep duration correlates with productivity levels across different professions.
    - Compare sleep trends and identify professions with higher sleep debt risks.
    - Examine how extended work hours contribute to sleep deprivation and reduced efficiency.
    - Provide actionable insights that can guide wellness initiatives in workplaces.

    ### 📈 Expected Outcomes
    - Identify the optimal sleep duration for peak productivity.
    - Highlight professions that are most affected by sleep deprivation.
    - Demonstrate the impact of long work hours on sleep debt and performance.
    - Offer data-driven recommendations for individuals and organizations aiming to optimize productivity through better sleep habits.
    """)

# --- Tab 2: Dataset Analysis ---
with tabs[1]:
    st.header("📊 Dataset Overview")
    avg_sleep = df['Sleep Duration (hrs)'].mean()
    avg_productivity = df['Productivity Level'].mean()
    avg_workhours = df['Work Hours'].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("😴 Avg Sleep", f"{avg_sleep:.2f} hrs")

    with col2:
        st.metric("⚡ Productivity", f"{avg_productivity:.2f}")

    with col3:
        st.metric("💼 Work Hours", f"{avg_workhours:.2f}")

    with st.expander("📄 View Dataset"):

        st.dataframe(df, use_container_width=True)

        st.subheader("🔎 Summary Statistics")

        st.dataframe(df.describe(), use_container_width=True)

    st.subheader("📊 Choose Visualization")
    plot_type = st.selectbox("Select Plot Type:", ["Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Pie Chart"])

    if plot_type != "Pie Chart":
        x_axis = st.selectbox("Select X-axis:", df.columns)
        y_axis = st.selectbox("Select Y-axis:", df.columns)

    if st.button("Generate Plot"):
        if plot_type == "Bar Plot":
            fig = px.bar(df, x=x_axis, y=y_axis, color=x_axis)
            st.plotly_chart(fig, use_container_width=True,key='1')
        elif plot_type == "Line Plot":
            fig = px.line(df, x=x_axis, y=y_axis, markers=True)
            st.plotly_chart(fig, use_container_width=True,key='2')
        elif plot_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis, color='Profession')
            st.plotly_chart(fig, use_container_width=True,key='3')
        elif plot_type == "Histogram":
            fig = px.histogram(df, x=x_axis)
            st.plotly_chart(fig, use_container_width=True,key='4')
        elif plot_type == "Pie Chart":
            pie_col = st.selectbox("Select Column for Pie Chart:", df.columns)
            fig = px.pie(df, names=pie_col)
            st.plotly_chart(fig, use_container_width=True,key='5')

# --- Tab 3: Filtered Data ---
with tabs[2]:
    st.header("🔍 View and Explore Filtered Data")
    professions = st.multiselect("Filter by Profession:", options=df['Profession'].unique(), default=df['Profession'].unique())
    sleep_range = st.slider("Sleep Duration Range (hrs):", 0.0, 12.0, (5.0, 9.0))
    filtered_df = df[(df['Profession'].isin(professions)) & (df['Sleep Duration (hrs)'].between(*sleep_range))]

    st.subheader("📄 Filtered Data Table")
    st.dataframe(filtered_df)

    st.subheader("📊 Summary Statistics of Filtered Data")
    st.write(filtered_df.describe())

    st.subheader("📈 Visualize Filtered Data")
    plot_type_filtered = st.selectbox("Select Plot Type (Filtered Data):", ["Bar Plot", "Line Plot", "Scatter Plot", "Histogram", "Pie Chart"])

    if plot_type_filtered != "Pie Chart":
        x_axis_filtered = st.selectbox("Select X-axis (Filtered Data):", filtered_df.columns)
        y_axis_filtered = st.selectbox("Select Y-axis (Filtered Data):", filtered_df.columns)

    if st.button("Generate Filtered Data Plot"):
        if plot_type_filtered == "Bar Plot":
            fig = px.bar(filtered_df, x=x_axis_filtered, y=y_axis_filtered, color=x_axis_filtered)
            st.plotly_chart(fig, use_container_width=True,key='6')
        elif plot_type_filtered == "Line Plot":
            fig = px.line(filtered_df, x=x_axis_filtered, y=y_axis_filtered, markers=True)
            st.plotly_chart(fig, use_container_width=True,key='7')
        elif plot_type_filtered == "Scatter Plot":
            fig = px.scatter(filtered_df, x=x_axis_filtered, y=y_axis_filtered, color='Profession')
            st.plotly_chart(fig, use_container_width=True,key='8')
        elif plot_type_filtered == "Histogram":
            fig = px.histogram(filtered_df, x=x_axis_filtered)
            st.plotly_chart(fig, use_container_width=True,key='9')
        elif plot_type_filtered == "Pie Chart":
            pie_col_filtered = st.selectbox("Select Column for Pie Chart:", filtered_df.columns)
            fig = px.pie(filtered_df, names=pie_col_filtered)
            st.plotly_chart(fig, use_container_width=True,key='10')

# --- Tab 4: Correlation Heatmap ---
with tabs[3]:
    st.header("📌 Correlation Heatmap Analysis")
    st.markdown("""
    The correlation heatmap below visualizes the strength and direction of relationships between key variables such as
    sleep duration, productivity level, work hours, and sleep debt.
    """)

    st.subheader("🎯 Set Correlation Threshold")
    threshold = st.slider("Select minimum correlation to display (absolute value):", 0.0, 1.0, 0.5, 0.1)

    corr_matrix = df[['Sleep Duration (hrs)', 'Productivity Level', 'Work Hours', 'Sleep Debt (hrs)']].corr()
    strong_corr = corr_matrix[(corr_matrix.abs() >= threshold)]

    st.subheader("📊 Correlation Heatmap (Filtered)")
    fig = px.bar(
    df,
    x="Profession",
    y="Productivity Level",
    color="Profession",
    title="⚡ Productivity Across Professions",

    template="plotly_dark"
    )

    fig.update_layout(

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        size=14
    ),

    title_font=dict(
        size=24,
        color="white"
    ),

    legend=dict(
        font=dict(
            color="white",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),

    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )
)

    st.plotly_chart(
    fig,
    use_container_width=True,
    key='11'
    )

    fig2 = px.scatter(
    df,

    x="Sleep Duration (hrs)",
    y="Productivity Level",

    color="Profession",

    size="Work Hours",

    hover_data=[
        "Profession",
        "Sleep Debt (hrs)"
    ],

    title="🧠 Sleep Duration vs Productivity",

    template="plotly_dark"
    )

    fig2.update_layout(

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        size=14
    ),

    title_font=dict(
        size=24,
        color="white"
    ),

    legend=dict(
        font=dict(
            color="white",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),

    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )
)

    st.plotly_chart(
    fig2,
    use_container_width=True,
    key='12'
    )
    fig3 = px.histogram(
    df,

    x="Sleep Duration (hrs)",

    nbins=20,

    color_discrete_sequence=["#60A5FA"],

    title="🌙 Sleep Duration Distribution",

    template="plotly_dark"
    )

    fig3.update_layout(

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        size=14
    ),

    title_font=dict(
        size=24,
        color="white"
    ),

    legend=dict(
        font=dict(
            color="white",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),

    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )
)

    st.plotly_chart(
    fig3,
    use_container_width=True,
    key='13'
    )

    fig4 = px.box(
    df,

    x="Profession",
    y="Work Hours",

    color="Profession",

    title="💼 Work Hours Distribution by Profession",

    template="plotly_dark"
    )

    fig4.update_layout(

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        size=14
    ),

    title_font=dict(
        size=24,
        color="white"
    ),

    legend=dict(
        font=dict(
            color="white",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),

    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )
)

    st.plotly_chart(
    fig4,
    use_container_width=True,
    key='14'
    )

    corr = df.select_dtypes(include='number').corr()

    fig5 = px.imshow(
    corr,

    text_auto=True,

    color_continuous_scale="RdBu_r",

    title="📊 Correlation Heatmap",

    template="plotly_dark"
    )

    fig5.update_layout(

    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white",
        size=14
    ),

    title_font=dict(
        size=24,
        color="white"
    ),

    legend=dict(
        font=dict(
            color="white",
            size=12
        )
    ),

    xaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    ),

    yaxis=dict(
        title_font=dict(color="white"),
        tickfont=dict(color="white")
    )
)

    st.plotly_chart(
    fig5,
    use_container_width=True,
    key='15'
    )

    st.subheader("📋 Correlation Values Table")
    st.dataframe(strong_corr)

# --- Tab 5: Modeling ---
with tabs[4]:
    st.header("🤖 Model Training")
    X = df[
    [
        'Sleep Duration (hrs)',
        'Work Hours',
        'Sleep Debt (hrs)',
        'Sleep Efficiency',
        'Recovery Score',
        'Work-Life Balance'
    ]
]
    y = df['Productivity Level']

    st.subheader("🎯 Set Train/Test Split Ratio")
    test_size = st.slider("Select Test Set Percentage:", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.subheader("⚙️ Choose a Machine Learning Model")
    model_choice = st.selectbox(
    "Choose Model:",
    [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
    ]
)
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
        )
    else:
        model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
        )

    if st.button("Train Model"):
        model.fit(X_train, y_train)
        st.session_state.model = model  
        st.session_state.features = X.columns.tolist()# Save model for prediction tab
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        st.success(f"✅ {model_choice} Model Trained Successfully!")

        st.subheader("📋 Model Summary")
        st.write(f"- **Training Set R² Score**: {model.score(X_train, y_train):.2f}")
        st.write(f"- **Testing Set R² Score**: {model.score(X_test, y_test):.2f}")

        st.subheader("📊 Model Performance Metrics")
        perf_df = pd.DataFrame({
            'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score'],
            'Value': [mse, rmse, mae, r2]
        })
        st.dataframe(perf_df)
        
        comparison_df = pd.DataFrame({

        "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost"
        ],

        "Type": [
        "Linear",
        "Ensemble",
        "Boosting"
        ]
        })
        st.subheader("📊 Available ML Models")

        st.dataframe(
        comparison_df,
        use_container_width=True
        )
        if model_choice in ["Random Forest", "XGBoost"]:

            importance_df = pd.DataFrame({

            "Feature": X.columns,
            "Importance": model.feature_importances_

            })  

            importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
            )

            fig_importance = px.bar(

            importance_df,
            x="Importance",
            y="Feature",

            orientation="h",

            color="Importance",

            title="🧠 Feature Importance Analysis",

            template="plotly_dark"
            )

            fig_importance = apply_dark_theme(
                fig_importance,
                height=500
                )

            st.plotly_chart(
            fig_importance,
            use_container_width=True,
            key='16'
            )
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("MSE", f"{mse:.2f}")

            col2.metric("RMSE", f"{rmse:.2f}")

            col3.metric("MAE", f"{mae:.2f}")

            col4.metric("R² Score", f"{r2:.2f}")

# --- Tab 6: Prediction ---
with tabs[5]:
    st.header("🔮 Predict Your Productivity, Profession & Sleep Health")
    sleep_input = st.slider("Enter your Sleep Duration (hrs):", 0.0, 12.0, 7.0)
    work_input = st.slider("Enter your Work Hours per Day:", 0.0, 24.0, 8.0)
    # Feature Engineering for Prediction
    sleep_debt = max(0, 8 - sleep_input)

    sleep_efficiency = (
    sleep_input / work_input
    if work_input != 0 else 0
    )

    recovery_score = (
    sleep_input - sleep_debt
    )

    work_life_balance = (
    sleep_input / (work_input + 1)
    )

# Input DataFrame

    if 'model' in st.session_state and st.session_state.model is not None:

    # Feature Engineering for Prediction

        sleep_debt = max(0, 8 - sleep_input)

        sleep_efficiency = (
        sleep_input / work_input
        if work_input != 0 else 0
        )

        recovery_score = (
        sleep_input - sleep_debt
        )

        work_life_balance = (
        sleep_input / (work_input + 1)
        )

    # Prediction Input

        input_df = pd.DataFrame({

        "Sleep Duration (hrs)": [sleep_input],

        "Work Hours": [work_input],

        "Sleep Debt (hrs)": [sleep_debt],

        "Sleep Efficiency": [sleep_efficiency],

        "Recovery Score": [recovery_score],

        "Work-Life Balance": [work_life_balance]
    })

    # Prediction

        pred_output = st.session_state.model.predict(input_df)

    # Show Prediction

        st.success(
        f"📈 Estimated Productivity Level: {pred_output[0]:.2f}"
    )

    # Gauge Chart

        fig_pred = go.Figure(go.Indicator(

        mode="gauge+number",

        value=pred_output[0],

        title={'text': "Predicted Productivity"},

        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "#60A5FA"}
        }
    ))

        fig_pred.update_layout(
        template="plotly_dark",
        height=400
    )

        st.plotly_chart(
        fig_pred,
        use_container_width=True,
        key='17'
    )

    # Profession Prediction
        df_copy = df.copy()
        df_copy['Sleep_Diff'] = abs(df_copy['Sleep Duration (hrs)'] - sleep_input)
        df_copy['Work_Diff'] = abs(df_copy['Work Hours'] - work_input)
        df_copy['Total_Diff'] = df_copy['Sleep_Diff'] + df_copy['Work_Diff']
        predicted_profession = df_copy.sort_values('Total_Diff').iloc[0]['Profession']

        st.info(f"🧑‍💼 Based on your habits, you most closely match: **{predicted_profession}**")

    # Insomnia Prediction
        if sleep_input < 5:
            st.error("⚠️ Risk Alert: Based on your sleep duration, you might be suffering from **Insomnia** (Sleep < 5 hrs).")
        else:
            st.success("😴 Good Sleep Health: Your sleep duration looks healthy!")
    
        st.subheader("🧠 AI Wellness Recommendations")
        st.subheader("🤖 AI Productivity Insights")

        if pred_output[0] >= 8:

            st.success("""
    🚀 High Productivity Forecast

    AI Analysis:
    Your sleep-work balance indicates
    strong cognitive efficiency,
    healthy recovery, and optimal
    productivity potential.
    """)

        elif pred_output[0] >= 6:

            st.info("""
    ⚡ Moderate Productivity Forecast

    AI Analysis:
    Your current sleep schedule supports
    average productivity levels, but
    small improvements in recovery and
    workload balance may increase efficiency.
    """)

        else:

            st.error("""
    🚨 Productivity Risk Detected

    AI Analysis:
    Sleep debt and workload imbalance
    may negatively impact productivity,
    focus, and mental recovery.
    """)

        if sleep_input < 6:

            st.warning("""⚠ Your sleep duration is below the recommended range.
        AI Recommendation:Increase sleep duration to improve focus,memory retention, and productivity.""")

        elif sleep_input >= 7:
            st.success("""✅ Healthy sleep duration detected.
        AI Insight:Your current sleep schedule supports stronger productivity and recovery.""")

        if work_input > 10:
            st.error("""🚨 High workload detected.
    Long work hours combined with insufficient
    sleep may increase burnout risk.
    """)
        
        sleep_score = min(
    int((sleep_input / 8) * 100),
    100)

        st.metric(
    "😴 Sleep Health Score",
    f"{sleep_score}/100")

        fig_pred = go.Figure(go.Indicator(

        mode="gauge+number",

        value=pred_output[0],

        title={'text': "Predicted Productivity"},

        gauge={
        'axis': {'range': [0, 10]},
        'bar': {'color': "#60A5FA"}
    }))

        fig_pred.update_layout(
    template="plotly_dark",
    height=400
    )

        st.plotly_chart(
    fig_pred,
    use_container_width=True,
    key='18'
    )
        risk_score = 100 - sleep_score

        fig_risk = go.Figure(go.Indicator(

        mode="gauge+number",

        value=risk_score,

        title={'text': "Burnout Risk Score"},

        gauge={

        'axis': {'range': [0, 100]},

        'bar': {'color': "#EF4444"},

        'steps': [

            {'range': [0, 30], 'color': "#10B981"},
            {'range': [30, 70], 'color': "#F59E0B"},
            {'range': [70, 100], 'color': "#EF4444"}

        ]
    }
))

        fig_risk.update_layout(
    template="plotly_dark",
    height=350
    )

        st.plotly_chart(
    fig_risk,
    use_container_width=True,
    key='19'
    )

# --- Tab 7: Conclusion ---
with tabs[6]:
    st.header("📌 Conclusion")
    st.markdown("""
    - **Sleep matters**: We found that 7-8 hours is the sweet spot for productivity.
    - **Workload affects sleep debt**: More hours → less sleep → lower productivity.
    - **Professions vary**: Doctors, tech workers face higher sleep debt.

    ### ✅ Takeaways
    - Balance sleep and work for optimal performance.
    - Companies can use this data for wellness strategies.
    - Individuals can plan routines around these insights.

    Thank you for viewing this project! 🙏
    """)

st.markdown("""
<br><br><br>

<div style='text-align:center;
color:#9CA3AF;
padding:20px;
font-size:14px;'>

🌙 <b>SleepSync AI</b><br>

Built using Streamlit • Plotly • Scikit-Learn

</div>
""", unsafe_allow_html=True)
st.caption("© 2025 Chirudeep Bandapalli")
