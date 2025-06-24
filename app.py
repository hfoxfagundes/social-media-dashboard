import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans

# Set page config FIRST
st.set_page_config(page_title="Social Media & Student Dashboard", layout="wide")

# Load dataset
df = pd.read_csv("social_media_dataset.csv")
df.drop(columns=['Student_ID'], inplace=True, errors='ignore')

# Convert mental health score to category
def categorize_mental_health(score):
    if score <= 3:
        return "Poor"
    elif score <= 6:
        return "Fair"
    elif score <= 8:
        return "Good"
    else:
        return "Excellent"

df['Mental Health Category'] = df['Mental_Health_Score'].apply(categorize_mental_health)
df['Affects_Academic_Performance_Binary'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})

# App title
st.title("📱 Social Media Impact on Students")
st.markdown("This dashboard analyzes how social media habits relate to students' mental health, academics, and relationships.")

# Tabs layout
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Usage & Sleep",
    "2. Conflicts",
    "3. Platforms",
    "4. Clustering",
    "5. Academic Performance",
    "6. Country Usage",
    "7. Relationships"
])

with tab1:
    st.subheader("1. Usage vs Sleep & Mental Health")
    mh_toggle = st.radio("Color by Mental Health?", ["Yes", "No"])
    if mh_toggle == "Yes":
        fig1 = px.scatter(df, x='Avg_Daily_Usage_Hours', y='Sleep_Hours_Per_Night', color='Mental Health Category')
    else:
        fig1 = px.scatter(df, x='Avg_Daily_Usage_Hours', y='Sleep_Hours_Per_Night')
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.subheader("2. Relationship Conflicts")
    rel_status = st.selectbox("Select Relationship Status:", ["All"] + df['Relationship_Status'].dropna().unique().tolist())
    rel_df = df if rel_status == "All" else df[df['Relationship_Status'] == rel_status]
    fig2 = px.box(rel_df, x='Relationship_Status', y='Conflicts_Over_Social_Media')
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("3. Most Used Platforms by Country")
    country_input = st.text_input("Enter countries separated by commas (e.g., India, Canada):")
    selected_countries = [c.strip() for c in country_input.split(",") if c.strip()] if country_input else df['Country'].unique().tolist()
    filtered_df = df[df['Country'].isin(selected_countries)]
    fig3 = px.histogram(filtered_df, x='Most_Used_Platform', color='Country', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    st.subheader("4. Clustering by Usage & Mental Health")
    cluster_n = st.slider("Number of clusters:", min_value=2, max_value=10, value=4)
    cluster_data = df[['Avg_Daily_Usage_Hours', 'Mental_Health_Score']].dropna()
    if len(cluster_data) >= cluster_n:
        kmeans = KMeans(n_clusters=cluster_n, random_state=42)
        cluster_data['Cluster'] = kmeans.fit_predict(cluster_data)
        fig4 = px.scatter(cluster_data, x='Avg_Daily_Usage_Hours', y='Mental_Health_Score', color='Cluster')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Not enough data for clustering.")

with tab5:
    st.subheader("5. Addiction vs Academic Performance")
    level_options = df['Academic_Level'].dropna().unique().tolist()
    selected_level = st.selectbox("Select Academic Level:", ["All"] + level_options)
    academic_df = df if selected_level == "All" else df[df['Academic_Level'] == selected_level]
    fig5 = px.box(academic_df, x='Affects_Academic_Performance', y='Addicted_Score')
    
    st.plotly_chart(fig5, use_container_width=True)

with tab6:
    st.subheader("6. Average Usage by Country")
    country_input2 = st.text_input("Filter Country (e.g., US, UK):", key="country_input2")
    selected_countries2 = [c.strip() for c in country_input2.split(",") if c.strip()] if country_input2 else df['Country'].unique().tolist()
    filtered_df2 = df[df['Country'].isin(selected_countries2)]
    country_usage = filtered_df2.groupby('Country')['Avg_Daily_Usage_Hours'].mean().reset_index()
    fig6 = px.bar(country_usage, x='Country', y='Avg_Daily_Usage_Hours')
    st.plotly_chart(fig6, use_container_width=True)

with tab7:
    st.subheader("7. Usage & Mental Health by Relationship Status")
    rel_status_7 = st.selectbox("Select Relationship Status:", ["All"] + df['Relationship_Status'].dropna().unique().tolist(), key="rel_status_7")
    rel_df_7 = df if rel_status_7 == "All" else df[df['Relationship_Status'] == rel_status_7]
    fig7 = px.box(rel_df_7, x='Relationship_Status', y='Avg_Daily_Usage_Hours', color='Mental Health Category')
    st.plotly_chart(fig7, use_container_width=True)


