import streamlit as st
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Collectivault - Sports Memorabilia Rating System",
    page_icon="🏆",
    layout="wide"
)

# Title and description
st.title("🏆 Collectivault Rating System")
st.markdown("### Automated ML-Based Rating & Valuation for Sports Memorabilia")
st.markdown("---")

# Sidebar for category selection
st.sidebar.header("Configuration")

# Load data to get actual categories
@st.cache_data
def load_categories():
    """Load unique categories from the Excel file"""
    try:
        df = pd.read_excel("Collectivault_data.xlsx")
        categories = sorted(df['Category'].unique().tolist())
        return categories
    except Exception as e:
        st.sidebar.error(f"Error loading categories: {str(e)}")
        return ["Badminton", "Baseball", "Cricket Bats", "Cricket Medals", "Cricket Other Items", "F1 Racing", "Hockey"]

CATEGORIES = load_categories()

st.sidebar.info(f"**{len(CATEGORIES)}** categories available")
selected_category = st.sidebar.selectbox("Select Product Category", CATEGORIES)

# Load model function
@st.cache_resource
def load_model(category):
    """Load the trained model for the selected category"""
    try:
        model_path = Path(f"models/{category.replace(' ', '_').lower()}_model.pkl")
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.warning(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Main content
st.header(f"Rate Your {selected_category}")

# Display category statistics
col_stat1, col_stat2, col_stat3 = st.columns(3)
try:
    df = pd.read_excel("Collectivault_data.xlsx")
    category_df = df[df['Category'] == selected_category]
    with col_stat1:
        st.metric("Items in Database", len(category_df))
    with col_stat2:
        avg_value = category_df['Value'].mean()
        st.metric("Avg Value", f"${avg_value:,.0f}")
    with col_stat3:
        avg_rating = category_df['OptimizedRatingScore'].mean()
        st.metric("Avg Rating", f"{avg_rating:.1f}/100")
except:
    pass

st.markdown("---")

# Create input form
with st.form("rating_form"):
    st.subheader("Enter Product Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.text_input("Brand/Manufacturer", placeholder="e.g., Gray-Nicolls, Nike, Adidas")
        condition = st.selectbox("Condition", ["Mint", "Excellent", "Good", "Fair", "Poor"])
        age_years = st.number_input("Age (Years)", min_value=0, max_value=150, value=5, help="Age affects both rating and valuation")
        rarity_score = st.slider("Rarity Score", min_value=1, max_value=10, value=5, help="1=Common, 10=Extremely Rare | Affects both rating and value")
        
    with col2:
        autographed = st.checkbox("Autographed/Signed", help="Significantly increases both rating and value")
        match_used = st.checkbox("Match-Used/Game-Worn", help="Increases both rating and value")
        provenance = st.selectbox("Provenance Documentation", ["None", "Basic", "Certificate", "Full Documentation"], help="Better documentation = higher rating and value")
        market_demand = st.slider("Market Demand", min_value=1, max_value=10, value=5, help="Current market interest level | Affects both rating and value")
    
    st.subheader("Category-Specific Details")
    
    bat_weight = None
    wood_type = None
    player_name = None
    driver_name = None
    
    if "Cricket Bats" in selected_category:
        bat_weight = st.number_input("Weight (kg)", min_value=0.5, max_value=2.0, value=1.2, step=0.1, help="Bat weight affects valuation")
        wood_type = st.selectbox("Wood Type", ["English Willow", "Kashmir Willow", "Other"], help="Wood quality significantly affects both rating and value")
    elif "Cricket" in selected_category:
        player_name = st.text_input("Player Name (if applicable)", placeholder="e.g., Sachin Tendulkar", help="Famous player association increases value")
    elif "F1" in selected_category:
        driver_name = st.text_input("Driver Name (if applicable)", placeholder="e.g., Lewis Hamilton", help="Famous driver association increases value")
    elif "Baseball" in selected_category:
        player_name = st.text_input("Player Name (if applicable)", placeholder="e.g., Babe Ruth", help="Famous player association increases value")
    elif "Badminton" in selected_category:
        player_name = st.text_input("Player Name (if applicable)", placeholder="e.g., P.V. Sindhu", help="Famous player association increases value")
    elif "Hockey" in selected_category:
        player_name = st.text_input("Player Name (if applicable)", placeholder="e.g., Wayne Gretzky", help="Famous player association increases value")
    
    submitted = st.form_submit_button("🎯 Calculate Rating & Valuation", use_container_width=True)

# Process the form when submitted
if submitted:
    try:
        # Convert ALL user inputs to feature scores (0-100 scale)
        features = {
            'ConditionScore': {'Mint': 100, 'Excellent': 85, 'Good': 70, 'Fair': 60, 'Poor': 40}[condition],
            'HistoricalScore': min(100, age_years * 1.0),
            'SignedScore': 100 if autographed else 60,
            'WornScore': 100 if match_used else 70,
            'MarketDemandScore': market_demand * 10,
            'RarityScore': rarity_score * 10,
            'ProvenanceScore': {'None': 0, 'Basic': 33, 'Certificate': 66, 'Full Documentation': 100}[provenance]
        }
        
        celebrity_bonus = 0
        if player_name or driver_name:
            celebrity_bonus = 15
        
        if "Cricket Bats" in selected_category and bat_weight and wood_type:
            features['BatWeightScore'] = min(100, bat_weight * 50)
            features['WoodQualityScore'] = {'English Willow': 100, 'Kashmir Willow': 70, 'Other': 50}[wood_type]
        
        model = load_model(selected_category)
        
        if model is not None:
            feature_df = pd.DataFrame([features])
            
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                available_features = [f for f in expected_features if f in feature_df.columns]
                feature_df_filtered = feature_df[available_features]
                missing_features = set(expected_features) - set(feature_df.columns)
                if missing_features:
                    st.info(f"ℹ️ Some advanced features not used: {', '.join(missing_features)}")
            else:
                feature_df_filtered = feature_df
            
            predicted_value = model.predict(feature_df_filtered)[0]
            
            if celebrity_bonus > 0:
                predicted_value = predicted_value * (1 + celebrity_bonus / 100)
            
            rating_weights = {
                'ConditionScore': 0.18,
                'HistoricalScore': 0.12,
                'SignedScore': 0.15,
                'WornScore': 0.10,
                'MarketDemandScore': 0.15,
                'RarityScore': 0.15,
                'ProvenanceScore': 0.10,
                'BatWeightScore': 0.03,
                'WoodQualityScore': 0.02
            }
            
            rating_score = 0
            total_weight = 0
            feature_impacts = []
            
            for feature, score in features.items():
                weight = rating_weights.get(feature, 0)
                if weight > 0:
                    contribution = score * weight
                    rating_score += contribution
                    total_weight += weight
                    feature_impacts.append({
                        'Feature': feature.replace('Score', ''),
                        'Score': f"{score:.1f}",
                        'Weight': f"{weight*100:.1f}%",
                        'Contribution': contribution
                    })
            
            if total_weight > 0:
                rating_score = rating_score / total_weight
            
            if celebrity_bonus > 0:
                rating_score = min(100, rating_score * (1 + celebrity_bonus / 200))
            
            rating_score = max(0, min(100, rating_score))
            
            if rating_score >= 90:
                grade = "AAA"
            elif rating_score >= 80:
                grade = "AA"
            elif rating_score >= 70:
                grade = "A"
            elif rating_score >= 60:
                grade = "BBB"
            else:
                grade = "BB"
            
            st.success("✅ Rating & Valuation Calculated Successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Overall Rating", f"{rating_score:.1f}/100", help="Comprehensive quality score based on all features")
            with col2:
                st.metric("💰 Estimated Value", f"${predicted_value:,.2f}", help="Predicted market value from Random Forest model")
            with col3:
                st.metric("🏆 Rating Band", grade, help="AAA=Exceptional, AA=Excellent, A=Good, BBB=Fair, BB=Poor")
            
            if celebrity_bonus > 0:
                st.info(f"⭐ Celebrity Association Bonus: +{celebrity_bonus}% to value, +{celebrity_bonus/2}% to rating")
            
            st.markdown("---")
            st.subheader("📈 Rating Breakdown - How Each Feature Contributes")
            
            feature_impacts_sorted = sorted(feature_impacts, key=lambda x: x['Contribution'], reverse=True)
            
            col_table, col_chart = st.columns([1, 1])
            
            with col_table:
                impact_df = pd.DataFrame(feature_impacts_sorted)
                impact_df['Contribution'] = impact_df['Contribution'].apply(lambda x: f"{x:.2f}")
                st.dataframe(impact_df, use_container_width=True, hide_index=True)
            
            with col_chart:
                fig = go.Figure(data=[go.Bar(
                    y=[f['Feature'] for f in feature_impacts_sorted],
                    x=[f['Contribution'] for f in feature_impacts_sorted],
                    orientation='h',
                    marker=dict(color=[f['Contribution'] for f in feature_impacts_sorted], colorscale='Viridis'),
                    text=[f"{f['Contribution']:.1f}" for f in feature_impacts_sorted],
                    textposition='auto',
                )])
                fig.update_layout(title="Feature Impact on Rating", xaxis_title="Weighted Contribution", yaxis_title="Feature", height=400, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Individual Feature Scores (0-100 Scale)")
            
            score_data = [{'Feature': feature.replace('Score', ''), 'Score': score} for feature, score in features.items()]
            
            fig2 = go.Figure(data=[go.Bar(
                x=[s['Feature'] for s in score_data],
                y=[s['Score'] for s in score_data],
                marker=dict(color=[s['Score'] for s in score_data], colorscale='RdYlGn', cmin=0, cmax=100),
                text=[f"{s['Score']:.0f}" for s in score_data],
                textposition='auto',
            )])
            fig2.update_layout(title="Raw Feature Scores", xaxis_title="Feature", yaxis_title="Score (0-100)", yaxis=dict(range=[0, 100]), height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("---")
            st.subheader("📊 Similar Items in Database")
            
            try:
                df = pd.read_excel("Collectivault_data.xlsx")
                category_df = df[df['Category'] == selected_category].head(5)
                st.dataframe(category_df[['ID', 'Item', 'Value', 'OptimizedRatingScore', 'OptimizedRatingBand']], use_container_width=True, hide_index=True)
            except:
                pass
            
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if rating_score < 60:
                st.info("Consider professional restoration or obtaining better provenance documentation to increase value.")
            elif rating_score < 80:
                st.info("Good collectible! Proper storage and documentation could further enhance its value.")
            else:
                st.success("Excellent item! Consider professional authentication and insurance for high-value pieces.")
                
    except Exception as e:
        st.error(f"❌ Error calculating rating: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("**Collectivault ML Rating System** | Powered by Random Forest | All Sports Coverage")
st.caption("Dissertation Project - Sports Memorabilia Valuation System")
