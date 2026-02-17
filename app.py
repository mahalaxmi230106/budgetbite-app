import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime

# Page config
st.set_page_config(page_title="BudgetBite Pro", page_icon="🍽️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .big-font {font-size:20px !important; font-weight: bold;}
    .highlight {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 15px; border-radius: 10px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("meal_data.csv")

df = load_data()

# Build ML model
@st.cache_resource
def build_ml_model(df):
    df_copy = df.copy()
    df_copy["combined_features"] = df_copy["Cuisine"] + " " + df_copy["Veg_NonVeg"] + " " + df_copy["Dish_Name"]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_copy["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim, df_copy

tfidf_matrix, cosine_sim, df_features = build_ml_model(df)

# Session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Header
st.markdown("<h1 style='text-align: center; color: #FF6B6B;'>🍽️ BudgetBite Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>AI-Powered Smart Meal Recommender</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://via.placeholder.com/300x100/FF6B6B/FFFFFF?text=BudgetBite", use_column_width=True)
st.sidebar.header("⚙️ Preferences")
budget = st.sidebar.slider("💰 Budget (Rs.)", 30, 250, 100)
veg_pref = st.sidebar.selectbox("🥗 Diet", ["Any", "Veg", "Non-Veg"])
cuisine_pref = st.sidebar.selectbox("🍜 Cuisine", ["Any"] + sorted(df["Cuisine"].unique().tolist()))
max_distance = st.sidebar.slider("📍 Max Distance (km)", 0.0, 2.0, 1.5, 0.1)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Find Meals", 
    "📊 Visualizations", 
    "🤖 ML Similar Dishes",
    "🏪 Compare Restaurants",
    "📈 Price Trends",
    "⭐ My Favorites"
])

# TAB 1: Find Meals
with tab1:
    st.header("Your Personalized Recommendations")

    filtered = df[(df["Price"] <= budget) & (df["Distance_km"] <= max_distance)].copy()

    if veg_pref != "Any":
        filtered = filtered[filtered["Veg_NonVeg"] == veg_pref]
    if cuisine_pref != "Any":
        filtered = filtered[filtered["Cuisine"] == cuisine_pref]

    if len(filtered) == 0:
        st.warning("😞 No meals found! Try adjusting filters.")
    else:
        filtered["rating_score"] = filtered["Rating"] / 5.0
        filtered["price_score"] = 1 - (filtered["Price"] / budget)
        max_dist = filtered["Distance_km"].max()
        filtered["distance_score"] = 1 - (filtered["Distance_km"] / max_dist) if max_dist > 0 else 1
        filtered["final_score"] = (
            filtered["rating_score"] * 0.4 +
            filtered["price_score"] * 0.3 +
            filtered["distance_score"] * 0.3
        )

        top_5 = filtered.nlargest(5, "final_score").reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Matches Found", len(filtered))
        with col2:
            st.metric("Avg Price", f"Rs.{filtered['Price'].mean():.0f}")
        with col3:
            st.metric("Avg Rating", f"⭐{filtered['Rating'].mean():.1f}")

        st.success(f"✅ Top 5 Recommendations for you:")

        for idx, row in top_5.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"### {idx+1}. {row['Dish_Name']}")
                    st.write(f"📍 **{row['Restaurant_Name']}**")
                    st.write(f"🍽️ {row['Cuisine']} | {row['Veg_NonVeg']}")
                with col2:
                    st.metric("Price", f"Rs.{row['Price']}")
                with col3:
                    st.metric("Rating", f"⭐{row['Rating']}")
                    st.write(f"📏 {row['Distance_km']} km")
                with col4:
                    st.metric("Score", f"{row['final_score']:.2f}")
                    dish_id = f"{row['Restaurant_Name']}_{row['Dish_Name']}"
                    if st.button("❤️ Save", key=f"fav_{idx}"):
                        if dish_id not in st.session_state.favorites:
                            st.session_state.favorites.append({
                                'dish': row['Dish_Name'],
                                'restaurant': row['Restaurant_Name'],
                                'price': row['Price'],
                                'rating': row['Rating']
                            })
                            st.success("Added to favorites!")
                st.markdown("---")

# TAB 2: Visualizations
with tab2:
    st.header("📊 Data Insights & Analytics")

    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig1 = px.histogram(df, x="Price", nbins=15, title="Price Distribution",
                           color_discrete_sequence=["#FF6B6B"])
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Cuisine distribution
        cuisine_counts = df["Cuisine"].value_counts()
        fig3 = px.bar(x=cuisine_counts.index, y=cuisine_counts.values,
                     title="Dishes by Cuisine Type",
                     labels={'x': 'Cuisine', 'y': 'Count'},
                     color=cuisine_counts.values,
                     color_continuous_scale="Viridis")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Veg vs Non-Veg
        veg_counts = df["Veg_NonVeg"].value_counts()
        fig2 = px.pie(values=veg_counts.values, names=veg_counts.index,
                     title="Veg vs Non-Veg Distribution",
                     color_discrete_sequence=["#90EE90", "#FFB6C1"])
        st.plotly_chart(fig2, use_container_width=True)

        # Price vs Rating scatter
        fig4 = px.scatter(df, x="Price", y="Rating", color="Veg_NonVeg",
                         size="Distance_km", hover_data=["Dish_Name", "Restaurant_Name"],
                         title="Price vs Rating Analysis",
                         color_discrete_map={"Veg": "green", "Non-Veg": "red"})
        st.plotly_chart(fig4, use_container_width=True)

    # Restaurant ratings
    st.subheader("🏆 Top Rated Restaurants")
    restaurant_ratings = df.groupby("Restaurant_Name")["Rating"].mean().sort_values(ascending=False)
    fig5 = px.bar(x=restaurant_ratings.index, y=restaurant_ratings.values,
                 title="Average Ratings by Restaurant",
                 labels={'x': 'Restaurant', 'y': 'Avg Rating'},
                 color=restaurant_ratings.values,
                 color_continuous_scale="RdYlGn")
    st.plotly_chart(fig5, use_container_width=True)

# TAB 3: ML Similar Dishes
with tab3:
    st.header("🤖 ML-Powered Similar Dish Finder")
    st.write("Find dishes similar to what you like using Machine Learning!")

    selected_dish = st.selectbox("Select a dish you like:", df["Dish_Name"].unique())
    similarity_budget = st.slider("Budget for similar dishes (Rs.)", 30, 250, 150)

    if st.button("Find Similar Dishes"):
        idx = df[df["Dish_Name"] == selected_dish].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        similar_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]

        similar_dishes = df.iloc[similar_indices].copy()
        similar_dishes = similar_dishes[similar_dishes["Price"] <= similarity_budget]
        similar_dishes["Similarity"] = similarity_scores[:len(similar_dishes)]

        if len(similar_dishes) > 0:
            st.success(f"Found {len(similar_dishes)} similar dishes!")

            for idx, row in similar_dishes.head(5).iterrows():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {row['Dish_Name']}")
                    st.write(f"🏪 {row['Restaurant_Name']} | {row['Cuisine']}")
                with col2:
                    st.metric("Price", f"Rs.{row['Price']}")
                    st.metric("Rating", f"⭐{row['Rating']}")
                with col3:
                    similarity_pct = row['Similarity'] * 100
                    st.metric("Similarity", f"{similarity_pct:.0f}%")
                st.markdown("---")
        else:
            st.warning("No similar dishes found within budget!")

# TAB 4: Compare Restaurants
with tab4:
    st.header("🏪 Restaurant Comparison Tool")

    restaurants = df["Restaurant_Name"].unique().tolist()

    col1, col2 = st.columns(2)
    with col1:
        restaurant1 = st.selectbox("Select Restaurant 1:", restaurants)
    with col2:
        restaurant2 = st.selectbox("Select Restaurant 2:", 
                                   [r for r in restaurants if r != restaurant1])

    if st.button("Compare Restaurants"):
        r1_data = df[df["Restaurant_Name"] == restaurant1]
        r2_data = df[df["Restaurant_Name"] == restaurant2]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📍 {restaurant1}")
            st.metric("Dishes Offered", len(r1_data))
            st.metric("Avg Price", f"Rs.{r1_data['Price'].mean():.0f}")
            st.metric("Avg Rating", f"⭐{r1_data['Rating'].mean():.2f}")
            st.metric("Price Range", f"Rs.{r1_data['Price'].min()}-{r1_data['Price'].max()}")

            st.write("**Cuisines:**", ", ".join(r1_data["Cuisine"].unique()))
            veg_count = len(r1_data[r1_data["Veg_NonVeg"] == "Veg"])
            st.write(f"**Veg Options:** {veg_count}/{len(r1_data)}")

        with col2:
            st.subheader(f"📍 {restaurant2}")
            st.metric("Dishes Offered", len(r2_data))
            st.metric("Avg Price", f"Rs.{r2_data['Price'].mean():.0f}")
            st.metric("Avg Rating", f"⭐{r2_data['Rating'].mean():.2f}")
            st.metric("Price Range", f"Rs.{r2_data['Price'].min()}-{r2_data['Price'].max()}")

            st.write("**Cuisines:**", ", ".join(r2_data["Cuisine"].unique()))
            veg_count = len(r2_data[r2_data["Veg_NonVeg"] == "Veg"])
            st.write(f"**Veg Options:** {veg_count}/{len(r2_data)}")

        # Comparison chart
        comparison_df = pd.DataFrame({
            'Restaurant': [restaurant1, restaurant2],
            'Avg Price': [r1_data['Price'].mean(), r2_data['Price'].mean()],
            'Avg Rating': [r1_data['Rating'].mean(), r2_data['Rating'].mean()],
            'Menu Size': [len(r1_data), len(r2_data)]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(name='Avg Price', x=comparison_df['Restaurant'], 
                            y=comparison_df['Avg Price']))
        fig.add_trace(go.Bar(name='Avg Rating (×20)', x=comparison_df['Restaurant'], 
                            y=comparison_df['Avg Rating']*20))
        fig.update_layout(title="Restaurant Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: Price Trends
with tab5:
    st.header("📈 Price Trend Analysis")

    # Price by cuisine
    st.subheader("Average Price by Cuisine Type")
    cuisine_prices = df.groupby("Cuisine")["Price"].mean().sort_values()
    fig1 = px.bar(x=cuisine_prices.values, y=cuisine_prices.index, 
                 orientation='h',
                 title="Which Cuisine is Most Affordable?",
                 labels={'x': 'Average Price (Rs.)', 'y': 'Cuisine'},
                 color=cuisine_prices.values,
                 color_continuous_scale="RdYlGn_r")
    st.plotly_chart(fig1, use_container_width=True)

    # Price distribution by veg/non-veg
    st.subheader("Price Comparison: Veg vs Non-Veg")
    fig2 = px.box(df, x="Veg_NonVeg", y="Price", color="Veg_NonVeg",
                 title="Price Distribution",
                 color_discrete_map={"Veg": "green", "Non-Veg": "red"})
    st.plotly_chart(fig2, use_container_width=True)

    # Best value analysis
    st.subheader("💎 Best Value Dishes (High Rating, Low Price)")
    df_value = df.copy()
    df_value["Value_Score"] = (df_value["Rating"] / 5.0) / (df_value["Price"] / df_value["Price"].max())
    best_value = df_value.nlargest(10, "Value_Score")

    fig3 = px.scatter(best_value, x="Price", y="Rating", size="Value_Score",
                     hover_data=["Dish_Name", "Restaurant_Name"],
                     title="Best Value Dishes",
                     color="Value_Score",
                     color_continuous_scale="Viridis")
    st.plotly_chart(fig3, use_container_width=True)

# TAB 6: Favorites
with tab6:
    st.header("⭐ My Favorite Dishes")

    if len(st.session_state.favorites) == 0:
        st.info("No favorites yet! Go to 'Find Meals' tab and save some dishes.")
    else:
        st.success(f"You have {len(st.session_state.favorites)} favorite dishes!")

        for idx, fav in enumerate(st.session_state.favorites):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"### {fav['dish']}")
                st.write(f"📍 {fav['restaurant']}")
            with col2:
                st.metric("Price", f"Rs.{fav['price']}")
            with col3:
                st.metric("Rating", f"⭐{fav['rating']}")
                if st.button("Remove", key=f"remove_{idx}"):
                    st.session_state.favorites.pop(idx)
                    st.rerun()
            st.markdown("---")

        # Favorites statistics
        if st.session_state.favorites:
            total_cost = sum([f['price'] for f in st.session_state.favorites])
            avg_rating = sum([f['rating'] for f in st.session_state.favorites]) / len(st.session_state.favorites)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Cost (if you order all)", f"Rs.{total_cost}")
            with col2:
                st.metric("Average Rating", f"⭐{avg_rating:.2f}")

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>
    🍽️ BudgetBite Pro - Powered by Machine Learning | Built with Python, Streamlit & Scikit-learn
</p>
""", unsafe_allow_html=True)

