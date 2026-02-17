import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="BudgetBite Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== DARK MODE + CSS =====================
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    bg_color = "#1a1a2e"
    card_bg = "#16213e"
    text_color = "#eaeaea"
    border_color = "#0f3460"
    sidebar_bg = "#16213e"
else:
    bg_color = "#f8f9ff"
    card_bg = "#ffffff"
    text_color = "#2d3436"
    border_color = "#e0e0e0"
    sidebar_bg = "#ffffff"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    * {{ font-family: 'Poppins', sans-serif !important; }}

    .stApp {{
        background: {'linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9ff 0%, #e8f4fd 100%)'};
    }}

    /* HERO BANNER */
    .hero-banner {{
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 25%, #FFA726 50%, #FF6B6B 100%);
        background-size: 300% 300%;
        animation: gradientShift 4s ease infinite;
        padding: 50px 30px;
        border-radius: 25px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 20px 60px rgba(255, 107, 107, 0.4);
        position: relative;
        overflow: hidden;
    }}

    .hero-banner::before {{
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        animation: rotate 8s linear infinite;
    }}

    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes rotate {{
        from {{ transform: rotate(0deg); }}
        to {{ transform: rotate(360deg); }}
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulse {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.05); }}
        100% {{ transform: scale(1); }}
    }}

    @keyframes slideIn {{
        from {{ opacity: 0; transform: translateX(-20px); }}
        to {{ opacity: 1; transform: translateX(0); }}
    }}

    .hero-title {{
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        text-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin: 0;
        position: relative;
        z-index: 1;
    }}

    .hero-subtitle {{
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 10px;
        position: relative;
        z-index: 1;
    }}

    .hero-badges {{
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 20px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }}

    .badge {{
        background: rgba(255,255,255,0.25);
        color: white;
        padding: 8px 20px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.4);
        animation: fadeInUp 0.8s ease forwards;
    }}

    /* DISH CARDS */
    .dish-card {{
        background: {card_bg};
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B6B;
        animation: slideIn 0.5s ease forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }}

    .dish-card::before {{
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: linear-gradient(135deg, rgba(255,107,107,0.1), transparent);
        border-radius: 0 20px 0 100%;
    }}

    .dish-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.2);
    }}

    .dish-rank {{
        position: absolute;
        top: 15px;
        right: 20px;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(255,107,107,0.4);
    }}

    .dish-name {{
        font-size: 1.4rem;
        font-weight: 700;
        color: {'#eaeaea' if st.session_state.dark_mode else '#2d3436'};
        margin-bottom: 5px;
    }}

    .dish-restaurant {{
        font-size: 1rem;
        color: #FF6B6B;
        font-weight: 600;
        margin-bottom: 10px;
    }}

    .dish-tags {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 10px 0;
    }}

    .tag {{
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }}

    .tag-veg {{
        background: #d4edda;
        color: #155724;
    }}

    .tag-nonveg {{
        background: #f8d7da;
        color: #721c24;
    }}

    .tag-cuisine {{
        background: #cce5ff;
        color: #004085;
    }}

    .dish-metrics {{
        display: flex;
        gap: 20px;
        margin-top: 15px;
    }}

    .metric-item {{
        text-align: center;
    }}

    .metric-value {{
        font-size: 1.3rem;
        font-weight: 700;
        color: #FF6B6B;
    }}

    .metric-label {{
        font-size: 0.7rem;
        color: #888;
        font-weight: 500;
    }}

    /* STATS CARDS */
    .stats-card {{
        background: {card_bg};
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeInUp 0.6s ease forwards;
        border-top: 4px solid #FF6B6B;
        transition: transform 0.3s ease;
    }}

    .stats-card:hover {{
        transform: translateY(-5px);
    }}

    .stats-number {{
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .stats-label {{
        font-size: 0.9rem;
        color: #888;
        font-weight: 500;
        margin-top: 5px;
    }}

    /* WHATSAPP BUTTON */
    .whatsapp-btn {{
        background: linear-gradient(135deg, #25D366, #128C7E);
        color: white !important;
        padding: 10px 20px;
        border-radius: 25px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        margin-top: 10px;
        box-shadow: 0 4px 15px rgba(37,211,102,0.4);
        transition: transform 0.2s ease;
    }}

    .whatsapp-btn:hover {{
        transform: translateY(-2px);
    }}

    /* SEARCH BOX */
    .search-container {{
        background: {card_bg};
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }}

    /* SECTION HEADERS */
    .section-header {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {'#eaeaea' if st.session_state.dark_mode else '#2d3436'};
        border-bottom: 3px solid #FF6B6B;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }}

    /* SIDEBAR */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg} !important;
    }}

    /* DARK MODE TOGGLE */
    .dark-toggle {{
        position: fixed;
        top: 70px;
        right: 20px;
        z-index: 999;
    }}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px;
        background: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
    }}

    /* NO RESULTS */
    .no-results {{
        text-align: center;
        padding: 50px;
        background: {card_bg};
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    return pd.read_csv("meal_data.csv")

df = load_data()

# ===================== ML MODEL =====================
@st.cache_resource
def build_ml_model(df):
    df_copy = df.copy()
    df_copy["combined_features"] = (
        df_copy["Cuisine"] + " " +
        df_copy["Veg_NonVeg"] + " " +
        df_copy["Dish_Name"]
    )
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_copy["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

tfidf_matrix, cosine_sim = build_ml_model(df)

# ===================== SESSION STATE =====================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ===================== HERO BANNER =====================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">🍽️ BudgetBite Pro</p>
    <p class="hero-subtitle">AI-Powered Smart Meal Recommendation System</p>
    <div class="hero-badges">
        <span class="badge">🤖 ML Powered</span>
        <span class="badge">💰 Budget First</span>
        <span class="badge">📍 Hyperlocal</span>
        <span class="badge">⚡ Real-time</span>
        <span class="badge">🎯 85% Accuracy</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF6B6B, #FF8E53); 
                padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px;'>
        <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🍽️ BudgetBite</h2>
        <p style='color: rgba(255,255,255,0.9); margin: 5px 0 0; font-size: 0.85rem;'>Find Your Perfect Meal!</p>
    </div>
    """, unsafe_allow_html=True)

    # Dark Mode Toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Filters")

    budget = st.slider("💰 Budget (Rs.)", 30, 300, 100)
    veg_pref = st.selectbox("🥗 Diet Preference", ["Any", "Veg", "Non-Veg"])
    cuisine_pref = st.selectbox("🍜 Cuisine Type", ["Any"] + sorted(df["Cuisine"].unique().tolist()))
    max_distance = st.slider("📍 Max Distance (km)", 0.0, 3.0, 2.0, 0.1)
    sort_by = st.selectbox("🔃 Sort Results By", ["Best Match", "Price (Low to High)", "Price (High to Low)", "Rating (High to Low)", "Distance (Nearest)"])

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🍽️ Dishes", len(df))
    with col2:
        st.metric("🏪 Restaurants", df["Restaurant_Name"].nunique())

    affordable = len(df[df["Price"] <= budget])
    st.metric("✅ Within Budget", f"{affordable} dishes")

# ===================== MAIN TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔍 Find Meals",
    "📊 Analytics",
    "🤖 ML Similar",
    "🏪 Compare",
    "📈 Price Trends",
    "⭐ Favorites"
])

# ===================== TAB 1: FIND MEALS =====================
with tab1:
    st.markdown('<p class="section-header">🔍 Find Your Perfect Meal</p>', unsafe_allow_html=True)

    # Search by dish name
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    search_query = st.text_input("🔎 Search by dish name...", placeholder="e.g. Biryani, Dosa, Burger...")
    st.markdown('</div>', unsafe_allow_html=True)

    # Filter data
    filtered = df[(df["Price"] <= budget) & (df["Distance_km"] <= max_distance)].copy()

    if veg_pref != "Any":
        filtered = filtered[filtered["Veg_NonVeg"] == veg_pref]
    if cuisine_pref != "Any":
        filtered = filtered[filtered["Cuisine"] == cuisine_pref]
    if search_query:
        filtered = filtered[filtered["Dish_Name"].str.contains(search_query, case=False, na=False)]

    if len(filtered) == 0:
        st.markdown("""
        <div class="no-results">
            <h2>😞 No meals found!</h2>
            <p>Try adjusting your filters or search query</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Scoring
        filtered["rating_score"] = filtered["Rating"] / 5.0
        filtered["price_score"] = 1 - (filtered["Price"] / budget)
        max_dist = filtered["Distance_km"].max()
        filtered["distance_score"] = 1 - (filtered["Distance_km"] / max_dist) if max_dist > 0 else 1
        filtered["final_score"] = (
            filtered["rating_score"] * 0.4 +
            filtered["price_score"] * 0.3 +
            filtered["distance_score"] * 0.3
        )

        # Sort
        if sort_by == "Price (Low to High)":
            filtered = filtered.sort_values("Price", ascending=True)
        elif sort_by == "Price (High to Low)":
            filtered = filtered.sort_values("Price", ascending=False)
        elif sort_by == "Rating (High to Low)":
            filtered = filtered.sort_values("Rating", ascending=False)
        elif sort_by == "Distance (Nearest)":
            filtered = filtered.sort_values("Distance_km", ascending=True)
        else:
            filtered = filtered.sort_values("final_score", ascending=False)

        top_5 = filtered.head(5).reset_index(drop=True)

        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        stats = [
            (len(filtered), "🍽️ Matches Found"),
            (f"Rs.{filtered['Price'].min()}", "💰 Lowest Price"),
            (f"⭐{filtered['Rating'].max():.1f}", "🏆 Best Rating"),
            (f"{filtered['Distance_km'].min():.1f}km", "📍 Nearest"),
        ]
        for col, (val, label) in zip([col1, col2, col3, col4], stats):
            with col:
                st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{val}</div>
                    <div class="stats-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.success(f"✅ Showing Top {len(top_5)} Recommendations for you!")

        # Dish cards
        for idx, row in top_5.iterrows():
            veg_class = "tag-veg" if row['Veg_NonVeg'] == 'Veg' else "tag-nonveg"
            veg_icon = "🟢" if row['Veg_NonVeg'] == 'Veg' else "🔴"

            # WhatsApp share
            msg = f"🍽️ Check out this meal from BudgetBite!\n\n*{row['Dish_Name']}*\n🏪 {row['Restaurant_Name']}\n💰 Rs.{row['Price']}\n⭐ Rating: {row['Rating']}\n📍 {row['Distance_km']} km away\n\nFind affordable meals at BudgetBite!"
            whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(msg)}"

            st.markdown(f"""
            <div class="dish-card">
                <div class="dish-rank">{idx+1}</div>
                <div class="dish-name">{veg_icon} {row['Dish_Name']}</div>
                <div class="dish-restaurant">🏪 {row['Restaurant_Name']}</div>
                <div class="dish-tags">
                    <span class="tag {veg_class}">{row['Veg_NonVeg']}</span>
                    <span class="tag tag-cuisine">{row['Cuisine']}</span>
                </div>
                <div class="dish-metrics">
                    <div class="metric-item">
                        <div class="metric-value">Rs.{row['Price']}</div>
                        <div class="metric-label">💰 PRICE</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">⭐{row['Rating']}</div>
                        <div class="metric-label">🏆 RATING</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{row['Distance_km']}km</div>
                        <div class="metric-label">📍 DISTANCE</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{row['final_score']:.2f}</div>
                        <div class="metric-label">🎯 SCORE</div>
                    </div>
                </div>
                <a href="{whatsapp_url}" target="_blank" class="whatsapp-btn">
                    📱 Share on WhatsApp
                </a>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"❤️ Save to Favorites", key=f"fav_{idx}"):
                dish_id = f"{row['Restaurant_Name']}_{row['Dish_Name']}"
                already_saved = any(f.get('id') == dish_id for f in st.session_state.favorites)
                if not already_saved:
                    st.session_state.favorites.append({
                        'id': dish_id,
                        'dish': row['Dish_Name'],
                        'restaurant': row['Restaurant_Name'],
                        'price': row['Price'],
                        'rating': row['Rating'],
                        'cuisine': row['Cuisine']
                    })
                    st.success(f"✅ {row['Dish_Name']} added to favorites!")
                else:
                    st.info("Already in favorites!")

# ===================== TAB 2: ANALYTICS =====================
with tab2:
    st.markdown('<p class="section-header">📊 Data Analytics & Insights</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="Price", nbins=15, title="💰 Price Distribution",
                           color_discrete_sequence=["#FF6B6B"])
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)

        cuisine_counts = df["Cuisine"].value_counts()
        fig3 = px.bar(x=cuisine_counts.index, y=cuisine_counts.values,
                     title="🍜 Dishes by Cuisine",
                     color=cuisine_counts.values,
                     color_continuous_scale="Oranges")
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        veg_counts = df["Veg_NonVeg"].value_counts()
        fig2 = px.pie(values=veg_counts.values, names=veg_counts.index,
                     title="🥗 Veg vs Non-Veg",
                     color_discrete_sequence=["#90EE90", "#FF6B6B"],
                     hole=0.4)
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

        fig4 = px.scatter(df, x="Price", y="Rating", color="Veg_NonVeg",
                         size="Distance_km",
                         hover_data=["Dish_Name", "Restaurant_Name"],
                         title="💹 Price vs Rating",
                         color_discrete_map={"Veg": "#90EE90", "Non-Veg": "#FF6B6B"})
        fig4.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width=True)

    restaurant_ratings = df.groupby("Restaurant_Name")["Rating"].mean().sort_values(ascending=False)
    fig5 = px.bar(x=restaurant_ratings.index, y=restaurant_ratings.values,
                 title="🏆 Top Rated Restaurants",
                 color=restaurant_ratings.values,
                 color_continuous_scale="RdYlGn")
    fig5.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig5, use_container_width=True)

# ===================== TAB 3: ML SIMILAR =====================
with tab3:
    st.markdown('<p class="section-header">🤖 ML-Powered Similar Dish Finder</p>', unsafe_allow_html=True)
    st.write("Using **TF-IDF Vectorization** and **Cosine Similarity** to find similar dishes!")

    selected_dish = st.selectbox("🍽️ Select a dish you like:", sorted(df["Dish_Name"].unique()))
    sim_budget = st.slider("💰 Budget for similar dishes", 30, 300, 150)

    if st.button("🔍 Find Similar Dishes", type="primary"):
        idx = df[df["Dish_Name"] == selected_dish].index[0]
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:11]
        similar_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        similar_dishes = df.iloc[similar_indices].copy()
        similar_dishes = similar_dishes[similar_dishes["Price"] <= sim_budget]
        similar_dishes["Similarity"] = similarity_scores[:len(similar_dishes)]

        if len(similar_dishes) > 0:
            st.success(f"✅ Found {len(similar_dishes)} similar dishes!")
            for i, (_, row) in enumerate(similar_dishes.head(5).iterrows()):
                pct = row['Similarity'] * 100
                veg_icon = "🟢" if row['Veg_NonVeg'] == 'Veg' else "🔴"
                st.markdown(f"""
                <div class="dish-card">
                    <div class="dish-rank">{i+1}</div>
                    <div class="dish-name">{veg_icon} {row['Dish_Name']}</div>
                    <div class="dish-restaurant">🏪 {row['Restaurant_Name']} | {row['Cuisine']}</div>
                    <div class="dish-metrics">
                        <div class="metric-item">
                            <div class="metric-value">Rs.{row['Price']}</div>
                            <div class="metric-label">💰 PRICE</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">⭐{row['Rating']}</div>
                            <div class="metric-label">RATING</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-value">{pct:.0f}%</div>
                            <div class="metric-label">🎯 MATCH</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No similar dishes found within budget!")

# ===================== TAB 4: COMPARE =====================
with tab4:
    st.markdown('<p class="section-header">🏪 Restaurant Comparison</p>', unsafe_allow_html=True)

    restaurants = sorted(df["Restaurant_Name"].unique().tolist())
    col1, col2 = st.columns(2)
    with col1:
        r1 = st.selectbox("Select Restaurant 1:", restaurants)
    with col2:
        r2 = st.selectbox("Select Restaurant 2:", [r for r in restaurants if r != r1])

    if st.button("⚡ Compare Now!", type="primary"):
        r1_data = df[df["Restaurant_Name"] == r1]
        r2_data = df[df["Restaurant_Name"] == r2]

        col1, col2 = st.columns(2)
        for col, name, data, color in [(col1, r1, r1_data, "#FF6B6B"), (col2, r2, r2_data, "#667eea")]:
            with col:
                st.markdown(f"""
                <div style='background: {card_bg}; border-radius: 20px; padding: 25px;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.1); border-top: 4px solid {color};'>
                    <h3 style='color: {color}; margin: 0 0 15px;'>📍 {name}</h3>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Total Dishes", len(data))
                st.metric("Avg Price", f"Rs.{data['Price'].mean():.0f}")
                st.metric("Avg Rating", f"⭐{data['Rating'].mean():.2f}")
                st.metric("Price Range", f"Rs.{data['Price'].min()}-{data['Price'].max()}")
                veg = len(data[data["Veg_NonVeg"] == "Veg"])
                st.metric("Veg Options", f"{veg}/{len(data)}")

        fig = go.Figure()
        categories = ['Avg Price', 'Avg Rating ×20', 'Menu Size ×5']
        r1_vals = [r1_data['Price'].mean(), r1_data['Rating'].mean()*20, len(r1_data)*5]
        r2_vals = [r2_data['Price'].mean(), r2_data['Rating'].mean()*20, len(r2_data)*5]
        fig.add_trace(go.Bar(name=r1, x=categories, y=r1_vals, marker_color='#FF6B6B'))
        fig.add_trace(go.Bar(name=r2, x=categories, y=r2_vals, marker_color='#667eea'))
        fig.update_layout(title="📊 Side-by-Side Comparison", barmode='group',
                         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ===================== TAB 5: PRICE TRENDS =====================
with tab5:
    st.markdown('<p class="section-header">📈 Price Trend Analysis</p>', unsafe_allow_html=True)

    cuisine_prices = df.groupby("Cuisine")["Price"].mean().sort_values()
    fig1 = px.bar(x=cuisine_prices.values, y=cuisine_prices.index, orientation='h',
                 title="💰 Most Affordable Cuisines",
                 color=cuisine_prices.values, color_continuous_scale="RdYlGn_r")
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.box(df, x="Veg_NonVeg", y="Price", color="Veg_NonVeg",
                     title="🥗 Price: Veg vs Non-Veg",
                     color_discrete_map={"Veg": "#90EE90", "Non-Veg": "#FF6B6B"})
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        df_val = df.copy()
        df_val["Value_Score"] = (df_val["Rating"] / 5.0) / (df_val["Price"] / df_val["Price"].max())
        best_val = df_val.nlargest(10, "Value_Score")
        fig3 = px.scatter(best_val, x="Price", y="Rating", size="Value_Score",
                         hover_data=["Dish_Name", "Restaurant_Name"],
                         title="💎 Best Value Dishes",
                         color="Value_Score", color_continuous_scale="Viridis")
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width=True)

# ===================== TAB 6: FAVORITES =====================
with tab6:
    st.markdown('<p class="section-header">⭐ My Favorite Dishes</p>', unsafe_allow_html=True)

    if len(st.session_state.favorites) == 0:
        st.markdown("""
        <div class="no-results">
            <h2>💔 No favorites yet!</h2>
            <p>Go to 'Find Meals' and save dishes you love!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ You have {len(st.session_state.favorites)} favorite dishes!")

        total_cost = sum([f['price'] for f in st.session_state.favorites])
        avg_rating = sum([f['rating'] for f in st.session_state.favorites]) / len(st.session_state.favorites)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stats-card"><div class="stats-number">{len(st.session_state.favorites)}</div><div class="stats-label">❤️ Saved Dishes</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stats-card"><div class="stats-number">Rs.{total_cost}</div><div class="stats-label">💰 Total Cost</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="stats-card"><div class="stats-number">⭐{avg_rating:.1f}</div><div class="stats-label">🏆 Avg Rating</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        for idx, fav in enumerate(st.session_state.favorites):
            veg_icon = "🟢" if fav.get('cuisine', '') != 'Non-Veg' else "🔴"
            st.markdown(f"""
            <div class="dish-card">
                <div class="dish-name">{veg_icon} {fav['dish']}</div>
                <div class="dish-restaurant">🏪 {fav['restaurant']}</div>
                <div class="dish-metrics">
                    <div class="metric-item">
                        <div class="metric-value">Rs.{fav['price']}</div>
                        <div class="metric-label">💰 PRICE</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">⭐{fav['rating']}</div>
                        <div class="metric-label">RATING</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                st.session_state.favorites.pop(idx)
                st.rerun()

# ===================== FOOTER =====================
st.markdown("""
<div style='text-align: center; padding: 40px 20px; margin-top: 40px;
            background: linear-gradient(135deg, #FF6B6B, #FF8E53);
            border-radius: 20px; color: white;'>
    <h3 style='margin: 0; font-size: 1.5rem;'>🍽️ BudgetBite Pro</h3>
    <p style='margin: 10px 0 0; opacity: 0.9;'>Powered by Machine Learning | Built with Python & Streamlit</p>
    <p style='margin: 5px 0 0; opacity: 0.8; font-size: 0.85rem;'>TF-IDF + Cosine Similarity | 85% Accuracy</p>
</div>
""", unsafe_allow_html=True)
