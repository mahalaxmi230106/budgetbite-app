import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import math

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="BudgetBite Pro - GPS Enabled",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE =====================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False
if 'user_location' not in st.session_state:
    st.session_state.user_location = None

# ===================== CALCULATE DISTANCE =====================
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    R = 6371  # Earth's radius in kilometers
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * \
        math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# ===================== PROFESSIONAL CSS =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* GLASS MORPHISM EFFECTS */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 30px;
        margin: 20px 0;
    }
    
    /* MODERN HEADER */
    .modern-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 30px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .modern-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 1.4rem;
        margin-top: 15px;
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }
    
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 25px;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    
    .feature-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 10px 25px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* PROFESSIONAL DISH CARDS */
    .pro-dish-card {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        border-left: 6px solid #667eea;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .pro-dish-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    .dish-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 20px;
    }
    
    .dish-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2d3436;
        margin: 0;
    }
    
    .dish-rank-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 1.3rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .restaurant-name {
        font-size: 1.1rem;
        color: #667eea;
        font-weight: 700;
        margin: 5px 0 15px;
    }
    
    .dish-info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 20px 0;
    }
    
    .info-box {
        text-align: center;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 15px;
    }
    
    .info-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #667eea;
        margin-bottom: 5px;
    }
    
    .info-label {
        font-size: 0.75rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tag-container {
        display: flex;
        gap: 10px;
        margin: 15px 0;
        flex-wrap: wrap;
    }
    
    .dish-tag {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .tag-veg {
        background: #d4edda;
        color: #155724;
    }
    
    .tag-nonveg {
        background: #f8d7da;
        color: #721c24;
    }
    
    .tag-cuisine {
        background: #cce5ff;
        color: #004085;
    }
    
    /* GPS LOCATION BUTTON */
    .gps-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px 30px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin: 10px 0;
    }
    
    .gps-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* DIRECTIONS BUTTON */
    .directions-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        padding: 12px 25px;
        border-radius: 50px;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        margin: 10px 5px;
    }
    
    .directions-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .whatsapp-btn {
        background: linear-gradient(135deg, #25D366, #128C7E);
        color: white !important;
        padding: 12px 25px;
        border-radius: 50px;
        text-decoration: none !important;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(37, 211, 102, 0.3);
        transition: all 0.3s ease;
        margin: 10px 5px;
    }
    
    .whatsapp-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 211, 102, 0.5);
    }
    
    /* STATS CARDS */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        margin: 30px 0;
    }
    
    .stat-card {
        background: white;
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #667eea;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* LOCATION INFO */
    .location-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-weight: 600;
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 25px;
        color: white;
    }
    
    .sidebar-title {
        font-size: 1.8rem;
        font-weight: 900;
        margin: 0;
    }
    
    .sidebar-subtitle {
        font-size: 0.9rem;
        margin-top: 8px;
        opacity: 0.95;
    }
    
    /* NO RESULTS */
    .no-results {
        text-align: center;
        padding: 60px 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    }
    
    .no-results h2 {
        font-size: 2rem;
        color: #2d3436;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv("meal_data.csv")
    # Add GPS coordinates for restaurants (sample data - you can update with real coordinates)
    restaurant_coords = {
        'Hotel Shadab': (17.4435, 78.4728),
        'Paradise Restaurant': (17.4402, 78.4483),
        'Cafe Coffee Day': (17.4399, 78.4983),
        'Mehfil Restaurant': (17.4326, 78.4071),
        'Udupi Hotel': (17.4239, 78.4738),
        'Bawarchi Restaurant': (17.4283, 78.4394),
        'Street Food Hub': (17.4365, 78.4879),
        'Chinese Corner': (17.4394, 78.4526),
        'Pizza Palace': (17.4205, 78.4683),
        'Biryani House': (17.4123, 78.4391),
        'South Spice': (17.4078, 78.4528),
        'Punjabi Dhaba': (17.4312, 78.4621),
        'Andhra Spice': (17.4398, 78.4812),
        'Wrap and Roll': (17.4287, 78.4593),
    }
    
    df['Latitude'] = df['Restaurant_Name'].map(lambda x: restaurant_coords.get(x, (17.4400, 78.4800))[0])
    df['Longitude'] = df['Restaurant_Name'].map(lambda x: restaurant_coords.get(x, (17.4400, 78.4800))[1])
    
    return df

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

# ===================== HEADER =====================
st.markdown("""
<div class="modern-header">
    <h1 class="header-title">🍽️ BudgetBite Pro</h1>
    <p class="header-subtitle">AI-Powered Meal Discovery with GPS Navigation</p>
    <div class="feature-badges">
        <span class="feature-badge">🤖 ML Powered</span>
        <span class="feature-badge">📍 GPS Enabled</span>
        <span class="feature-badge">🗺️ Live Directions</span>
        <span class="feature-badge">💰 Budget First</span>
        <span class="feature-badge">⚡ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 class="sidebar-title">🍽️ BudgetBite</h2>
        <p class="sidebar-subtitle">Smart Meal Discovery</p>
    </div>
    """, unsafe_allow_html=True)
    
    # GPS LOCATION
    st.markdown("### 📍 Your Location")
    
    # Manual location input
    col1, col2 = st.columns(2)
    with col1:
        user_lat = st.number_input("Latitude", value=17.4400, format="%.4f", step=0.0001)
    with col2:
        user_lon = st.number_input("Longitude", value=78.4800, format="%.4f", step=0.0001)
    
    st.session_state.user_location = (user_lat, user_lon)
    
    st.markdown(f"""
    <div class="location-info">
        📍 Current Location:<br>
        Lat: {user_lat:.4f}, Lon: {user_lon:.4f}
    </div>
    """, unsafe_allow_html=True)
    
    st.info("💡 **Tip:** You can manually enter your GPS coordinates or use your device's location services!")
    
    st.markdown("---")
    st.markdown("### 🔍 Search Preferences")
    
    # Search Form
    with st.form("search_form"):
        budget = st.number_input("💰 Budget (Rs.)", 20, 500, 100, 10)
        veg_pref = st.selectbox("🥗 Diet", ["Any", "Veg", "Non-Veg"])
        cuisine_pref = st.selectbox("🍜 Cuisine", ["Any"] + sorted(df["Cuisine"].unique().tolist()))
        max_dist = st.number_input("📍 Max Distance (km)", 0.1, 10.0, 3.0, 0.1)
        search_query = st.text_input("🔎 Search Dish", placeholder="e.g. Biryani...")
        sort_by = st.selectbox("🔃 Sort By", [
            "Nearest First",
            "Best Match",
            "Price (Low to High)",
            "Rating (High to Low)"
        ])
        
        submitted = st.form_submit_button("🔍 FIND MEALS", type="primary", use_container_width=True)
        
        if submitted:
            st.session_state.search_clicked = True

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🍽️ Dishes", len(df))
    with col2:
        st.metric("🏪 Restaurants", df["Restaurant_Name"].nunique())

# ===================== MAIN CONTENT =====================
if st.session_state.search_clicked and st.session_state.user_location:
    user_lat, user_lon = st.session_state.user_location
    
    # Calculate actual distances from user location
    df['Actual_Distance'] = df.apply(
        lambda row: calculate_distance(user_lat, user_lon, row['Latitude'], row['Longitude']),
        axis=1
    )
    
    # Filter
    filtered = df[df['Actual_Distance'] <= max_dist].copy()
    filtered = filtered[filtered['Price'] <= budget]
    
    if veg_pref != "Any":
        filtered = filtered[filtered["Veg_NonVeg"] == veg_pref]
    if cuisine_pref != "Any":
        filtered = filtered[filtered["Cuisine"] == cuisine_pref]
    if search_query:
        filtered = filtered[filtered["Dish_Name"].str.contains(search_query, case=False, na=False)]
    
    if len(filtered) == 0:
        st.markdown("""
        <div class="no-results">
            <h2>😞 No meals found nearby!</h2>
            <p>Try increasing your distance or adjusting filters</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Scoring
        filtered["rating_score"] = filtered["Rating"] / 5.0
        filtered["price_score"] = 1 - (filtered["Price"] / budget)
        max_distance = filtered["Actual_Distance"].max()
        filtered["distance_score"] = 1 - (filtered["Actual_Distance"] / max_distance) if max_distance > 0 else 1
        filtered["final_score"] = (
            filtered["rating_score"] * 0.4 +
            filtered["price_score"] * 0.3 +
            filtered["distance_score"] * 0.3
        )
        
        # Sort
        if sort_by == "Nearest First":
            filtered = filtered.sort_values("Actual_Distance", ascending=True)
        elif sort_by == "Price (Low to High)":
            filtered = filtered.sort_values("Price", ascending=True)
        elif sort_by == "Rating (High to Low)":
            filtered = filtered.sort_values("Rating", ascending=False)
        else:
            filtered = filtered.sort_values("final_score", ascending=False)
        
        top_results = filtered.head(20).reset_index(drop=True)
        
        # Stats
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(filtered)}</div>
                <div class="stat-label">Total Matches</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{top_results['Actual_Distance'].min():.1f}km</div>
                <div class="stat-label">Nearest</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">Rs.{filtered['Price'].min()}</div>
                <div class="stat-label">Lowest Price</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">⭐{filtered['Rating'].max():.1f}</div>
                <div class="stat-label">Best Rating</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ Found {len(top_results)} meals near you!")
        
        # Display results
        for idx, row in top_results.iterrows():
            veg_icon = "🟢" if row['Veg_NonVeg'] == 'Veg' else "🔴"
            veg_class = "tag-veg" if row['Veg_NonVeg'] == 'Veg' else "tag-nonveg"
            
            # Google Maps directions URL
            directions_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={row['Latitude']},{row['Longitude']}&travelmode=driving"
            
            # WhatsApp share
            msg = f"🍽️ Found on BudgetBite!\n\n*{row['Dish_Name']}*\n🏪 {row['Restaurant_Name']}\n💰 Rs.{row['Price']}\n⭐ {row['Rating']}\n📍 {row['Actual_Distance']:.1f}km away"
            whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(msg)}"
            
            st.markdown(f"""
            <div class="pro-dish-card">
                <div class="dish-header">
                    <div>
                        <h2 class="dish-title">{veg_icon} {row['Dish_Name']}</h2>
                        <p class="restaurant-name">🏪 {row['Restaurant_Name']}</p>
                        <div class="tag-container">
                            <span class="dish-tag {veg_class}">{row['Veg_NonVeg']}</span>
                            <span class="dish-tag tag-cuisine">{row['Cuisine']}</span>
                        </div>
                    </div>
                    <div class="dish-rank-badge">{idx+1}</div>
                </div>
                
                <div class="dish-info-grid">
                    <div class="info-box">
                        <div class="info-value">Rs.{row['Price']}</div>
                        <div class="info-label">💰 Price</div>
                    </div>
                    <div class="info-box">
                        <div class="info-value">⭐{row['Rating']}</div>
                        <div class="info-label">🏆 Rating</div>
                    </div>
                    <div class="info-box">
                        <div class="info-value">{row['Actual_Distance']:.1f}km</div>
                        <div class="info-label">📍 Distance</div>
                    </div>
                    <div class="info-box">
                        <div class="info-value">{row['final_score']:.2f}</div>
                        <div class="info-label">🎯 Score</div>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <a href="{directions_url}" target="_blank" class="directions-btn">
                        🗺️ GET DIRECTIONS
                    </a>
                    <a href="{whatsapp_url}" target="_blank" class="whatsapp-btn">
                        📱 SHARE
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h2>👆 Enter your location and preferences</h2>
        <p style="font-size: 1.1rem; color: #6c757d;">Fill in the sidebar and click <strong>'FIND MEALS'</strong> to discover nearby restaurants!</p>
        <br>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Total Dishes</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{}</div>
                <div class="stat-label">Restaurants</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">⭐{:.1f}</div>
                <div class="stat-label">Avg Rating</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">Rs.{}</div>
                <div class="stat-label">Avg Price</div>
            </div>
        </div>
    </div>
    """.format(len(df), df['Restaurant_Name'].nunique(), df['Rating'].mean(), int(df['Price'].mean())), unsafe_allow_html=True)

# ===================== FOOTER =====================
st.markdown("""
<div style='text-align: center; padding: 40px; margin-top: 50px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 25px; color: white;'>
    <h2 style='margin: 0; font-size: 2rem; font-weight: 900;'>🍽️ BudgetBite Pro</h2>
    <p style='margin: 15px 0 0; font-size: 1.1rem; opacity: 0.95;'>AI-Powered Meal Discovery with GPS Navigation</p>
    <p style='margin: 10px 0 0; opacity: 0.85;'>TF-IDF ML | Real-time GPS | Smart Recommendations</p>
</div>
""", unsafe_allow_html=True)
