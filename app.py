import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse
import math

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="BudgetBite Pro - Auto GPS",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== SESSION STATE =====================
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'search_clicked' not in st.session_state:
    st.session_state.search_clicked = False

# ===================== CALCULATE DISTANCE =====================
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# ===================== CSS =====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    .modern-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 60px 40px;
        border-radius: 30px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
    }
    .header-title {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.4rem;
        margin-top: 15px;
    }
    .feature-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 10px 25px;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
    }
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
    }
    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 25px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
    }
    .location-success {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        text-align: center;
        font-weight: 700;
    }
    .gps-button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        padding: 15px 30px;
        border: none;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1rem;
        cursor: pointer;
        width: 100%;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-align: center;
        text-decoration: none;
        display: block;
        margin: 10px 0;
    }
    .gps-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ===================== GPS DETECTION HTML =====================
gps_html = """
<script>
function detectLocation() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                
                // Reload page with coordinates as URL parameters
                const url = new URL(window.location);
                url.searchParams.set('lat', lat.toFixed(6));
                url.searchParams.set('lon', lon.toFixed(6));
                window.location.href = url.toString();
            },
            function(error) {
                alert('Could not get location: ' + error.message + '. Please enable location services and try again.');
            }
        );
    } else {
        alert('Geolocation is not supported by your browser.');
    }
}
</script>

<button class="gps-button" onclick="detectLocation()">
    📍 AUTO-DETECT MY LOCATION
</button>
"""

# ===================== LOAD DATA =====================
@st.cache_data
def load_data():
    df = pd.read_csv("meal_data.csv")
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
    df_copy["combined_features"] = df_copy["Cuisine"] + " " + df_copy["Veg_NonVeg"] + " " + df_copy["Dish_Name"]
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_copy["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf_matrix, cosine_sim

tfidf_matrix, cosine_sim = build_ml_model(df)

# ===================== CHECK URL PARAMETERS =====================
query_params = st.query_params
auto_lat = query_params.get("lat", None)
auto_lon = query_params.get("lon", None)

# ===================== HEADER =====================
st.markdown("""
<div class="modern-header">
    <h1 class="header-title">🍽️ BudgetBite Pro</h1>
    <p class="header-subtitle">AI-Powered Meal Discovery with Auto GPS</p>
    <div style='margin-top: 25px;'>
        <span class="feature-badge">🤖 ML Powered</span>
        <span class="feature-badge">📍 Auto GPS</span>
        <span class="feature-badge">🗺️ Live Directions</span>
        <span class="feature-badge">💰 Budget First</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style='margin: 0; font-size: 1.8rem; font-weight: 900;'>🍽️ BudgetBite</h2>
        <p style='margin: 5px 0 0; font-size: 0.9rem;'>Smart Meal Discovery</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📍 Your Location")
    
    # Auto GPS Button
    st.markdown(gps_html, unsafe_allow_html=True)
    
    # Check if location was auto-detected
    if auto_lat and auto_lon:
        user_lat = float(auto_lat)
        user_lon = float(auto_lon)
        st.markdown(f"""
        <div class="location-success">
            ✅ Location Detected!<br>
            <strong>{user_lat:.4f}, {user_lon:.4f}</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Click the button above to auto-detect your location!")
        user_lat = st.number_input("Or enter Latitude manually", value=17.4400, format="%.4f", step=0.0001)
        user_lon = st.number_input("Or enter Longitude manually", value=78.4800, format="%.4f", step=0.0001)
    
    st.markdown("---")
    st.markdown("### 🔍 Search Preferences")
    
    with st.form("search_form"):
        budget = st.number_input("💰 Budget (Rs.)", 20, 500, 100, 10)
        veg_pref = st.selectbox("🥗 Diet", ["Any", "Veg", "Non-Veg"])
        cuisine_pref = st.selectbox("🍜 Cuisine", ["Any"] + sorted(df["Cuisine"].unique().tolist()))
        max_dist = st.number_input("📍 Max Distance (km)", 0.1, 10.0, 3.0, 0.1)
        search_query = st.text_input("🔎 Search Dish", placeholder="e.g. Biryani...")
        sort_by = st.selectbox("🔃 Sort By", ["Nearest First", "Best Match", "Price (Low to High)", "Rating (High to Low)"])
        
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
if st.session_state.search_clicked:
    
    df['Actual_Distance'] = df.apply(
        lambda row: calculate_distance(user_lat, user_lon, row['Latitude'], row['Longitude']), axis=1
    )
    
    filtered = df[df['Actual_Distance'] <= max_dist].copy()
    filtered = filtered[filtered['Price'] <= budget]
    
    if veg_pref != "Any":
        filtered = filtered[filtered["Veg_NonVeg"] == veg_pref]
    if cuisine_pref != "Any":
        filtered = filtered[filtered["Cuisine"] == cuisine_pref]
    if search_query:
        filtered = filtered[filtered["Dish_Name"].str.contains(search_query, case=False, na=False)]
    
    if len(filtered) == 0:
        st.error("😞 No meals found nearby! Try increasing your distance or adjusting filters.")
    else:
        filtered["rating_score"] = filtered["Rating"] / 5.0
        filtered["price_score"] = 1 - (filtered["Price"] / budget)
        max_distance = filtered["Actual_Distance"].max()
        filtered["distance_score"] = 1 - (filtered["Actual_Distance"] / max_distance) if max_distance > 0 else 1
        filtered["final_score"] = (
            filtered["rating_score"] * 0.4 + filtered["price_score"] * 0.3 + filtered["distance_score"] * 0.3
        )
        
        if sort_by == "Nearest First":
            filtered = filtered.sort_values("Actual_Distance", ascending=True)
        elif sort_by == "Price (Low to High)":
            filtered = filtered.sort_values("Price", ascending=True)
        elif sort_by == "Rating (High to Low)":
            filtered = filtered.sort_values("Rating", ascending=False)
        else:
            filtered = filtered.sort_values("final_score", ascending=False)
        
        top_results = filtered.head(20).reset_index(drop=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🍽️ Matches", len(filtered))
        with col2:
            st.metric("📍 Nearest", f"{top_results['Actual_Distance'].min():.1f}km")
        with col3:
            st.metric("💰 Cheapest", f"Rs.{filtered['Price'].min()}")
        with col4:
            st.metric("⭐ Best", f"{filtered['Rating'].max():.1f}")
        
        st.success(f"✅ Found {len(top_results)} meals near you!")
        st.markdown("---")
        
        for idx, row in top_results.iterrows():
            veg_icon = "🟢" if row['Veg_NonVeg'] == 'Veg' else "🔴"
            
            directions_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lon}&destination={row['Latitude']},{row['Longitude']}&travelmode=driving"
            msg = f"🍽️ {row['Dish_Name']}\n🏪 {row['Restaurant_Name']}\n💰 Rs.{row['Price']}\n⭐ {row['Rating']}\n📍 {row['Actual_Distance']:.1f}km"
            whatsapp_url = f"https://wa.me/?text={urllib.parse.quote(msg)}"
            
            with st.container():
                st.markdown(f"### {idx+1}. {veg_icon} {row['Dish_Name']}")
                st.caption(f"🏪 **{row['Restaurant_Name']}** | {row['Veg_NonVeg']} | {row['Cuisine']}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Price", f"Rs.{row['Price']}")
                with col2:
                    st.metric("⭐ Rating", f"{row['Rating']}")
                with col3:
                    st.metric("📍 Distance", f"{row['Actual_Distance']:.1f}km")
                with col4:
                    st.metric("🎯 Score", f"{row['final_score']:.2f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.link_button("🗺️ GET DIRECTIONS", directions_url, use_container_width=True)
                with col2:
                    st.link_button("📱 SHARE", whatsapp_url, use_container_width=True)
                st.markdown("---")

else:
    st.info("👆 Click **'AUTO-DETECT MY LOCATION'** in the sidebar, then click **'FIND MEALS'**!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🍽️ Dishes", len(df))
    with col2:
        st.metric("🏪 Restaurants", df['Restaurant_Name'].nunique())
    with col3:
        st.metric("⭐ Rating", f"{df['Rating'].mean():.1f}")
    with col4:
        st.metric("💰 Price", f"Rs.{int(df['Price'].mean())}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 25px; color: white;'>
    <h2 style='margin: 0; font-size: 2rem; font-weight: 900;'>🍽️ BudgetBite Pro</h2>
    <p style='margin: 15px 0 0;'>AI-Powered Meal Discovery with Auto GPS Navigation</p>
</div>
""", unsafe_allow_html=True)
