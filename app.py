import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Perilaku Belanja",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
/* App background */
.stApp {
    background: radial-gradient(circle at top left, #020617 0, #000 55%);
    color: #e5e7eb;
}

/* Main container padding */
.block-container {
    padding-top: 1.2rem;
    max-width: 1300px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: radial-gradient(circle at top left, #020617, #020617 60%);
    border-right: 1px solid rgba(34,211,238,0.35);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Neon card */
.neon-card {
    background: radial-gradient(circle at top left, #020617 0%, #020617 70%);
    border-radius: 24px;
    padding: 18px 22px;
    border: 1px solid rgba(34,211,238,0.55);
    box-shadow: 0 0 25px rgba(34,211,238,0.28);
    margin-bottom: 22px;
    animation: fadeIn 0.6s ease;
}

/* Tabs container – beri jarak */
div[data-baseweb="tab-list"] {
    display: flex;
    gap: 0.6rem !important;        /* jarak antar tab */
    padding: 0.4rem 0.3rem !important;
    margin-bottom: 0.8rem !important;
    flex-wrap: wrap;               /* jika tab banyak, tetap rapi */
}

/* Tab button – rounded, neon, empuk */
button[data-baseweb="tab"] {
    background: rgba(30,41,59,0.6) !important;
    border-radius: 14px !important;
    padding: 8px 18px !important;
    font-size: 0.92rem !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    color: #e2e8f0 !important;
    transition: all 0.20s ease-in-out !important;
    backdrop-filter: blur(6px);
}

/* Hover tabs */
button[data-baseweb="tab"]:hover {
    border: 1px solid rgba(56,189,248,0.75) !important;
    box-shadow: 0 0 12px rgba(56,189,248,0.4);
    transform: translateY(-1px);
}

/* Active tab */
button[data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(34,211,238,0.35), rgba(14,165,233,0.25)) !important;
    border: 1px solid rgba(34,211,238,0.9) !important;
    box-shadow: 0 0 18px rgba(34,211,238,0.45) !important;
    color: #e0faff !important;
    font-weight: 600 !important;
}

/* Title */
h1 {
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    color: #e0f2fe !important;
}
h2, h3 {
    color: #e0f2fe !important;
}

/* Horizontal line */
hr {
    border: none;
    border-top: 1px solid rgba(148,163,184,0.35);
    margin: 0.4rem 0 1rem 0;
}

/* Info box tweaks */
div[data-baseweb="notification"] {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(56,189,248,0.4) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #020617;
}
::-webkit-scrollbar-thumb {
    background: #1f2937;
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: #4b5563;
}

/* Fade-in animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(4px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# --- Helper: Dark style untuk Matplotlib/Seaborn axes ---
def style_dark_ax(ax):
    """Menerapkan style gelap ke axis Matplotlib."""
    fig = ax.get_figure()
    fig.patch.set_facecolor("#020617")
    ax.set_facecolor("#020617")

    # Spines
    for spine in ax.spines.values():
        spine.set_color("#4b5563")

    # Ticks
    ax.tick_params(colors="#e5e7eb")

    # Labels & title
    ax.title.set_color("#e5e7eb")
    ax.xaxis.label.set_color("#e5e7eb")
    ax.yaxis.label.set_color("#e5e7eb")

    # Tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color("#e5e7eb")

    # Legend
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#020617")
        legend.get_frame().set_edgecolor("#4b5563")
        for text in legend.get_texts():
            text.set_color("#e5e7eb")


# Global seaborn theme
sns.set_theme(style="darkgrid")


# --- Pemuatan dan Preprocessing Data ---
@st.cache_data
def load_data():
    """Memuat data dan melakukan preprocessing awal (termasuk membuat kolom kelompok usia)."""
    file_path = 'shopping_behavior_updated.csv'
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan. Pastikan file CSV berada di direktori yang sama dengan aplikasi Streamlit.")
        return None

    df = pd.read_csv(file_path)

    # Preprocessing: Membuat kolom 'Age Group' 10 tahunan (Plot 1)
    max_age = df['Age'].max()
    df['Age Group'] = pd.cut(
        df['Age'],
        bins=range(0, max_age + 11, 10),
        right=False,
        labels=[f'{i}-{i+9}' for i in range(0, max_age + 1, 10) if i <= max_age]
    ).astype(str)
    
    # Preprocessing: Membuat kolom 'Age Group' rentang khusus (Plot 6)
    bins_bar = [18, 25, 35, 45, 55, 65, 100]
    labels_bar = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['Age Group_Bar'] = pd.cut(df['Age'], bins=bins_bar, labels=labels_bar, right=False)

    return df


df = load_data()

# --- Mapping Dictionaries untuk Plot Peta ---
US_STATE_ABBR = {
    "Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA",
    "Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA",
    "Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA",
    "Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD",
    "Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO",
    "Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ",
    "New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH",
    "Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC",
    "South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT",
    "Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY",
    "District of Columbia":"DC"
}

STATE_CENTROIDS = {
    'AL': (32.806671, -86.791130), 'AK': (61.370716, -152.404419), 'AZ': (33.729759, -111.431221),
    'AR': (34.969704, -92.373123), 'CA': (36.116203, -119.681564), 'CO': (39.059811, -105.311104),
    'CT': (41.597782, -72.755371), 'DE': (39.318523, -75.507141), 'FL': (27.766279, -81.686783),
    'GA': (33.040619, -83.643074), 'HI': (21.094318, -157.498337), 'ID': (44.240459, -114.478828),
    'IL': (40.349457, -88.986137), 'IN': (39.849426, -86.258278), 'IA': (42.011539, -93.210526),
    'KS': (38.526600, -96.726486), 'KY': (37.668140, -84.670067), 'LA': (31.169546, -91.867805),
    'ME': (44.693947, -69.381927), 'MD': (39.063946, -76.802101), 'MA': (42.230171, -71.530106),
    'MI': (43.326618, -84.536095), 'MN': (45.694454, -93.900192), 'MS': (32.741646, -89.678696),
    'MO': (38.456085, -92.288368), 'MT': (46.921925, -110.454353), 'NE': (41.125370, -98.268082),
    'NV': (38.313515, -117.055374), 'NH': (43.452492, -71.563896), 'NJ': (40.298904, -74.521011),
    'NM': (34.840515, -106.248482), 'NY': (42.165726, -74.948051), 'NC': (35.630066, -79.806419),
    'ND': (47.528912, -99.784012), 'OH': (40.388783, -82.764915), 'OK': (35.565342, -96.928917),
    'OR': (44.572021, -122.070938), 'PA': (40.590752, -77.209755), 'RI': (41.680893, -71.511780),
    'SC': (33.856892, -80.945007), 'SD': (44.299782, -99.438828), 'TN': (35.747845, -86.692345),
    'TX': (31.054487, -97.563461), 'UT': (40.150032, -111.862434), 'VT': (44.045876, -72.710686),
    'VA': (37.769337, -78.169968), 'WA': (47.400902, -121.490494), 'WV': (38.491226, -80.954453),
    'WI': (44.268543, -89.616508), 'WY': (42.755966, -107.302490), 'DC': (38.9072, -77.0369)
}


# --- FUNGSI KESIMPULAN OTOMATIS ---
def generate_conclusion_age(filtered_df):
    if filtered_df.empty or 'Age Group' not in filtered_df.columns:
        return "Tidak ada data usia yang cukup untuk dianalisis."
    
    counts = filtered_df['Age Group'].value_counts()
    top_group = counts.idxmax()
    count = counts.max()
    total = len(filtered_df)
    percent = (count / total) * 100

    return (
        f"Kelompok usia **{top_group}** mendominasi transaksi dengan "
        f"**{count} pembelian ({percent:.1f}%)**, menunjukkan bahwa rentang usia tersebut "
        f"merupakan segmen pelanggan paling aktif pada filter saat ini."
    )


def generate_conclusion_location(filtered_df):
    if filtered_df.empty or 'Location' not in filtered_df.columns:
        return "Tidak ada data lokasi untuk dianalisis."

    counts = filtered_df['Location'].value_counts()
    top_loc = counts.idxmax()
    count = counts.max()

    return (
        f"Lokasi dengan transaksi terbanyak adalah **{top_loc}** dengan "
        f"**{count} transaksi**, sehingga wilayah ini dapat diprioritaskan "
        f"sebagai target utama aktivitas pemasaran."
    )


def generate_conclusion_map(filtered_df):
    if filtered_df.empty or 'Location' not in filtered_df.columns:
        return "Data lokasi tidak cukup untuk dianalisis di peta."

    top_states = filtered_df['Location'].value_counts().head(3)
    txt = ", ".join([f"{st} ({ct})" for st, ct in top_states.items()])

    return (
        f"Tiga state dengan transaksi terbanyak adalah: **{txt}**. "
        f"Hal ini menunjukkan konsentrasi aktivitas belanja yang kuat pada wilayah-wilayah tersebut."
    )


def generate_conclusion_payment(filtered_df):
    if filtered_df.empty or 'Payment Method' not in filtered_df.columns:
        return "Tidak ada data metode pembayaran untuk dianalisis."

    counts = filtered_df['Payment Method'].value_counts()
    top_pay = counts.idxmax()
    count = counts.max()
    total = len(filtered_df)
    percent = (count / total) * 100

    return (
        f"Metode pembayaran yang paling banyak digunakan adalah **{top_pay}** dengan "
        f"**{count} transaksi ({percent:.1f}%)**, menunjukkan preferensi pelanggan yang kuat "
        f"terhadap metode pembayaran tersebut."
    )


def generate_conclusion_heatmap(filtered_df):
    if filtered_df.empty:
        return "Tidak ada data yang cukup untuk membuat analisis musiman."

    pivot = filtered_df.pivot_table(
        index='Category',
        columns='Season',
        values='Purchase Amount (USD)',
        aggfunc='mean'
    )

    if pivot.empty:
        return "Data tidak cukup untuk membentuk pola musiman antar kategori."

    max_cat = pivot.max(axis=1).idxmax()
    max_season = pivot.loc[max_cat].idxmax()
    max_value = pivot.loc[max_cat].max()

    return (
        f"Kategori **{max_cat}** memiliki rata-rata pengeluaran tertinggi pada musim **{max_season}** "
        f"yakni sekitar **${max_value:.2f}**, mengindikasikan adanya pola musiman yang kuat "
        f"untuk kategori tersebut."
    )


def generate_conclusion_age_product(filtered_df):
    if filtered_df.empty or 'Age Group_Bar' not in filtered_df.columns:
        return "Tidak ada data yang cukup untuk analisis produk per kelompok usia."

    group = filtered_df.groupby(['Age Group_Bar', 'Category']).size()
    top = group.idxmax()
    count = group.max()
    age, category = top

    return (
        f"Kelompok usia **{age}** paling banyak membeli kategori **{category}** "
        f"dengan **{count} transaksi**, menunjukkan preferensi produk yang cukup jelas "
        f"berdasarkan segmen umur."
    )


# --- MAIN DASHBOARD ---
if df is not None:
    # Header
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    st.title("📊 Analisis Perilaku Belanja Pelanggan 📊")
   

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar: Filter Global & Spesifik
    st.sidebar.header("📍 Filter Global")

    all_gender = df['Gender'].unique()
    all_category = df['Category'].unique()
    all_season = df['Season'].unique()

    selected_gender = st.sidebar.multiselect("Pilih Gender:", all_gender, default=all_gender)
    selected_category = st.sidebar.multiselect("Pilih Kategori Produk:", all_category, default=all_category)
    selected_season = st.sidebar.multiselect("Pilih Musim (Season):", all_season, default=all_season)

    st.sidebar.markdown("---")
    st.sidebar.header("📍 Filter Spesifik")

    min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
    age_range = st.sidebar.slider("Rentang Usia:", min_age, max_age, (min_age, max_age), step=1)

    all_location = df['Location'].unique()
    all_payment = df['Payment Method'].unique()

    selected_location = st.sidebar.multiselect("Lokasi:", all_location, default=all_location)
    selected_payment = st.sidebar.multiselect("Metode Pembayaran:", all_payment, default=all_payment)

    # Terapkan semua filter
    filtered_df = df[
        (df['Gender'].isin(selected_gender)) &
        (df['Category'].isin(selected_category)) &
        (df['Season'].isin(selected_season)) &
        (df['Location'].isin(selected_location)) &
        (df['Payment Method'].isin(selected_payment)) &
        (df['Age'].between(age_range[0], age_range[1]))
    ]

    st.sidebar.markdown("---")
    st.sidebar.info(f"Menampilkan {len(filtered_df)} dari {len(df)} transaksi.")

    # KPI + Insight Cepat
    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Transaksi Terfilter", len(filtered_df))
    with col_b:
        st.metric("Jumlah Kategori Aktif", filtered_df['Category'].nunique() if not filtered_df.empty else 0)
    with col_c:
        st.metric("Jumlah Lokasi Aktif", filtered_df['Location'].nunique() if not filtered_df.empty else 0)

    if not filtered_df.empty and 'Age Group' in filtered_df.columns:
        ag_counts = filtered_df['Age Group'].value_counts(normalize=True) * 100
        top_age = ag_counts.idxmax()
        top_age_pct = ag_counts.max()

        loc_counts = filtered_df['Location'].value_counts()
        top_loc = loc_counts.idxmax()
        top_loc_val = loc_counts.max()

        cat_counts = filtered_df['Category'].value_counts()
        top_cat = cat_counts.idxmax()
        top_cat_val = cat_counts.max()

        st.markdown(
            f"**Insight Cepat:** Kelompok usia yang paling dominan saat ini adalah **{top_age}** "
            f"dengan sekitar **{top_age_pct:.1f}%** dari transaksi terfilter. "
            f"Lokasi dengan transaksi terbanyak adalah **{top_loc}** (**{top_loc_val} transaksi**), "
            f"dan kategori produk yang paling sering dibeli adalah **{top_cat}** "
            f"(**{top_cat_val} transaksi**)."
        )
    else:
        st.markdown("_Tidak ada data yang cukup setelah filter diterapkan untuk membentuk insight cepat._")
    st.markdown("</div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "👦🏼 Distribusi Usia",
        "🌎 Lokasi per Kategori",
        "🗺️ Peta USA",
        "💳 Metode Pembayaran",
        "🔎 Heatmap Musim",
        "🛍️ Produk per Usia"
    ])

    # Tab 1: Distribusi Usia
    with tab1:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Distribusi Usia Pelanggan (Rentang 10 Tahun)")
        
        if 'Age Group' in filtered_df.columns and not filtered_df.empty:
            age_group_counts = filtered_df['Age Group'].value_counts().sort_index()
            top_age_groups = age_group_counts.sort_values(ascending=False).head(3)

            fig_age = go.Figure()

            fig_age.add_bar(
                x=age_group_counts.index,
                y=age_group_counts.values,
                marker=dict(
                    color=age_group_counts.values,
                    colorscale="Teal",
                ),
                name="Jumlah Pelanggan"
            )

            fig_age.add_trace(
                go.Scatter(
                    x=list(age_group_counts.index),
                    y=age_group_counts.values,
                    mode="lines+markers",
                    line=dict(color="#22d3ee", width=2),
                    name="Tren Usia"
                )
            )

            for rank, (age_group, count) in enumerate(top_age_groups.items(), start=1):
                fig_age.add_annotation(
                    x=age_group,
                    y=count,
                    text=f"Top {rank}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    bgcolor="rgba(15,23,42,0.9)",
                    font=dict(color="#e5e7eb", size=12)
                )

            fig_age.update_layout(
                title_text="Distribusi Usia Pelanggan (Rentang 10 Tahun)",
                xaxis_title="Rentang Usia Pelanggan",
                yaxis_title="Jumlah Pelanggan",
                bargap=0.15,
                font=dict(size=13, color="#e5e7eb"),
                title_x=0.5,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_age, use_container_width=True)

            st.markdown("###Kesimpulan ")
            st.info(generate_conclusion_age(filtered_df))
        else:
            st.write("Tidak ada data untuk visualisasi ini.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Lokasi per Kategori
    with tab2:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Lokasi dengan Pembelian Terbanyak per Kategori Produk")

    grouped = filtered_df.groupby(['Category', 'Location']).size().reset_index(name='Count')

    if not grouped.empty:
        grouped_sorted = grouped.sort_values(['Category', 'Count'], ascending=[True, False])
        top_locations = grouped_sorted.groupby('Category').head(5)   # sudah jadi top 5

        # Plotly bar chart (INTERAKTIF)
        fig = px.bar(
            top_locations,
            x="Count",
            y="Category",
            color="Location",
            barmode="group",
            hover_data=["Location", "Count"],
            title="Top 5 Lokasi dengan Pembelian Terbanyak per Kategori Produk",
            height=400 + (50 * top_locations['Category'].nunique())
        )

        fig.update_layout(
            legend_title_text="Lokasi",
            title_font_size=16,
            xaxis_title="Jumlah Pembelian",
            yaxis_title="Kategori Produk",
            plot_bgcolor='#0f172a',
            paper_bgcolor='#0f172a',
            font_color='white'
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Kesimpulan")
        st.info(generate_conclusion_location(filtered_df))

    else:
        st.write("Tidak ada data untuk kategori yang difilter.")

    st.markdown("</div>", unsafe_allow_html=True)


    # Tab 3: Peta USA
    with tab3:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Jumlah Transaksi per Lokasi (Peta USA)")
        
        location_counts = filtered_df.groupby('Location').size().reset_index(name='Count')
        location_counts['state_code'] = location_counts['Location'].map(US_STATE_ABBR)
        location_counts = location_counts.dropna(subset=['state_code'])
        
        if not location_counts.empty:
            fig_map = px.choropleth(
                location_counts,
                locations='state_code',
                locationmode='USA-states',
                scope='usa',
                color='Count',
                color_continuous_scale='Tealgrn',
                title='Jumlah Transaksi per Lokasi (Peta USA)',
                template='plotly_dark'
            )

            top3 = location_counts.sort_values('Count', ascending=False).head(3).reset_index(drop=True)
            top3['lat'] = top3['state_code'].map(lambda c: STATE_CENTROIDS.get(c, (None, None))[0])
            top3['lon'] = top3['state_code'].map(lambda c: STATE_CENTROIDS.get(c, (None, None))[1])
            rank_text = [f"Top {i+1}" for i in range(len(top3))]

            fig_map.add_trace(go.Scattergeo(
                lat=top3['lat'],
                lon=top3['lon'],
                mode='markers+text',
                text=rank_text,
                textposition='top center',
                marker=dict(size=10, line=dict(width=1, color="#22d3ee"), color="#22d3ee"),
                hovertemplate="State: %{customdata[0]}<br>Jumlah: %{customdata[1]}<extra></extra>",
                customdata=top3[['Location', 'Count']].values,
                showlegend=False
            ))

            fig_map.update_layout(
                margin=dict(l=10, r=10, t=60, b=10),
                title_x=0.5,
                paper_bgcolor="rgba(0,0,0,0)",
                geo_bgcolor="#020617",
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.markdown("### Kesimpulan")
            st.info(generate_conclusion_map(filtered_df))
        else:
            st.write("Tidak ada data lokasi yang valid untuk ditampilkan di peta.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Metode Pembayaran (Donut)
    with tab4:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Penggunaan Metode Pembayaran")

        if not filtered_df.empty:
            payment_counts = filtered_df['Payment Method'].value_counts().reset_index()
            payment_counts.columns = ['Payment Method', 'Count']

            fig_pay = px.pie(
                payment_counts,
                names='Payment Method',
                values='Count',
                hole=0.55,
                template='plotly_dark',
            )

            fig_pay.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )

            fig_pay.update_layout(
                title_text="Proporsi Penggunaan Metode Pembayaran",
                title_x=0.5,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_pay, use_container_width=True)

            st.markdown("### Kesimpulan")
            st.info(generate_conclusion_payment(filtered_df))
        else:
            st.write("Tidak ada data untuk metode pembayaran yang difilter.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Heatmap Musim
    with tab5:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Heatmap Rata-rata Jumlah Pembelian berdasarkan Kategori dan Musim")
        
        pivot_data = filtered_df.pivot_table(
            index='Category',
            columns='Season',
            values='Purchase Amount (USD)',
            aggfunc='mean'
        )
        
        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".2f",
                cmap='viridis',
                ax=ax
            )

            ax.set_title('Rata-rata Jumlah Pembelian ($) per Kategori dan Musim', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Musim (Season)', fontsize=12)
            ax.set_ylabel('Kategori (Category)', fontsize=12)

            style_dark_ax(ax)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### Kesimpulan")
            st.info(generate_conclusion_heatmap(filtered_df))
        else:
            st.write("Tidak ada data untuk membuat Heatmap.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 6: Produk per Usia
    with tab6:
        st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
        st.subheader("Produk Paling Laris per Kelompok Umur")

        if 'Age Group_Bar' in filtered_df.columns and not filtered_df.empty:
            product_sales_by_age = filtered_df.groupby(['Age Group_Bar', 'Category']).size().reset_index(name='Count')

            pivot_stack = product_sales_by_age.pivot(
                index='Age Group_Bar',
                columns='Category',
                values='Count'
            ).fillna(0)

            pivot_stack = pivot_stack.sort_index()

            fig_stack = go.Figure()
            for cat in pivot_stack.columns:
                fig_stack.add_bar(
                    x=pivot_stack.index.astype(str),
                    y=pivot_stack[cat],
                    name=str(cat)
                )

            fig_stack.update_layout(
                barmode='stack',
                title_text="Produk Paling Laris per Kelompok Umur (Stacked)",
                xaxis_title="Kelompok Umur",
                yaxis_title="Jumlah Pembelian",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#020617",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            st.plotly_chart(fig_stack, use_container_width=True)

            st.markdown("### Kesimpulan")
            st.info(generate_conclusion_age_product(filtered_df))
        else:
            st.write("Tidak ada data untuk Produk Paling Laris per Kelompok Umur.")
        st.markdown("</div>", unsafe_allow_html=True)
