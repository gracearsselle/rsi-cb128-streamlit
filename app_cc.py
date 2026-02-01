import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="RSI-CB128 AI Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ULTRA MODERNE ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #666;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.5);
    }
    
    .metric-card h3 {
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .metric-card h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 40px;
        border-radius: 25px;
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 30px 0;
        box-shadow: 0 15px 50px rgba(245, 87, 108, 0.4);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #2196f3;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(33, 150, 243, 0.2);
    }
    
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
            
           
    .feature-card h3 {
        color: #1A1A1A !important;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .feature-card:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.3rem;
        padding: 20px 40px;
        border-radius: 50px;
        border: none;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .category-badge {
        display: inline-block;
        padding: 8px 15px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        margin: 5px;
        box-shadow: 0 3px 10px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==================== CHARGEMENT DES DONNEES ====================
@st.cache_resource
def load_model_and_data():
    try:
        model = tf.keras.models.load_model('rsi_cb128_cnn_model.keras')
        
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        with open('class_mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            
        # CORRECTION: Convertir les cles en int
        if isinstance(list(mappings['idx_to_class'].keys())[0], str):
            mappings['idx_to_class'] = {int(k): v for k, v in mappings['idx_to_class'].items()}
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
        
        return model, class_names, mappings, metadata, history, True
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None, None, None, None, None, False

def preprocess_image(image, img_size=128):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((img_size, img_size))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Chargement
model, class_names, mappings, metadata, history, MODEL_LOADED = load_model_and_data()

# ==================== HEADER ====================
st.markdown('<div class="main-header">🛰️ RSI-CB128 AI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Systeme Intelligent de Classification d\'Images Satellites par Deep Learning</div>', unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("---")
    st.markdown("## 🎯 Navigation")
    page = st.selectbox(
        "",
        ["🏠 Accueil", "🔮 Classification", "📈 Performances", "📚 Dataset", "ℹ️ A propos"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if MODEL_LOADED:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 15px; color: white; text-align: center;">
            <h3>✅ Systeme Operationnel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Statistiques")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classes", metadata['num_classes'], delta=None)
        with col2:
            st.metric("Accuracy", f"{metadata['best_val_accuracy']*100:.1f}%", delta=None)
    else:
        st.error("❌ Systeme Hors Ligne")
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Powered by TensorFlow & Streamlit</p>
        <p>© 2026 Deep Learning Project</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== PAGE: ACCUEIL ====================
if page == "🏠 Accueil":
    if not MODEL_LOADED:
        st.error("Le systeme n'a pas pu demarrer. Verifiez les fichiers.")
        st.stop()
    
    # Metriques principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Precision</h3>
            <h1>{metadata['best_val_accuracy']*100:.1f}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Categories</h3>
            <h1>{metadata['num_classes']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🧠 Parametres</h3>
            <h1>{metadata['total_params']/1e6:.1f}M</h1>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Sections principales
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 🎯 Capacites du Systeme")
        
        features = [
            ("🌍", "Classification de 45 types de terrains satellites"),
            ("⚡", "Analyse en temps reel avec IA avancee"),
            ("🎯", "Precision superieure a 85% validee"),
            ("📊", "Confiance de prediction detaillee"),
            ("🔄", "Support multi-formats (TIF, JPG, PNG)")
        ]
        
        for icon, text in features:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{icon} {text}</h3>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("## 📋 Categories Principales")
        
        categories = [
            ("🏗️", "Construction Land", "Batiments, tours, residences, conteneurs"),
            ("🌾", "Cultivated Land", "Terres agricoles, fermes, cultures"),
            ("🏔️", "Other Land", "Deserts, montagnes, plages, neige"),
            ("🚢", "Other Objects", "Infrastructures diverses"),
            ("🛣️", "Transportation", "Routes, ponts, aeroports, rails"),
            ("💧", "Water Area", "Rivieres, lacs, mers, barrages"),
            ("🌲", "Woodland", "Forets, vegetation, prairies")
        ]
        
        for icon, title, desc in categories:
            st.markdown(f"""
            <div class="feature-card">
                <h3>{icon} {title}</h3>
                <p style="color: #000000; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Utilisez le menu de navigation pour explorer les fonctionnalites")

# ==================== PAGE: CLASSIFICATION ====================
elif page == "🔮 Classification":
    if not MODEL_LOADED:
        st.error("Le systeme n'a pas pu demarrer.")
        st.stop()
    
    st.markdown("## 🔮 Classification d'Image Satellite")
    st.markdown("Chargez une image satellite pour obtenir une analyse instantanee par intelligence artificielle")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Glissez-deposez ou cliquez pour selectionner",
            type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
            help="Formats supportes: JPG, PNG, TIF"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Image: {uploaded_file.name}", use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 ANALYSER L'IMAGE", use_container_width=True):
                with st.spinner("🔄 Analyse en cours par le reseau neuronal..."):
                    img_array = preprocess_image(image, metadata['img_size'])
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class_idx] * 100
                    predicted_class = mappings['idx_to_class'][predicted_class_idx]
                    
                    st.session_state.prediction = predicted_class
                    st.session_state.confidence = confidence
                    st.session_state.all_predictions = predictions[0]
                    st.session_state.image_analyzed = True
    
    with col2:
        st.markdown("### 🎯 Resultats de l'Analyse")
        
        if hasattr(st.session_state, 'image_analyzed') and st.session_state.image_analyzed:
            # Prediction principale
            st.markdown(f"""
            <div class="prediction-box">
                🏆 {st.session_state.prediction.replace('_', ' ').upper()}
            </div>
            """, unsafe_allow_html=True)
            
            # Jauge de confiance avec plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=st.session_state.confidence,
                title={'text': "Niveau de Confiance", 'font': {'size': 24, 'family': 'Poppins'}},
                delta={'reference': 80, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue", 'thickness': 0.75},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffcdd2'},
                        {'range': [50, 80], 'color': '#fff9c4'},
                        {'range': [80, 100], 'color': '#c8e6c9'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=350,
                font={'family': 'Poppins', 'size': 16},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 5 predictions
            st.markdown("### 📊 Top 5 Predictions")
            top5_idx = np.argsort(st.session_state.all_predictions)[-5:][::-1]
            
            for rank, idx in enumerate(top5_idx, 1):
                class_name = mappings['idx_to_class'][idx].replace('_', ' ').title()
                prob = st.session_state.all_predictions[idx] * 100
                
                col_a, col_b, col_c = st.columns([0.5, 3, 1])
                with col_a:
                    st.markdown(f"**#{rank}**")
                with col_b:
                    st.progress(float(prob/100))
                with col_c:
                    st.markdown(f"**{prob:.1f}%**")
                st.markdown(f"<p style='margin-top: -10px; color: #666;'>{class_name}</p>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h3>💡 Instructions</h3>
                <p>1. Chargez une image satellite dans la zone de gauche</p>
                <p>2. Cliquez sur le bouton ANALYSER</p>
                <p>3. Obtenez les resultats instantanement</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== PAGE: PERFORMANCES ====================
elif page == "📈 Performances":
    if not MODEL_LOADED:
        st.error("Le systeme n'a pas pu demarrer.")
        st.stop()
    
    st.markdown("## 📈 Performances du Modele")
    
    # Metriques
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Validation", f"{metadata['best_val_accuracy']*100:.2f}%", delta=f"+{(metadata['best_val_accuracy']-0.8)*100:.1f}%")
    col2.metric("📉 Loss", f"{metadata['test_loss']:.4f}", delta=f"-{metadata['test_loss']*10:.2f}")
    col3.metric("✅ Test", f"{metadata['test_accuracy']*100:.2f}%", delta="Excellent")
    col4.metric("🔄 Epochs", metadata['epochs_trained'], delta="Complet")
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=history['accuracy'],
            name='Entrainement',
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig_acc.add_trace(go.Scatter(
            y=history['val_accuracy'],
            name='Validation',
            mode='lines+markers',
            line=dict(color='#f5576c', width=3),
            marker=dict(size=8)
        ))
        fig_acc.update_layout(
            title="Evolution de la Precision",
            xaxis_title="Epoch",
            yaxis_title="Accuracy",
            hovermode='x unified',
            height=450,
            font={'family': 'Poppins', 'size': 14},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=history['loss'],
            name='Entrainement',
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ))
        fig_loss.add_trace(go.Scatter(
            y=history['val_loss'],
            name='Validation',
            mode='lines+markers',
            line=dict(color='#f5576c', width=3),
            marker=dict(size=8)
        ))
        fig_loss.update_layout(
            title="Evolution de la Loss",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode='x unified',
            height=450,
            font={'family': 'Poppins', 'size': 14},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Tableau detaille
    st.markdown("### 📋 Historique Detaille par Epoch")
    df_history = pd.DataFrame({
        'Epoch': range(1, len(history['accuracy']) + 1),
        'Train Acc': [f"{x*100:.2f}%" for x in history['accuracy']],
        'Val Acc': [f"{x*100:.2f}%" for x in history['val_accuracy']],
        'Train Loss': [f"{x:.4f}" for x in history['loss']],
        'Val Loss': [f"{x:.4f}" for x in history['val_loss']]
    })
    st.dataframe(df_history, use_container_width=True, height=400)

# ==================== PAGE: DATASET ====================
elif page == "📚 Dataset":
    st.markdown("## 📚 Dataset RSI-CB128")
    
    st.markdown("""
    Le **RSI-CB128** (Remote Sensing Image Classification Benchmark) est un dataset de reference 
    pour la classification d'images satellites et aeriennes haute resolution.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🖼️ Total Images", "36,707")
    col2.metric("🏷️ Categories", "45")
    col3.metric("📐 Resolution", "128x128 px")
    
    st.markdown("---")
    
    st.markdown("### 🗂️ Les 7 Categories Principales")
    
    categories_detail = {
        "🏗️ Construction Land": {
            "count": 6,
            "classes": ["city_building", "container", "grave", "residents", "storage_room", "tower"]
        },
        "🌾 Cultivated Land": {
            "count": 3,
            "classes": ["bare_land", "dry_farm", "green_farmland"]
        },
        "🏔️ Other Land": {
            "count": 4,
            "classes": ["desert", "mountain", "sandbeach", "snow_mountain"]
        },
        "🚢 Other Objects": {
            "count": 2,
            "classes": ["pipeline", "town"]
        },
        "🛣️ Transportation": {
            "count": 13,
            "classes": ["airport_runway", "avenue", "bridge", "city_road", "crossroads", "fork_road", 
                       "highway", "marina", "mountain_road", "overpass", "parkinglot", "rail", "turning_circle"]
        },
        "💧 Water Area": {
            "count": 7,
            "classes": ["coastline", "dam", "hirst", "lakeshore", "river", "sea", "stream"]
        },
        "🌲 Woodland": {
            "count": 10,
            "classes": ["artificial_grassland", "city_avenue", "city_green_tree", "forest", "mangrove", 
                       "natural_grassland", "river_protection_forest", "sapling", "shrubwood", "sparse_forest"]
        }
    }
    
    for cat_name, cat_data in categories_detail.items():
        with st.expander(f"{cat_name} ({cat_data['count']} sous-categories)", expanded=False):
            cols = st.columns(3)
            for i, cls in enumerate(cat_data['classes']):
                with cols[i % 3]:
                    st.markdown(f"""
                    <span class="category-badge">{cls.replace('_', ' ').title()}</span>
                    """, unsafe_allow_html=True)

# ==================== PAGE: A PROPOS ====================
elif page == "ℹ️ A propos":
    if not MODEL_LOADED:
        st.error("Le systeme n'a pas pu demarrer.")
        st.stop()
    
    st.markdown("## ℹ️ A propos du Systeme")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### 🎓 Informations du Projet
        
        **Projet**: Controle Continu Deep Learning - CNN  
        **Dataset**: RSI-CB128 (Remote Sensing Image Classification)  
        **Modele**: Convolutional Neural Network Personnalise  
        **Framework**: TensorFlow 2.x / Keras  
        **Interface**: Streamlit 1.x  
        
        ### 🏗️ Architecture du Reseau
        
        - **Date de creation**: {metadata['date_created']}
        - **Type**: CNN avec 4 blocs convolutionnels
        - **Parametres entrainables**: {metadata['total_params']:,}
        - **Taille d'entree**: {metadata['img_size']} x {metadata['img_size']} x 3 (RGB)
        - **Nombre de classes**: {metadata['num_classes']}
        - **Fonction d'activation**: ReLU (couches cachees), Softmax (sortie)
        - **Optimizer**: Adam (lr=0.001)
        - **Fonction de perte**: Sparse Categorical Crossentropy
        - **Regularisation**: Dropout (0.5, 0.3), BatchNormalization
        
        ### 🎯 Resultats des Performances
        
        - **Accuracy sur Test Set**: {metadata['test_accuracy']*100:.2f}%
        - **Meilleure Accuracy Validation**: {metadata['best_val_accuracy']*100:.2f}%
        - **Loss sur Test Set**: {metadata['test_loss']:.4f}
        - **Nombre d'epochs**: {metadata['epochs_trained']}
        
        ### 🚀 Fonctionnalites Principales
        
        1. Classification automatique d'images satellites
        2. Analyse en temps reel avec predictions instantanees
        3. Niveau de confiance pour chaque prediction
        4. Top 5 des predictions les plus probables
        5. Visualisation interactive des performances
        6. Historique detaille de l'entrainement
        7. Interface moderne et responsive
        
        ### 👨‍💻 Stack Technologique
        
        - **Python**: 3.10+
        - **TensorFlow/Keras**: Deep Learning
        - **Streamlit**: Interface Web
        - **Plotly**: Visualisations interactives
        - **NumPy**: Calcul numerique
        - **Pandas**: Manipulation de donnees
        - **PIL**: Traitement d'images
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 20px; color: white; text-align: center;">
            <h2>📊 Stats Globales</h2>
            <hr style="border-color: white;">
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Precision Globale", f"{metadata['test_accuracy']*100:.1f}%")
        st.metric("Classes Supportees", metadata['num_classes'])
        st.metric("Images Traitees", "36,707")
        st.metric("Parametres Modele", f"{metadata['total_params']/1e6:.2f}M")
        
        st.markdown("---")
        
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <p style="font-size: 1.2rem; color: #667eea; font-weight: 600;">
                Developpe avec ❤️
            </p>
            <p style="color: #666;">
                Pour le cours de Deep Learning<br>
                Master 2 - 2026
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.balloons()