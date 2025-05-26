import streamlit as st
import numpy as np
from PIL import Image
#import pickle
import joblib

# === Charger le modèle entraîné ===
model = joblib.load("model.joblib")

#with open("model.pkl", "rb") as f:
#    model = pickle.load(f)

# === Liste des 37 features (booléennes) ===
features = [
    'sq_blanc', 'glob_Rouge', 'points_noirs', 'Erythème', 'sqjaunes', 'duvet',
    'anisotrichie', 'poils_en_points_exclamation', 'poils_en_virgule', 'tire_bouchon',
    'vx_arborisants', 'sx_périp', 'vx_glom', 'points_jaunes', 'cheveux_cassés',
    'absostiumf', 'poils_en_code_à_barre', 'pustules', 'IFOilyM', 'vx_torsadé_twistedredloops',
    'poils_en_cercle', 'cheveux_dystrophiques', 'gaine_coulissante', 'aire_blanche',
    'poils_en_V', 'zigzag', 'rx_pigmenté', 'coudés', 'sq_périfolliculaires', 'vx_linéaires',
    'halo_blanc_squameux', 'allumette', 'sq_blanc_jaunatres', 'poils_fourchus', 'hemorragie',
    'poils_en_touffe', 'poils_en_epingle_à_cheveux'
]

# === Interface Streamlit ===
st.title("🔬 Prédiction automatique basée sur 37 signes cliniques")

image_file = st.file_uploader("🖼️ Charger une image dermatologique", type=["jpg", "jpeg", "png"])

if image_file:
    img = Image.open(image_file)
    st.image(img, caption="Image affichée (faites clic-droit → Ouvrir dans un nouvel onglet pour zoomer)", use_column_width=True)

    st.markdown("---")



st.markdown("**Sélectionnez les signes observés :**")
cols = st.columns(3)  # Disposition en 3 colonnes
inputs = []

for i, feature in enumerate(features):
    checked = cols[i % 3].checkbox(feature)
    inputs.append(int(checked))

# === Prédiction ===
if st.button("🔍 Prédire la classe"):
    X_input = np.array([inputs])
    prediction = model.predict(X_input)[0]

    st.success(f"✅ **Classe prédite : {prediction}**")

    # Probabilités (si disponibles)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input)[0]
        class_names = model.classes_ if hasattr(model, "classes_") else [f"Classe {i}" for i in range(len(probs))]
        
        st.subheader("📊 Probabilités des classes")
        st.bar_chart(dict(zip(class_names, probs)))
