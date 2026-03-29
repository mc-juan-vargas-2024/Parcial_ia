import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Configuración general de la página ──────────────────────────────────────
st.set_page_config(
    page_title="Dog Breed Detector",
    page_icon="🐾",
    layout="centered"
)

# ── Hoja de estilos personalizada ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

.stApp {
    background-color: #0d0d0d;
}

/* Título principal */
h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem !important;
    letter-spacing: 6px;
    color: #f0f0f0;
    margin-bottom: 0 !important;
    line-height: 1;
}

/* Subtítulo descriptivo */
.subtitle {
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Área de carga de imagen */
.upload-box {
    border: 1px dashed #2a2a2a;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    background: #111;
    margin-bottom: 1.5rem;
}

/* Tarjeta de resultados */
.result-box {
    background: #111;
    border: 1px solid #222;
    border-left: 4px solid #e5ff00;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
}

/* Etiqueta "Raza identificada" */
.result-label {
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #e5ff00;
    margin-bottom: 0.5rem;
}

/* Nombre de la raza */
.result-breed {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 3px;
    color: #f0f0f0;
    line-height: 1;
}

/* Texto de confianza */
.result-confidence {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.5rem;
}

/* Fondo de la barra de progreso */
.confidence-bar-bg {
    background: #1e1e1e;
    border-radius: 999px;
    height: 5px;
    margin-top: 0.9rem;
    overflow: hidden;
}

/* Relleno de la barra de progreso */
.confidence-bar-fill {
    background: linear-gradient(90deg, #e5ff00, #b8cc00);
    height: 5px;
    border-radius: 999px;
}

/* Sección de predicciones secundarias */
.top-preds {
    margin-top: 1.2rem;
    border-top: 1px solid #1e1e1e;
    padding-top: 1rem;
}

/* Fila de cada predicción secundaria */
.top-pred-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #444;
    padding: 0.2rem 0;
}

/* Botón principal */
.stButton > button {
    background: #e5ff00;
    color: #0d0d0d;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    width: 100%;
    font-size: 0.85rem;
}

.stButton > button:hover {
    background: #f0f0f0;
    color: #0d0d0d;
}

hr {
    border-color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# ── Encabezado de la aplicación ──────────────────────────────────────────────
st.markdown("<h1>DOG BREED</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color:#e5ff00;margin-top:-0.6rem!important;'>DETECTOR</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Stanford Dogs · 120 Breeds · CNN</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Carga del modelo entrenado ───────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    ruta_modelo = os.path.join(os.getcwd(), "mejor_modelo.keras")

    if not os.path.exists(ruta_modelo):
        st.error(f"❌ Modelo no encontrado en: {ruta_modelo}")
        return None

    import builtins
    import tensorflow as tf

    # Necesario para que la capa Lambda pueda resolver 'tf' al deserializar
    builtins.tf = tf

    return tf.keras.models.load_model(
        ruta_modelo,
        custom_objects={"tf": tf},
        safe_mode=False
    )

modelo = cargar_modelo()

# ── Lista de las 120 razas del dataset Stanford Dogs ─────────────────────────
clases = [
    'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih', 'Blenheim_spaniel',
    'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
    'bloodhound', 'bluetick', 'black', 'Walker_hound', 'English_foxhound', 'redbone',
    'borzoi', 'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound',
    'Norwegian_elkhound', 'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
    'Staffordshire_bullterrier', 'American_Staffordshire_terrier', 'Bedlington_terrier',
    'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier', 'Norfolk_terrier',
    'Norwich_terrier', 'Yorkshire_terrier', 'wire', 'Lakeland_terrier', 'Sealyham_terrier',
    'Airedale', 'cairn', 'Australian_terrier', 'Dandie_Dinmont', 'Boston_bull',
    'miniature_schnauzer', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
    'Tibetan_terrier', 'silky_terrier', 'soft', 'West_Highland_white_terrier', 'Lhasa',
    'flat', 'curly', 'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
    'German_short', 'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
    'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
    'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke',
    'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old_English_sheepdog',
    'Shetland_sheepdog', 'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
    'German_shepherd', 'Doberman', 'miniature_pinscher', 'Greater_Swiss_Mountain_dog',
    'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
    'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog',
    'malamute', 'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg',
    'Newfoundland', 'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond',
    'Brabancon_griffon', 'Pembroke', 'Cardigan', 'toy_poodle', 'miniature_poodle',
    'standard_poodle', 'Mexican_hairless', 'dingo', 'dhole', 'African_hunting_dog'
]

# ── Sección de carga de imagen ────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Sube una foto de un perro",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded:
    # Mostrar la imagen cargada por el usuario
    img = Image.open(uploaded).convert("RGB")
    st.image(img, use_container_width=True)

    if st.button("🐾  Identificar raza"):
        if modelo is None:
            st.error("El modelo no está disponible.")
        else:
            with st.spinner("Analizando imagen..."):
                # Preprocesamiento: redimensionar a 100x100 y normalizar a [0, 1]
                img_resized = img.resize((100, 100))
                img_array  = np.array(img_resized).astype("float32") / 255.0
                img_array  = np.expand_dims(img_array, axis=0)  # Agregar dimensión de batch

                # Inferencia: obtener las 5 predicciones más probables
                predicciones = modelo.predict(img_array)[0]
                top5_idx     = predicciones.argsort()[-5:][::-1]

                raza_pred = clases[top5_idx[0]].replace("_", " ")
                confianza = predicciones[top5_idx[0]] * 100
                bar_width = int(confianza)

            # Mostrar resultado principal y top 5 en la tarjeta de resultados
            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Raza identificada</div>
                <div class="result-breed">{raza_pred.upper()}</div>
                <div class="result-confidence">Confianza: {confianza:.1f}%</div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width:{bar_width}%"></div>
                </div>
                <div class="top-preds">
                    {''.join([
                        f'<div class="top-pred-item"><span>{clases[top5_idx[i]].replace("_"," ").title()}</span><span>{predicciones[top5_idx[i]]*100:.1f}%</span></div>'
                        for i in range(1, 5)
                    ])}
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Placeholder visible cuando no se ha cargado ninguna imagen
    st.markdown("""
    <div class="upload-box">
        <p style="font-size:2.5rem;margin:0">🐾</p>
        <p style="color:#333;margin:0.5rem 0 0;font-size:0.9rem">Arrastra una imagen aquí o usa el botón de carga</p>
    </div>
    """, unsafe_allow_html=True)

# ── Pie de página ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center;color:#222;font-size:0.75rem;letter-spacing:3px'>STANFORD DOGS DATASET · CNN MODEL · 120 BREEDS</p>", unsafe_allow_html=True)
