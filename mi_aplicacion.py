# ### Creando front end


#utilizano la libreria  streamlit
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Configuración de la página
st.set_page_config(layout="wide", page_title="Prediccion Adidas o Nike ")

st.write("## Adidas vs Nike Clasificación de imagenes")
st.write(
    "Carga una imagen y nuestro modelo predirá si la imagen corresponde a Adidas o Nike. "
)

# Cargar el modelo (asegúrate de que esté en el mismo directorio o especifica la ruta correcta)
model = tf.keras.models.load_model('cnn_modelo_adidas_nike.keras')# Cambia esto por la ruta correcta

# Función para hacer predicciones
def predict_image(image):
    image = image.resize((224, 224))  # Ajusta al tamaño que tu modelo espera
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

# Cargar imagen desde la interfaz
uploaded_file = st.sidebar.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    
    # Hacer predicción
    prediction = predict_image(image)
    
    # Mostrar resultados
    if prediction[0][0] > 0.89:
        st.write("### Predicción: ** Es Nike**")
    else:
        st.write("### Predicción: **Es Adidas**")

 