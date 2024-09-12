#!/usr/bin/env python
# coding: utf-8

# #Este conjunto de datos se puede utilizar para construir un modelo CNN que pueda clasificar si un zapato es de marca Adidas o Nike.

# In[83]:


import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
# lee las imágenes, decodifica las imágenes,
# convierte en tensores, cambia el tamaño de las imágenes, las empaca en lotes
from tensorflow.keras.utils import image_dataset_from_directory # similar a la de NLP
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16 # modelo preentrenado
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from keras.layers import Activation, MaxPool2D, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[72]:


import pathlib

direccion_original = pathlib.Path('C:\\Users\\yadko\\Downloads\\bootcam AI\\data\\adidas_nike')



# In[73]:


# Crea objetos de tipo tf.data.Dataset
# text_dataset_from_directory: NLP
train_dataset = image_dataset_from_directory(
    direccion_original / "train",
    image_size = (224, 224),
    batch_size = 32
)
validation_dataset = image_dataset_from_directory(
    direccion_original / "validation",
    image_size = (224, 224),
    batch_size = 32
)
test_dataset = image_dataset_from_directory(
    direccion_original / "test",
    image_size = (224, 224),
    batch_size = 32
)


# In[113]:


train_dataset.class_names


# In[74]:


lenet_5= Sequential()

# C1 Capa Convolucional
lenet_5.add(Conv2D(filters= 6,kernel_size = 5,strides = 1, activation = 'tanh',
                 input_shape = (224, 224, 3), padding = 'same'))

# S2 Capa de pooling
lenet_5.add(AveragePooling2D(pool_size=2, strides=2, padding = 'valid'))

# C3 Capa Convolucional
lenet_5.add(Conv2D(filters = 16, kernel_size = 5, strides = 1,activation = 'tanh',
                 padding = 'valid'))

# S4 Capa de pooling
lenet_5.add(AveragePooling2D(pool_size = 2, strides=2, padding = 'valid'))

# C5 Capa Convolucional
lenet_5.add(Conv2D(filters = 120, kernel_size = 5, strides = 1, activation = 'tanh',
                  padding = 'valid'))

lenet_5.add(Flatten())

# FC6 Capa densa
lenet_5.add(Dense(units = 84, activation = 'tanh'))

#FC7 Capa densa
lenet_5.add(Dense(units = 1, activation = 'sigmoid')) #siempre para problemas de clasificacion

lenet_5.summary()


# In[75]:


lenet_5.compile(loss="binary_crossentropy",
               optimizer="rmsprop",
               metrics=["accuracy"])


# In[76]:


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "cnn_clasificacion_tennis_adidas_nike_lenet_5.keras",
        save_best_only = True,
        monitor = "val_loss"
    ),
    
]


# In[77]:


import os


# Obtén la lista de archivos en el directorio
archivos = os.listdir(direccion_original/"train/adidas")

# Filtra solo los archivos (excluyendo directorios)
archivos = [f for f in archivos if os.path.isfile(os.path.join(direccion_original/"train/adidas", f))]

# Muestra los nombres de los archivos
print(archivos)


# In[78]:


import os

def process_image(image_path):
  # ... your existing code for reading and decoding the image

# Assuming `archivos` contains the names of image files
    for archivo in archivos:
        nombre = direccion_original / "data/adidas_nike" / archivo
        print(f"Processing image: {nombre}")  # Print the constructed path for verification
        process_image(nombre)


# In[79]:


def rename_images(root_dir):
  """
  Renombra todas las imágenes dentro de un directorio y sus subdirectorios,
  eliminando los espacios de los nombres de los archivos.

  Args:
    root_dir: El directorio raíz donde se encuentran las imágenes.
  """

  for root, dirs, files in os.walk(root_dir):
    for file in files:
      if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # Puedes agregar más extensiones si es necesario
        old_name = os.path.join(root, file)
        new_name = os.path.join(root, file.replace(' ', '_'))
        os.rename(old_name, new_name)
        print(f"Renombrado: {old_name} -> {new_name}")

# Reemplaza 'ruta/a/tu/dataset' con la ruta real a tu conjunto de datos
root_directory = 'adidas_nike'
rename_images(root_directory)


# In[80]:


#fit entrenamiento del modelo
historia = lenet_5.fit(
    train_dataset,
    epochs = 20,
    validation_data = validation_dataset,
    callbacks = callbacks,
    shuffle = True
)


# In[82]:


test_modelo = keras.models.load_model('cnn_clasificacion_tennis_adidas_nike_lenet_5.keras')
test_loss,test_acc = test_modelo.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# ### Arquitectura AlexNet

# In[91]:


base_dir = 'C:/Users/yadko/Downloads/bootcam AI/data'  # Ajusta según tu ruta
train_dir = os.path.join(base_dir, 'adidas_nike', 'train')

datagen = ImageDataGenerator(...)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)


# In[92]:


modelo = Sequential()
# primera capa: (CONV + pool + batchroom)
# Para calcular la dimensión del ouptput de la capa se usa la formula
#(224(largo/ancho de la imagen)-11(largo/ancho kernel))/4(largo7ancho strides)+1= 54
modelo.add(
    Conv2D(filters = 96, kernel_size=(11, 11),
           strides = (4, 4), padding = 'valid',
           input_shape = (224, 224, 3)))
## Otra forma de adicionar la función de activación en una capa
modelo.add(Activation('relu'))
# Cálculo del tamaño de la siguiente capa es:
#(54 (largo/ancho output anterior) - 3)/2 + 1 = 26
modelo.add(MaxPool2D(pool_size=(3,3), strides = (2,2)))
modelo.add(BatchNormalization())
# segunda capa (CONV + pool + batchnorm)
# kernel_regularizer es lo mismo que la regularización L2
# Ayuda a reducir overfitting en el modelo
#(26 - 5)/1 + 1 = 21 pero como el padding es 'same' no se altera el tamaño del
# output
modelo.add(Conv2D(filters=256, kernel_size=(5,5), strides = (1, 1), padding = 'same',
                  kernel_regularizer = l2(0.0005)))
modelo.add(Activation('relu'))
# (26 - 3)/2 + 1 = 12
modelo.add(MaxPool2D(pool_size=(3, 3), strides = (2, 2), padding = 'valid'))
modelo.add(BatchNormalization())
# tercera capa (CONV + batchnorm) # Los autores no adicionaron una capa de Pooling
modelo.add(Conv2D(filters = 384, kernel_size=(3, 3),
                  strides=(1, 1), padding = 'same',
                  kernel_regularizer = l2(0.0005)))
modelo.add(Activation('relu'))
modelo.add(BatchNormalization())
# Cuarta capa (CONV + batchnorm)
modelo.add(Conv2D(filters = 384, kernel_size = (3, 3), strides=(1, 1), padding = 'same',
                  kernel_regularizer = l2(0.0005)))
modelo.add(Activation('relu'))
modelo.add(BatchNormalization())
# quinta capa (CONV + batchnorm)
modelo.add(Conv2D(filters=256, kernel_size=(3, 3), strides = (1, 1),
                  padding = 'same',
                  kernel_regularizer = l2(0.0005)))
modelo.add(Activation('relu'))
modelo.add(BatchNormalization())
#(12 - 3)/2 + 1 = 5
modelo.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding = 'valid'))
# 5*5*256 = 6400
modelo.add(Flatten())

# Sexta capa (Capa densa + dropout)
# 4906
modelo.add(Dense(units = 4906, activation='relu'))
modelo.add(Dropout(0.5))

# Séptima capa (Capa densa)
modelo.add(Dense(units=4096, activation='relu'))
modelo.add(Dropout(0.5))
#Octava capa (Capa densa)
modelo.add(Dense(units = 1, activation='sigmoid'))
modelo.summary()


# In[107]:


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath = "cnn_modelo_adidas_nike.keras",
        save_best_only = True,
        monitor = "val_loss",
        verbose = 1
    )
]


# In[108]:


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"), # Refleja la imagen con respecto a un eje vertical (espejo)
        layers.RandomRotation(0.01), # Rota la imagen un 10% como 36°
        layers.RandomZoom(0.2) # Acerca la imagen un 20%
    ]
)


# In[109]:


# Krizhevsky varia la taza de aprendizaje dividiendola por 10 (0.1)
# cuando el valor de la función de costo no se reduce
reducir_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = np.sqrt(0.1)
)
optimizer = optimizers.SGD(learning_rate = 0.01, momentum = 0.9)
modelo.compile(loss = 'binary_crossentropy', optimizer = optimizer,
               metrics = ['accuracy'])
modelo.fit(train_dataset, batch_size=128, epochs = 20,
           validation_data = validation_dataset, verbose = 2,
           callbacks = [reducir_lr,callbacks])


# In[110]:


###TESTEO EL MODELO

modelo_da = keras.models.load_model(
"cnn_modelo_adidas_nike.keras")
test_loss, test_acc = modelo_da.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")


# In[114]:


estado_0 = [] # adidas
estado_1 = [] # nike
for lote in test_dataset:
    images, labels = lote
    for i, label in enumerate(labels):
        if label == 0:
            estado_0.append(images[i])
        elif label == 1:
            estado_1.append(images[i])

    break


# Graficando una imagen por cada categoría

# In[121]:


# Creando una figura y un conjunto de subgráficos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 4))
ax1.imshow(estado_0[1].numpy().astype("uint8"))
ax1.axis("off")
ax1.set_title("adidas")
ax2.imshow(estado_1[4].numpy().astype("uint8"))
ax2.axis("off")
ax2.set_title("nike")
# Ajustar el diseño
plt.tight_layout()


# In[122]:


def preprocesamiento_imagen(imagen_arreglo):
    img_array = imagen_arreglo/255 # Estandarizando la imagen
    img_array = np.expand_dims(imagen_arreglo, axis=0)  # adicionando la dimensión del lote
    return img_array


# ###
# Predecir si es una imagen es adidas o nike

# In[123]:


img_array_0 = preprocesamiento_imagen(estado_0[0])
modelo.predict(img_array_0)


# In[124]:


img_array_0_1 = preprocesamiento_imagen(estado_0[1]) #cambiar el [1], varia la imagen a tomar del data set, se debe cambiar en la parte de crear las figuras.
modelo.predict(img_array_0_1)


# In[118]:


img_array_1 = preprocesamiento_imagen(estado_1[4])
modelo.predict(img_array_1)


# ### Creando front end

# In[3]:


#utilizano la libreria  streamlit
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Configuración de la página
st.set_page_config(layout="wide", page_title="Adidas vs Nike Predictor")

st.write("## Adidas vs Nike Image Classifier")
st.write(
    "Carga una imagen y nuestro modelo predirá si la imagen corresponde a Adidas o Nike. "
)

# Cargar el modelo (asegúrate de que esté en el mismo directorio o especifica la ruta correcta)
model = tf.keras.models.load_model('cnn_modelo_adidas_nike.keras')  # Cambia esto por la ruta correcta
# Cambia esto por la ruta correcta

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
    if prediction[0][0] > 0.5:
        st.write("### Predicción: **Nike**")
    else:
        st.write("### Predicción: **Adidas**")

