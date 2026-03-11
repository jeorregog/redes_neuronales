import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# ----------------------------------
# TITULO
# ----------------------------------

st.title("Clasificación de prendas - Fashion MNIST")


# ----------------------------------
# CARGAR DATASET
# ----------------------------------

@st.cache_data
def cargar_datos():

    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True)

    X = X / 255.0
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = cargar_datos()


# ----------------------------------
# INTERFAZ USUARIO
# ----------------------------------

st.sidebar.header("Configuración del modelo")

tipo_red = st.sidebar.selectbox(
    "Tipo de red",
    ["MLP", "DNN"]
)

activacion = st.sidebar.selectbox(
    "Función de activación",
    ["relu", "tanh", "sigmoid"]
)

num_capas = st.sidebar.slider(
    "Número de capas ocultas",
    1, 5, 2
)

epochs = st.sidebar.slider(
    "Epochs",
    1, 10, 5
)


# ----------------------------------
# MLP
# ----------------------------------

def crear_mlp(activacion):

    model = Sequential()

    model.add(Dense(
        128,
        activation=activacion,
        input_shape=(784,)
    ))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------------
# DNN
# ----------------------------------

def crear_dnn(num_capas, activacion):

    model = Sequential()

    model.add(Dense(
        128,
        activation=activacion,
        input_shape=(784,)
    ))

    for i in range(num_capas - 1):

        model.add(Dense(
            128,
            activation=activacion
        ))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------------
# ENTRENAR MODELO
# ----------------------------------

if st.button("Entrenar red neuronal"):

    if tipo_red == "MLP":

        modelo = crear_mlp(activacion)

    else:

        modelo = crear_dnn(num_capas, activacion)

    history = modelo.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        verbose=1
    )


    # ----------------------------------
    # EVALUACIÓN
    # ----------------------------------

    loss, acc = modelo.evaluate(X_test, y_test)

    st.subheader("Desempeño del modelo")

    st.write("Accuracy:", acc)
    st.write("Loss:", loss)


    # ----------------------------------
    # GRAFICA DE ACCURACY
    # ----------------------------------

    fig, ax = plt.subplots()

    ax.plot(history.history["accuracy"], label="train")
    ax.plot(history.history["val_accuracy"], label="validation")

    ax.set_title("Accuracy del modelo")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    st.pyplot(fig)
