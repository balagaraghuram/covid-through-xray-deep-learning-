import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import streamlit as st
from PIL import Image

# Streamlit app title
st.title("COVID-19 Image Classifier")

# Sidebar for user inputs
st.sidebar.header("Settings")
dataset_path = st.sidebar.text_input("Dataset Path", value="path/to/dataset")
plot_path = st.sidebar.text_input("Plot Save Path", value="plot.png")
model_path = st.sidebar.text_input("Model Save Path", value="covid19.model")

# Hyperparameters
INIT_LR = 1e-3
EPOCHS = 25
BS = 8

# Load dataset
if st.sidebar.button("Load Dataset"):
    st.write("[INFO] Loading images...")
    try:
        imagePaths = list(paths.list_images(dataset_path))
        data = []
        labels = []

        for imagePath in imagePaths:
            label = imagePath.split(os.path.sep)[-2]
            image = Image.open(imagePath).convert("RGB")
            image = image.resize((224, 224))
            image = np.array(image)
            data.append(image)
            labels.append(label)

        data = np.array(data) / 255.0
        labels = np.array(labels)

        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

        st.write(f"[INFO] Loaded {len(imagePaths)} images.")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Initialize and compile the model
if st.sidebar.button("Train Model"):
    try:
        st.write("[INFO] Initializing model...")
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)

        model = Model(inputs=baseModel.input, outputs=headModel)

        for layer in baseModel.layers:
            layer.trainable = False

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

        st.write("[INFO] Training model...")
        trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
        H = model.fit_generator(
            trainAug.flow(trainX, trainY, batch_size=BS),
            steps_per_epoch=len(trainX) // BS,
            validation_data=(testX, testY),
            validation_steps=len(testX) // BS,
            epochs=EPOCHS,
        )

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="Training Loss")
        plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="Validation Loss")
        plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="Training Accuracy")
        plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(plot_path)
        st.pyplot(plt)

        model.save(model_path)
        st.write("[INFO] Model saved successfully.")
    except Exception as e:
        st.error(f"Error during training: {e}")

# Evaluate the model
if st.sidebar.button("Evaluate Model"):
    try:
        st.write("[INFO] Evaluating network...")
        predIdxs = model.predict(testX, batch_size=BS)
        predIdxs = np.argmax(predIdxs, axis=1)
        st.text(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

        cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
        st.write("Confusion Matrix:")
        st.write(cm)

        total = sum(sum(cm))
        acc = (cm[0, 0] + cm[1, 1]) / total
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Sensitivity: {sensitivity:.4f}")
        st.write(f"Specificity: {specificity:.4f}")
    except Exception as e:
        st.error(f"Error during evaluation: {e}")
