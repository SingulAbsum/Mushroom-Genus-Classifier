# app.py
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision import transforms
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
#pip install streamlit torch torchvision numpy scikit-learn matplotlib seaborn pillow joblib
#use above line to install required packages if not already installed in the command prompt
#python -c "import streamlit, torch, torchvision, sklearn, matplotlib, seaborn, PIL, joblib, numpy; print('All imports OK')"
#command for verifying installations


st.set_page_config(page_title="Mushroom Classifier (VGG16 + SVM)", layout="centered")

# ---------------------------
# File paths (edit if needed)
# ---------------------------
MODEL_PATH = r"C:\\Users\\User\\Downloads\\final_svm_model.pth"
SVM_PATH   = r"C:\\Users\\User\\Downloads\\svm_linear_mushrooms.joblib"
#edit above paths depending on where the files were saved


# ---------------------------
# Class mapping 
# ---------------------------
classes = ["Agaricus","Amanita","Boletus","Cortinarius",
           "Entoloma","Hygrocybe","Lactarius","Russula","Suillus"]

# ---------------------------
# Preprocessing (inference ONLY)
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------
# Feature extraction from the CNN model
# ---------------------------
def extract_features(model, pil_image):
    """Return numpy array shape (1, 256)"""
    x = preprocess(pil_image).unsqueeze(0)  # (1,3,224,224)
    x = x.to(torch.device("cpu"))
    model = model.to(torch.device("cpu"))
    with torch.no_grad():
        feats = model(x)  # tensor shape (1,256)
    return feats.cpu().numpy()

# ---------------------------
# Load CNN: recreating the same nn.Sequential as used in training
# ---------------------------
@st.cache_resource
def load_cnn(model_path=MODEL_PATH):
    # Create exact same architecture as training
    vgg = vgg16(weights=None)  # important: don't load ImageNet weights here
    model = nn.Sequential(
        vgg.features,
        vgg.avgpool,
        nn.Flatten(),
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(0.6)
    )
# ---------------------------
# Load CNN:  loading weights from the checkpoint or our saved model file,
# if the path exists or our file is found,we try to load the models weights and 
# if we encounter weights or model sections mismatch we load the weights with strict=False
# in order to load what we can and ignore the rest.
# ---------------------------
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            # checkpoint might be a state_dict directly or a dict with "model_state_dict"
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            st.warning(f"Warning loading CNN weights: {e}")
    else:
        st.warning(f"Model file not found at {model_path}. Using random-initialized model (predictions will be meaningless).")

    model.eval()
    return model

# ---------------------------
# Load SVM
# ---------------------------
@st.cache_resource
def load_svm(svm_path=SVM_PATH):
    if os.path.exists(svm_path):
        try:
            return joblib.load(svm_path)
        except Exception as e:
            st.error(f"Failed to load SVM file: {e}")
            return None
    else:
        st.warning(f"SVM file not found at {svm_path}.")
        return None


# ---------------------------
# UI
# ---------------------------
st.title("🍄 Mushroom Classifier (VGG16 + SVM)")

st.markdown("Upload a mushroom image and get predictions (index + class name).")
st.markdown("Model must be the same one used during SVM training (architecture + weights).")
#loaders
cnn = load_cnn()
svm = load_svm()

# uploader
uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features
    feats = extract_features(cnn, image)  # (1,256)
    st.write(f"Feature shape: {feats.shape}")

    if svm is None:
        st.error("SVM model not loaded — cannot predict.")
    else:
        try:
            pred_idx = int(svm.predict(feats)[0])# get predicted class index
            pred_class = classes[pred_idx] if 0 <= pred_idx < len(classes) else str(pred_idx)
            st.success(f"Prediction → Index: {pred_idx}, Class: {pred_class}")# display prediction

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------
# Metrics display
# ---------------------------
st.subheader("📊 Model Test Metrics")

# ---------------------------
# Static (fixed) metrics
# ---------------------------
STATIC_ACCURACY = 0.9114391143911439  # SVM test accuracy
STATIC_LOSS = None  # SVM has no CE loss
STATIC_CM = [
    [36, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 23, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 33, 1, 0, 0, 1, 1, 0],
    [0, 1, 0, 25, 2, 0, 2, 2, 2],
    [0, 1, 0, 0, 27, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 28, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 25, 0, 2],
    [2, 1, 1, 0, 0, 1, 0, 18, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 27],
]
STATIC_SVM_CM = [
    ["  ","Precision","Recall","F1-Score","Support"],
    ["0","   1.00  ", " 0.97 ","  0.99  ", "  37  "],
    ["1","   0.88  ", " 0.92 ","  0.90  ", "  25  "],
    ["2","   0.97  ", " 0.92 ","  0.94  ", "  36  "],
    ["3","   0.85  ", " 0.82 ","  0.84  ", "  34  "],
    ["4","   0.96  ", " 0.90 ","  0.93  ", "  30  "],
    ["5","   1.00  ", " 1.00 ","  1.00  ", "  28  "],
    ["6","   0.82  ", " 0.79 ","  0.81  ", "  24  "],
    ["7","   0.70  ", " 0.88 ","  0.78  ", "  37  "],
    ["8","   1.00  ", " 1.00 ","  1.00  ", "  28  "],
    ["accuracy","           ", "  0.91 ","   271  "],
    ["macro avg","0.91","0.91","  0.91 ","  271   "],
    ["weighted avg","0.91","0.91","0.91","  271   "],
],

# ---------------------------
# Display
# ---------------------------
st.write(f"**CNN Accuracy:** {0.8930 * 100:.2f}%")
st.write(f"**CNN Loss:** {0.3576:.2f}")
st.write(f"**SVM Accuracy:** {STATIC_ACCURACY * 100:.2f}%")

st.write("**CNN Confusion Matrix:**")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    np.array(STATIC_CM),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
st.write(STATIC_SVM_CM)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
st.pyplot(fig)

# Footer help
st.markdown("---")
st.markdown("**Notes / debugging:**")
st.markdown(
    "- Make sure `MODEL_PATH` points to the checkpoint you saved after training (it should contain a `model_state_dict`).\n"
    "- When creating the CNN inside Streamlit we use `vgg16(weights=None)` and build the exact same `nn.Sequential` you used during training.\n"
    "- Preprocessing must match the validation/test transforms used when extracting features for SVM training.\n"
    "- If predictions still favor a few classes, verify the saved `model_state_dict` is the one from the trained `my_model` and that the SVM was trained on those features."
)
