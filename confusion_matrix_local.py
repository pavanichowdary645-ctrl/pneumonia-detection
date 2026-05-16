import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt

# ----------------------------
# Test dataset path
# ----------------------------
TEST_DIR = "dataset/test"
IMG_SIZE = (224, 224)

# ----------------------------
# 1) Create test generator
# ----------------------------
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

y_true = test_gen.classes
print("✅ Loaded test set. Total images:", len(y_true))

# ----------------------------
# 2) Load saved models
# ----------------------------
print("⏳ Loading models...")
vgg_model    = tf.keras.models.load_model("vgg_finetuned.h5")
resnet_model = tf.keras.models.load_model("resnet_finetuned.h5")
dense_model  = tf.keras.models.load_model("densenet_model_trained.h5")
print("✅ All models loaded successfully.")

# ----------------------------
# 3) Helper: Predict
# ----------------------------
def get_probabilities(model, generator):
    return model.predict(generator).flatten()

def get_binary(preds):
    return (preds > 0.5).astype(int)

# ----------------------------
# 4) Predictions
# ----------------------------
print("⏳ Predicting with VGG16...")
vgg_prob = get_probabilities(vgg_model, test_gen)
vgg_pred = get_binary(vgg_prob)

print("⏳ Predicting with ResNet50...")
res_prob = get_probabilities(resnet_model, test_gen)
res_pred = get_binary(res_prob)

print("⏳ Predicting with DenseNet121...")
dense_prob = get_probabilities(dense_model, test_gen)
dense_pred = get_binary(dense_prob)

# Ensemble
print("⏳ Predicting with ensemble...")
ensemble_prob = (vgg_prob  + dense_prob) / 2
ensemble_pred = get_binary(ensemble_prob)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_classification_table(y_true, y_pred, model_name):
    # Convert classification report to dict → DataFrame
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose().round(3)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )

    # ---- Styling ----
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.4)

    # Header color (column names)
    for key, cell in table.get_celld().items():
        row, col = key
        if row == 0:
            cell.set_facecolor("#4C72B0")   # blue
            cell.set_text_props(color='white', weight='bold')

    # Row labels color styling
    for r in range(1, len(df) + 1):
        table[(r, -1)].set_facecolor("#DDDDDD")

    # Alternate row colors
    for r in range(1, len(df) + 1):
        color = "#F7F7F7" if r % 2 == 0 else "#EFEFEF"
        for c in range(len(df.columns)):
            table[(r, c)].set_facecolor(color)

    plt.title(f"{model_name} - Classification Report", fontsize=16, weight='bold')
    plt.show()
print("\n📊 Showing Classification Tables...\n")

show_classification_table(y_true, vgg_pred, "VGG16")
show_classification_table(y_true, res_pred, "ResNet50")
show_classification_table(y_true, dense_pred, "DenseNet121")
show_classification_table(y_true, ensemble_pred, "Ensemble Model")


# ----------------------------
# 5) Confusion matrix helper
# ----------------------------
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Normal', 'Pneumonia'])
    plt.figure(figsize=(5,4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    print("\n", title)
    print(cm)

# ----------------------------
# 6) Show confusion matrices
# ----------------------------
plot_cm(y_true, vgg_pred, "VGG16 Confusion Matrix")
plot_cm(y_true, res_pred, "ResNet50 Confusion Matrix")
plot_cm(y_true, dense_pred, "DenseNet121 Confusion Matrix")
plot_cm(y_true, ensemble_pred, "Ensemble Confusion Matrix")

# ----------------------------
# 7) Classification Reports
# ----------------------------
def show_report(name, y_true, y_pred):
    print(f"\n📘 Classification Report: {name}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

show_report("VGG16", y_true, vgg_pred)
show_report("ResNet50", y_true, res_pred)
show_report("DenseNet121", y_true, dense_pred)
show_report("Ensemble", y_true, ensemble_pred)

# ----------------------------
# 8) Accuracy Comparison Graph
# ----------------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

vgg_acc   = accuracy(y_true, vgg_pred)
res_acc   = accuracy(y_true, res_pred)
dense_acc = accuracy(y_true, dense_pred)
ens_acc   = accuracy(y_true, ensemble_pred)

plt.figure(figsize=(7,5))
plt.bar(["VGG16","ResNet50","DenseNet121","Ensemble"],
        [vgg_acc, res_acc, dense_acc, ens_acc],
        color=["#4C72B0","#55A868","#C44E52","#8172B2"])
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.title("Model Accuracy Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

print("\n📊 Accuracies:")
print("VGG16 Accuracy:", vgg_acc)
print("ResNet50 Accuracy:", res_acc)
print("DenseNet121 Accuracy:", dense_acc)
print("Ensemble Accuracy:", ens_acc)
import matplotlib.pyplot as plt

def show_accuracy_table(title, acc_value):
    fig, ax = plt.subplots(figsize=(4,2))
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ["Model", title],
        ["Accuracy", f"{acc_value*100:.2f}%"]
    ]

    table = ax.table(
        cellText=table_data,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    plt.title(f"{title} Accuracy Table", fontsize=14)
    plt.show()






# ----------------------------
# 9) INDIVIDUAL ROC CURVES
# ----------------------------
from sklearn.metrics import roc_curve, auc

def plot_single_roc(model_name, y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    auc_value = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}", linewidth=2)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    return auc_value

print("\n🔵 Generating Individual ROC Curves...")

auc_vgg   = plot_single_roc("VGG16", y_true, vgg_prob)
auc_res   = plot_single_roc("ResNet50", y_true, res_prob)
auc_dense = plot_single_roc("DenseNet121", y_true, dense_prob)
auc_ens   = plot_single_roc("Ensemble Model", y_true, ensemble_prob)

print("\n🔵 AUC Scores:")
print("VGG16 AUC:", auc_vgg)
print("ResNet50 AUC:", auc_res)
print("DenseNet121 AUC:", auc_dense)
print("Ensemble AUC:", auc_ens)
