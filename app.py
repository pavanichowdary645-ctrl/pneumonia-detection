
# Pneumonia Detection Using XAI (Flask + PneumoFusion-XAI+)
# ------------------------------------------------------------
import os, datetime, time, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, send_file
from fpdf import FPDF

# ------------------------------------------------------------
#  App setup & directories
# ------------------------------------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
#  Load trained models
# ------------------------------------------------------------
vgg_model = tf.keras.models.load_model("vgg_finetuned.h5")
resnet_model = tf.keras.models.load_model("resnet_finetuned.h5")
densenet_model = tf.keras.models.load_model("densenet_model_trained.h5")
models = [vgg_model, resnet_model, densenet_model]

print("✅ Models loaded successfully!")


# ------------------------------------------------------------
#  PneumoFusion-XAI+
# ------------------------------------------------------------
def get_activation_maps(models, img_array):
    feature_maps = []
    for model in models:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                feature_model = Model(inputs=model.input, outputs=layer.output)
                fmap = feature_model.predict(img_array)
                feature_maps.append(fmap)
                break
    return feature_maps


def pneumofusion_xai(models, img_array):
    feature_maps = get_activation_maps(models, img_array)

    resized_maps = []
    for fmap in feature_maps:
        activation = np.mean(fmap, axis=-1)[0]
        resized = cv2.resize(activation, (7, 7))
        resized_maps.append(resized)

    fused_map = np.mean(np.array(resized_maps), axis=0)
    fused_map = (fused_map - fused_map.min()) / (fused_map.max() - fused_map.min() + 1e-8)

    heatmap = cv2.resize(fused_map, (224, 224))
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), 0)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    return heatmap_colored


def estimate_pneumonia_severity(heatmap):
    gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
    intensity = np.mean(gray) / 255.0

    if intensity < 0.3:
        return "Mild Pneumonia", round(intensity * 100, 1)
    elif intensity < 0.6:
        return "Moderate Pneumonia", round(intensity * 100, 1)
    else:
        return "Severe Pneumonia", round(intensity * 100, 1)


# ------------------------------------------------------------
#  PDF Generator
# ------------------------------------------------------------
def generate_pdf(upload_path, result_path, prediction, confidence, severity, intensity, time_taken):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, "PneumoFusion-XAI Diagnostic Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, f"Prediction: {prediction} ({confidence}%)", ln=True)
    pdf.cell(200, 10, f"Severity: {severity} ({intensity}%)", ln=True)
    pdf.cell(200, 10, f"Processing Time: {time_taken:.2f}s", ln=True)

    pdf.ln(10)
    pdf.image(upload_path, x=15, y=70, w=80)
    pdf.image(result_path, x=110, y=70, w=80)

    pdf.output("PneumoFusion_Report.pdf")
    return "PneumoFusion_Report.pdf"

# ------------------------------------------------------------
#  X-RAY VALIDATION FUNCTION
# ------------------------------------------------------------
def is_xray_image(img):
    """
    Simple grayscale + texture check to identify chest X-ray images.
    Works by measuring:
    - How grayscale the image is
    - The contrast level (X-rays have higher contrast)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 1) Check if image is almost grayscale
    diff = np.abs(img[:, :, 0] - gray)
    grayscale_score = np.mean(diff)

    # 2) Check contrast (X-rays have higher contrast)
    contrast = np.std(gray)

    # Conditions for chest X-ray
    if grayscale_score < 12 and contrast > 25:
        return True
    return False

# ------------------------------------------------------------
#  Flask Routes
# ------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        start_time = time.time()
        file = request.files['file']

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # ------------------------------------------------------------
            # Load image EXACTLY like Colab (fixes wrong heatmap overlay)
            # ------------------------------------------------------------
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
            original = tf.keras.preprocessing.image.img_to_array(img)
            original = original.astype(np.uint8)
            # ------------------------------------------------------------
            # VALIDATE X-RAY INPUT
            # ------------------------------------------------------------
            if not is_xray_image(original):
                return render_template(
                    "index.html",
                    error_message="❌ Invalid image! Please upload a Chest X-Ray only.",
                    uploaded_image=None
                )


            img_array = original / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Ensemble prediction
            preds = [model.predict(img_array)[0][0] for model in models]
            avg_pred = np.mean(preds)

            result_label = "PNEUMONIA" if avg_pred > 0.5 else "NORMAL"
            confidence = round(avg_pred * 100 if avg_pred > 0.5 else (1 - avg_pred) * 100, 2)

            # Generate heatmap
            heatmap = pneumofusion_xai(models, img_array)

            # ✔ FIXED: heatmap overlay exactly like Colab
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
            overlay = np.uint8(np.clip(overlay * 1.10, 0, 255))

            result_filename = f"result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            result_path = os.path.join(RESULT_FOLDER, result_filename)

            cv2.imwrite(result_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Severity estimation
            if result_label == "PNEUMONIA":
                severity, intensity = estimate_pneumonia_severity(heatmap)
            else:
                severity = "No Pneumonia"
                intensity = 0.0

            time_taken = time.time() - start_time

            report_path = generate_pdf(filepath, result_path, result_label, confidence, severity, intensity, time_taken)

            return render_template(
                'index.html',
                prediction=result_label,
                confidence=confidence,
                severity=severity,
                intensity=intensity,
                result_image=result_filename,
                uploaded_image=file.filename,
                report_available=True,
                processing_time=round(time_taken, 2)
            )

    return render_template('index.html')


@app.route('/download_report')
def download_report():
    return send_file("PneumoFusion_Report.pdf", as_attachment=True)


# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
