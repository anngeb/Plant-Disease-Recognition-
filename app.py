
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50    import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import numpy as np
import os

app = Flask(__name__)

MODELS = {
    "Konwolucyjna sieć neuronowa": "cnn_model.keras",
    "MobileNetV2": "mobilenetv2_best_v2.keras",
    "ResNet50": "resnet50_stage1_best.keras",
    "EfficientNetB0": "efficientnetb0_trained_best.keras",
}

class_labels = [
    'Jabłoń - chora (parch)', 'Jabłoń - chora (czarna zgnilizna)', 'Jabłoń - chora (rdza cedrowa)', 'Jabłoń - zdrowa',
    'Borówka - zdrowa', 'Wiśnia - chora (mączniak prawdziwy)', 'Wiśnia - zdrowa',
    'Kukurydza - chora (plamistość Cercospora)', 'Kukurydza - chora (rdza zwyczajna)',
    'Kukurydza - chora (północna plamistość liści)', 'Kukurydza - zdrowa',
    'Winorośl - chora (czarna zgnilizna)', 'Winorośl - chora (Esca - czarne odbarwienia)',
    'Winorośl - chora (plamistość liści)', 'Winorośl - zdrowa',
    'Pomarańcza - chora (greening cytrusów)', 'Brzoskwinia - chora (plamistość bakteryjna)',
    'Brzoskwinia - zdrowa', 'Papryka - chora (plamistość bakteryjna)', 'Papryka - zdrowa',
    'Ziemniak - chory (wczesna zaraza)', 'Ziemniak - chory (późna zaraza)', 'Ziemniak - zdrowy',
    'Malina - zdrowa', 'Soja - zdrowa', 'Dynia - chora (mączniak prawdziwy)',
    'Truskawka - chora (przypalenie liści)', 'Truskawka - zdrowa',
    'Pomidor - chory (plamistość bakteryjna)', 'Pomidor - chory (wczesna zaraza)', 'Pomidor - chory (późna zaraza)',
    'Pomidor - chory (pleśń liści)', 'Pomidor - chory (plamistość septoriowa)',
    'Pomidor - chory (przędziorki)', 'Pomidor - chory (target spot)',
    'Pomidor - chory (wirus żółtego zwijania liści)', 'Pomidor - chory (mozaika)', 'Pomidor - zdrowy'
]

#wczytaanie modeli
models = {name: load_model(path) for name, path in MODELS.items() if os.path.exists(path)}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_results = []
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)

            for name, model in models.items():
                # wczytaj i skonwertuj do tablicy bez normalizacji
                img = load_img(file_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_batch = np.expand_dims(img_array, axis=0)

                # model-specific preprocessing
                if name == "ResNet50":
                    img_batch = resnet_preprocess(img_batch.copy())
                elif name == "EfficientNetB0":
                    img_batch = effnet_preprocess(img_batch.copy())
                elif name == "MobileNetV2":
                    img_batch = mobilenet_preprocess(img_batch.copy())
                else:
                    # Twój stary CNN czy inna sieć oczekuje [0,1]
                    img_batch = img_batch / 255.0

                preds = model.predict(img_batch, verbose=0)[0]
                top_idx = np.argmax(preds)
                confidence = preds[top_idx]
                label = class_labels[top_idx] if top_idx < len(class_labels) else f"Klasa {top_idx}"
                prediction_results.append((name, label, f"{confidence*100:.2f}%"))

    return render_template('index.html', predictions=prediction_results, image_file=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

