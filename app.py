from flask import Flask, render_template, request, redirect, url_for
import torch
from src.model import SimpleCNN, ResNetModel  
from src.config import MODEL_SAVE_PATH
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

model_type = 'SimpleCNN'  
if model_type == 'SimpleCNN':
    model = SimpleCNN()
elif model_type == 'ResNet':
    model = ResNetModel()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        confidence_score = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()
    return predicted.item(), confidence_score

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_folder = app.config['UPLOAD_FOLDER']
            image_path = os.path.join(image_folder, image.filename)
            os.makedirs(image_folder, exist_ok=True)
            image.save(image_path)
            class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            predicted_class, confidence = predict_image(image_path)
            label = class_labels[predicted_class]
            return render_template('result.html', label=label, confidence=confidence, image_path=image.filename)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)