import torch
from src.model import EmotionCNN
from src.config import MODEL_SAVE_PATH, DATA_PATH
from src.data_loader import load_data
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

def predict(image_tensor):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()

def test_model(test_loader):
    class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    y_true = []
    y_pred = []

    for inputs, labels in test_loader:
        for i in range(inputs.size(0)):
            image_tensor = inputs[i].unsqueeze(0)
            predicted_class = predict(image_tensor)
            y_true.append(labels[i].item())
            y_pred.append(predicted_class)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels, labels=range(len(class_labels))))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=range(len(class_labels))))

if __name__ == "__main__":
    _, test_loader = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    test_model(test_loader)