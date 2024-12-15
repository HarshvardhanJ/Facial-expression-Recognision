from src.config import *
from src.data_loader import load_data
from src.train import train_model

def main():
    train_loader, test_loader = load_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    model = train_model(train_loader, test_loader)
    # ...existing code...

if __name__ == '__main__':
    main()

