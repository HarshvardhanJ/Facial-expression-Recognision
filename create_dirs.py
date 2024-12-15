
import os

def create_required_directories():
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print("Created static directory")

if __name__ == "__main__":
    create_required_directories()