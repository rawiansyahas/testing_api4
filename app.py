from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
import cv2
from mtcnn import MTCNN
from PIL import Image
import logging
import requests
from urllib.parse import urlparse

app = Flask(__name__)
target_img = os.path.join(os.getcwd(), 'static/images')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_from_google_drive(file_id, destination):
    """
    Download file from Google Drive using file ID
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def extract_file_id_from_drive_url(url):
    """
    Extract file ID from Google Drive URL
    """
    if 'drive.google.com' in url:
        if '/file/d/' in url:
            # Extract from URL like: https://drive.google.com/file/d/FILE_ID/view
            return url.split('/file/d/')[1].split('/')[0]
        elif 'id=' in url:
            # Extract from URL like: https://drive.google.com/uc?id=FILE_ID
            return url.split('id=')[1].split('&')[0]
    return None

def download_model_from_url(url, local_path):
    """
    Download model from a URL (supports Google Drive)
    """
    try:
        if not os.path.exists(local_path):
            logger.info(f"Downloading model from {url}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Check if it's a Google Drive URL
            file_id = extract_file_id_from_drive_url(url)
            if file_id:
                logger.info(f"Detected Google Drive URL, using file ID: {file_id}")
                download_from_google_drive(file_id, local_path)
            else:
                # Regular URL download
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            logger.info("Model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        return False

def load_model_safely():
    """
    Load model with multiple fallback options
    """
    
    model_path = 'vggface_model.h5'
    
    # Option 1: Try to load local file
    if os.path.exists(model_path):
        try:
            logger.info("Loading model from local file")
            model = load_model(model_path)
            logger.info("Model loaded successfully from local file")
            return model
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
    
    # Option 2: Try to download from Google Drive
    # Your file ID from the URL: https://drive.google.com/file/d/1JQeKTM59S_OaFi-IvU5d0PwDywiD6V77/view
    google_drive_file_id = "1JQeKTM59S_OaFi-IvU5d0PwDywiD6V77"
    
    # You can also set this via environment variable
    model_url = os.environ.get('MODEL_URL', f'https://drive.google.com/file/d/{google_drive_file_id}/view')
    
    if download_model_from_url(model_url, model_path):
        try:
            model = load_model(model_path)
            logger.info("Model loaded successfully from Google Drive")
            return model
        except Exception as e:
            logger.error(f"Failed to load downloaded model: {str(e)}")
    
    # Option 3: Check if model exists in different locations
    possible_paths = [
        './vggface_model.h5',
        '/app/vggface_model.h5',
        os.path.join(os.getcwd(), 'vggface_model.h5'),
        os.path.join(os.path.dirname(__file__), 'vggface_model.h5')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                logger.info(f"Trying to load model from: {path}")
                model = load_model(path)
                logger.info(f"Model loaded successfully from: {path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {str(e)}")
                continue
    
    logger.error("Could not load model from any location")
    return None

# Try to load the model
model = load_model_safely()

# Initialize MTCNN detector with error handling
try:
    detector = MTCNN()
    logger.info("MTCNN detector initialized successfully")
except Exception as e:
    logger.error(f"MTCNN initialization failed: {str(e)}")
    detector = None

@app.route('/')
def index_view():
    return render_template('index.html')

# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def extract_face_mtcnn(image_path, target_size=(224, 224)):
    """
    Extract face from image using MTCNN
    """
    if detector is None:
        logger.warning("MTCNN detector not available")
        return None
        
    try:
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None
            
        # Convert BGR to RGB (MTCNN expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = detector.detect_faces(img_rgb)
        
        if len(faces) == 0:
            logger.warning("No face detected in the image")
            return None
        
        # Get the face with highest confidence
        best_face = max(faces, key=lambda x: x['confidence'])
        
        # Extract bounding box coordinates
        x, y, width, height = best_face['box']
        
        # Add some padding around the face (10% of face dimensions)
        padding_x = int(width * 0.1)
        padding_y = int(height * 0.1)
        
        # Calculate padded coordinates
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img_rgb.shape[1], x + width + padding_x)
        y2 = min(img_rgb.shape[0], y + height + padding_y)
        
        # Crop the face
        face_crop = img_rgb[y1:y2, x1:x2]
        
        # Convert to PIL Image and resize
        face_pil = Image.fromarray(face_crop)
        face_resized = face_pil.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and preprocess
        face_array = np.array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = preprocess_input(face_array.astype(np.float32))
        
        logger.info(f"Face extracted successfully with confidence: {best_face['confidence']:.2f}")
        return face_array
        
    except Exception as e:
        logger.error(f"Error in face extraction: {str(e)}")
        return None

def read_image_fallback(filename, target_size=(224, 224)):
    """
    Fallback function to load image without face detection
    """
    try:
        img = load_img(filename, target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except Exception as e:
        logger.error(f"Error in fallback image loading: {str(e)}")
        return None

# Your patient data dictionary here
patient_data = {
    822: {
        "id": "P001",
        "name": "Harry Styles",
        "address": "123 Main St, Springfield",
        "admission_date": "2025-06-20",
        "ward": "Bangsal A"
    },
    48: {
        "id": "P048",
        "name": "Alan Dale",
        "address": "789 Pine St, Capital City",
        "admission_date": "2025-06-21",
        "ward": "Bangsal C"
    },
    12: {
        "id": "P049",
        "name": "Abraham Benrubi",
        "address": "101 Oak St, North Haverbrook",
        "admission_date": "2025-06-22",
        "ward": "Bangsal D"
    },
    18: {
        "id": "P050",
        "name": "Adam Driver",
        "address": "202 Maple Ave, Ogdenville",
        "admission_date": "2025-06-23",
        "ward": "Bangsal E"
    },
    214: {
        "id": "P051",
        "name": "Billie Joe Armstrong",
        "address": "303 Birch Ln, Brockway",
        "admission_date": "2025-06-24",
        "ward": "Bangsal F"
    },
    200: {
        "id": "P052",
        "name": "Ben Miller",
        "address": "404 Cedar Blvd, Springfield",
        "admission_date": "2025-06-25",
        "ward": "Bangsal G"
    },
    230: {
        "id": "P053",
        "name": "Bojana Novakovic",
        "address": "505 Elm St, Shelbyville",
        "admission_date": "2025-06-26",
        "ward": "Bangsal H"
    },
    312: {
        "id": "P054",
        "name": "Catherine Hardwicke",
        "address": "606 Fir Ave, Ogdenville",
        "admission_date": "2025-06-27",
        "ward": "Bangsal I"
    },
    496: {
        "id": "P055",
        "name": "David Letterman",
        "address": "707 Walnut St, Capital City",
        "admission_date": "2025-06-28",
        "ward": "Bangsal J"
    },
    553: {
        "id": "P056",
        "name": "Dick Van Dyke",
        "address": "808 Redwood Dr, Brockway",
        "admission_date": "2025-06-29",
        "ward": "Bangsal K"
    },
    928: {
        "id": "P057",
        "name": "James Cromwell",
        "address": "909 Chestnut Rd, North Haverbrook",
        "admission_date": "2025-06-30",
        "ward": "Bangsal L"
    },
    758: {
        "id": "P058",
        "name": "George Sampson",
        "address": "111 Palm Ct, Springfield",
        "admission_date": "2025-07-01",
        "ward": "Bangsal M"
    },
    814: {
        "id": "P059",
        "name": "Hannah Murray",
        "address": "222 Sequoia Dr, Ogdenville",
        "admission_date": "2025-07-02",
        "ward": "Bangsal N"
    },
    879: {
        "id": "P060",
        "name": "Irina Shayk",
        "address": "333 Poplar Pl, Capital City",
        "admission_date": "2025-07-03",
        "ward": "Bangsal O"
    },
    912: {
        "id": "P061",
        "name": "Jade Ramsey",
        "address": "444 Spruce St, Shelbyville",
        "admission_date": "2025-07-04",
        "ward": "Bangsal P"
    }
}

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if model is loaded
        if model is None:
            return render_template('predict.html',
                                 patient=None,
                                 message="Model tidak tersedia. Model sedang di-download atau terjadi error.",
                                 user_image=None)
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            
            # Create directory if it doesn't exist
            os.makedirs('static/images', exist_ok=True)
            file.save(file_path)

            # Extract face using MTCNN
            img = extract_face_mtcnn(file_path)
            
            # If face extraction fails, try fallback method
            if img is None:
                logger.warning("MTCNN face extraction failed, using fallback method")
                img = read_image_fallback(file_path)
                
                if img is None:
                    return render_template('predict.html',
                                         patient=None,
                                         message="Unable to process the image. Please try another image.",
                                         user_image=file_path)

            # Make prediction
            try:
                class_prediction = model.predict(img)
                predicted_class = int(np.argmax(class_prediction, axis=1)[0])
                confidence = float(np.max(class_prediction))

                patient = patient_data.get(predicted_class, None)

                if patient:
                    # Add confidence score to patient data
                    patient_with_confidence = patient.copy()
                    patient_with_confidence['confidence'] = f"{confidence:.2%}"
                    patient_with_confidence['predicted_class'] = predicted_class
                    
                    return render_template('predict.html',
                                         patient=patient_with_confidence,
                                         prob=class_prediction,
                                         user_image=file_path)
                else:
                    return render_template('predict.html',
                                         patient=None,
                                         message=f"Data Pasien Tidak Ditemukan (Class: {predicted_class})",
                                         user_image=file_path)
                                         
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return render_template('predict.html',
                                     patient=None,
                                     message="Error during prediction. Please try again.",
                                     user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "mtcnn_loaded": detector is not None
    }

if __name__ == '__main__':
    # Create images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 8000))
    
    app.run(debug=False, host='0.0.0.0', port=port)