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

MODEL_URL = "https://huggingface.co/nascafas/test-1/resolve/main/vggface_model.h5"
MODEL_LOCAL_PATH = "vggface_model.h5"

def download_model_if_needed():
    if not os.path.exists(MODEL_LOCAL_PATH):
        print("Downloading VGGFace model from HuggingFace...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_LOCAL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded successfully!")

download_model_if_needed()

app = Flask(__name__)
model = load_model(MODEL_LOCAL_PATH)
target_img = os.path.join(os.getcwd(), 'static/images')

# Initialize MTCNN detector
detector = MTCNN()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    Args:
        image_path: Path to the input image
        target_size: Target size for the extracted face (width, height)
    
    Returns:
        Preprocessed face array or None if no face detected
    """
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
    (original function for comparison or backup)
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

def save_face_crop(image_path, output_path):
    """
    Save the extracted face crop for debugging/visualization
    """
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)
        
        if len(faces) > 0:
            best_face = max(faces, key=lambda x: x['confidence'])
            x, y, width, height = best_face['box']
            
            # Add padding
            padding_x = int(width * 0.1)
            padding_y = int(height * 0.1)
            x1 = max(0, x - padding_x)
            y1 = max(0, y - padding_y)
            x2 = min(img_rgb.shape[1], x + width + padding_x)
            y2 = min(img_rgb.shape[0], y + height + padding_y)
            
            face_crop = img_rgb[y1:y2, x1:x2]
            face_pil = Image.fromarray(face_crop)
            face_pil.save(output_path)
            return True
    except:
        pass
    return False

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
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
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
                
                # Save face crop for visualization (optional)
                face_crop_path = os.path.join('static/images', f'face_{filename}')
                save_face_crop(file_path, face_crop_path)

                patient = patient_data.get(predicted_class, None)

                if patient:
                    # Add confidence score to patient data
                    patient_with_confidence = patient.copy()
                    patient_with_confidence['confidence'] = f"{confidence:.2%}"
                    patient_with_confidence['predicted_class'] = predicted_class
                    
                    return render_template('predict.html',
                                         patient=patient_with_confidence,
                                         prob=class_prediction,
                                         user_image=file_path,
                                         face_image=face_crop_path if os.path.exists(face_crop_path) else None)
                else:
                    return render_template('predict.html',
                                         patient=None,
                                         message=f"Data Pasien Tidak Ditemukan",
                                         user_image=file_path,
                                         face_image=face_crop_path if os.path.exists(face_crop_path) else None)
                                         
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return render_template('predict.html',
                                     patient=None,
                                     message="Error during prediction. Please try again.",
                                     user_image=file_path)
        else:
            return "Unable to read the file. Please check file extension"

@app.route('/test_face_detection')
def test_face_detection():
    """
    Test endpoint to check if MTCNN is working properly
    """
    try:
        # Test with a sample image if available
        test_images = []
        for filename in os.listdir('static/images'):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append(filename)
                break
        
        if test_images:
            test_path = os.path.join('static/images', test_images[0])
            result = extract_face_mtcnn(test_path)
            status = "MTCNN working properly" if result is not None else "No face detected in test image"
        else:
            status = "No test images available"
            
        return f"MTCNN Status: {status}"
    except Exception as e:
        return f"MTCNN Error: {str(e)}"

if __name__ == '__main__':
    # Create images directory if it doesn't exist
    os.makedirs('static/images', exist_ok=True)
    
    # Test MTCNN initialization
    try:
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector.detect_faces(test_array)
        logger.info("MTCNN initialized successfully")
    except Exception as e:
        logger.error(f"MTCNN initialization failed: {str(e)}")
    
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host='0.0.0.0', port=port)