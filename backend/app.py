from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir,"plant_disease_model.keras")
class_names_path = os.path.join(script_dir, "class_names.json")

# Load class names from JSON file
with open(class_names_path, 'r') as f:
    class_names = json.load(f)

# Get PostgreSQL connection URL from environment
DATABASE_URL = os.getenv('DB_URL')

# PostgreSQL connection helper
def get_db_connection():
    """Get a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None

# Initialize PostgreSQL database
def init_db():
    """Create users table if it doesn't exist"""
    conn = get_db_connection()
    if not conn:
        print("❌ Failed to connect to database")
        return
    
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ PostgreSQL database initialized!")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        if conn:
            conn.close()

# Initialize database on startup
init_db()

# Load model once with error handling
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    print(f"📝 Attempting to create a mock model for testing...")
    try:
        # Try to import and use the mock model creator
        import sys
        sys.path.insert(0, script_dir)
        from create_mock_model import create_mock_model
        model = create_mock_model(len(class_names))
        model.save(model_path)
        print(f"✅ Mock model created and saved successfully!")
        print("⚠️  Note: This is a test model. For production, train the real model using:")
        print("   E:\\smart-leaf\\splitted_dataset\\cnn.ipynb")
    except Exception as e2:
        print(f"❌ Failed to create mock model: {e2}")
        print("⚠️  To fix this:")
        print("   1. Run: python create_mock_model.py")
        print("   2. Or train the real model: E:\\smart-leaf\\splitted_dataset\\cnn.ipynb")
        print("   3. Restart this Flask app")
        model = None

print(f"\n📋 Class names loaded ({len(class_names)} classes) from class_names.json:")
for i, name in enumerate(class_names):
    print(f"  {i:2d}: {name}")
print()

# Image preprocessing - EXACTLY matches your CNN notebook workflow
def preprocess_image(image):
    """
    Matches the exact preprocessing from cnn.ipynb:
    1. Convert PIL to numpy/BGR (simulate cv2.imread)
    2. Convert BGR to RGB (like cv2.cvtColor)
    3. Resize to (256, 256)
    4. Add batch dimension
    5. NO normalization - raw pixel values [0-255]
    """
    print(f"🔍 Original PIL image: size={image.size}, mode={image.mode}")
    
    # Convert PIL Image to numpy array (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert PIL (RGB) to numpy array
    img_array = np.array(image, dtype=np.uint8)
    print(f"🔍 After PIL→numpy: shape={img_array.shape}, dtype={img_array.dtype}")
    
    # PIL gives RGB, but cv2.imread gives BGR. Since your notebook uses cv2.imread 
    # then converts BGR→RGB, we need to ensure we're in RGB format.
    # PIL already gives RGB, so we're good.
    
    # Resize using cv2 (MUST use cv2.resize exactly like notebook)
    img_array = cv2.resize(img_array, (256, 256))
    print(f"🔍 After cv2.resize: shape={img_array.shape}, dtype={img_array.dtype}")
    print(f"🔍 Value range: [{img_array.min()}, {img_array.max()}]")
    
    # NO normalization - use raw pixel values [0-255] (like notebook)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    print(f"🔍 Final shape: {img_array.shape}, dtype={img_array.dtype}")
    
    return img_array

# Route: Health check
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Plant Disease Detection API is running!"})

# Route: Signup
@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ("name", "email", "password")):
            return jsonify({"error": "Missing required fields: name, email, password"}), 400
        
        name = data["name"].strip()
        email = data["email"].strip().lower()
        password = data["password"]
        
        # Basic validation
        if not name or not email or not password:
            return jsonify({"error": "All fields are required"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        
        # Check if email already exists and insert new user
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.close()
                conn.close()
                return jsonify({"error": "Email already registered"}), 409
            
            # Insert new user
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s) RETURNING id",
                (name, email, password)
            )
            user_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                "message": "Signup successful",
                "user": {
                    "id": user_id,
                    "name": name,
                    "email": email
                }
            }), 201
        except Exception as e:
            if conn:
                conn.close()
            return jsonify({"error": str(e)}), 500
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Login
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or not all(k in data for k in ("email", "password")):
            return jsonify({"error": "Missing required fields: email, password"}), 400
        
        email = data["email"].strip().lower()
        password = data["password"]
        
        print(f"🔍 Login attempt - Email: {email}")
        
        # Query database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, email, password FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            
            if user:
                print(f"✅ User found: {user[2]}")
                
                if user[3] == password:
                    cursor.close()
                    conn.close()
                    return jsonify({
                        "message": "Login successful",
                        "user": {
                            "id": user[0],
                            "name": user[1],
                            "email": user[2]
                        }
                    }), 200
                else:
                    cursor.close()
                    conn.close()
                    return jsonify({"error": "Invalid password"}), 401
            else:
                print(f"❌ User not found with email: {email}")
                cursor.close()
                conn.close()
                return jsonify({"error": "Email not registered"}), 401
        except Exception as e:
            if conn:
                conn.close()
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route: Get all users (for testing/debugging)
@app.route("/users", methods=["GET"])
def get_users():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at FROM users")
        users = cursor.fetchall()
        cursor.close()
        conn.close()
        
        users_list = [
            {
                "id": user[0],
                "name": user[1],
                "email": user[2],
                "created_at": user[3]
            }
            for user in users
        ]
        
        return jsonify({"users": users_list}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route: Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded. Please fix the model file."}), 500
        
        # Check if file is uploaded
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        image = Image.open(file.stream)
        img_array = preprocess_image(image)

        # Make prediction - matching your notebook
        print("\n🔍 Making prediction...")
        predictions = model.predict(img_array)[0]
        print(f"Raw predictions shape: {predictions.shape}")
        print(f"Raw predictions (all values): {predictions}")
        print(f"Sum: {predictions.sum():.6f}, Max: {predictions.max():.6f}, Min: {predictions.min():.6f}")
        
        # Your model has softmax activation, so predictions are already probabilities [0-1]
        # Use them directly (like your notebook: np.argmax(pred))
        probabilities = predictions
        
        # Get prediction using argmax (matching your notebook)
        pred_index = np.argmax(probabilities)
        predicted_class = class_names[pred_index]
        confidence = float(probabilities[pred_index] * 100)
        
        print(f"\n🎯 Predicted class index: {pred_index}")
        print(f"🎯 Predicted class name: {predicted_class}")
        print(f"🎯 Confidence: {confidence:.2f}%")
        
        # Sort predictions to get top-5
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_5_predictions = {
            class_names[i]: float(probabilities[i] * 100) 
            for i in top_indices
        }
        
        print("Top 5 predictions:")
        for class_name, conf in top_5_predictions.items():
            print(f"  {class_name}: {conf:.2f}%")
        
        # Prepare response
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_confidences": top_5_predictions
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
