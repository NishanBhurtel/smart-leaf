from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import sqlite3
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "my_model.keras")
db_path = os.path.join(script_dir, "smartleaf.db")

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Database initialized!")

# Initialize database on startup
init_db()

# Load model once with error handling
try:
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("⚠️  The app will start but predictions won't work until the model is fixed.")
    model = None

# Class names
class_names = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_leaf",
    "Tomato_Two_spotted_spider_mite"
]

# Image preprocessing
def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((256, 256))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
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
        
        # Check if email already exists
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        existing = cursor.fetchone()
        
        if existing:
            conn.close()
            return jsonify({"error": "Email already registered"}), 409
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name, email, password)
        )
        conn.commit()
        user_id = cursor.lastrowid
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
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # First check if user exists
        cursor.execute("SELECT id, name, email, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if user:
            print(f"✅ User found: {user[2]}")
            print(f"🔑 Stored password: {user[3]}")
            print(f"🔑 Provided password: {password}")
            print(f"🔑 Match: {user[3] == password}")
            
            if user[3] == password:
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
                conn.close()
                return jsonify({"error": "Invalid password"}), 401
        else:
            print(f"❌ User not found with email: {email}")
            conn.close()
            return jsonify({"error": "Email not registered"}), 401
            
    except Exception as e:
        print(f"❌ Login error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route: Get all users (for testing/debugging)
@app.route("/users", methods=["GET"])
def get_users():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at FROM users")
        users = cursor.fetchall()
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

        # Make prediction
        predictions = model.predict(img_array)[0]
        pred_index = np.argmax(predictions)
        predicted_class = class_names[pred_index]
        confidence = float(predictions[pred_index] * 100)

        # Prepare response
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_confidences": {
                class_names[i]: float(predictions[i] * 100) for i in range(len(class_names))
            }
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
