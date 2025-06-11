
from flask import Flask, request, jsonify
import torch
from transformers import pipeline
from PIL import Image
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# Try to import BitsAndBytesConfig, but handle if not available
try:
    from transformers import BitsAndBytesConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    print("Warning: BitsAndBytesConfig not available. Quantization will be disabled.")
hf_token="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"

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model
pipe = None
device = None

def initialize_model():
    """Initialize MedGemma model once at startup"""
    global pipe, device
    
    print("Initializing MedGemma model...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Setup MedGemma model with memory optimization
        if device == "cuda" and QUANTIZATION_AVAILABLE:
            # GPU configuration with device_map and quantization
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "quantization_config": BitsAndBytesConfig(load_in_4bit=True)
            }
        elif device == "cuda":
            # GPU configuration without quantization
            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
            }
        else:
            # CPU configuration without device_map to avoid accelerate issues
            model_kwargs = {
                "torch_dtype": torch.float32,
                # Remove device_map for CPU to avoid accelerate dependency issues
            }
        
        # Initialize the pipeline
        pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", model_kwargs=model_kwargs,token=hf_token)
        
        # Move to device if CPU (since we didn't use device_map)
        if device == "cpu":
            pipe.model = pipe.model.to(device)
            
        pipe.model.generation_config.do_sample = False
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model with device_map, trying simpler approach: {str(e)}")
        
        # Fallback: Simple loading without device_map
        try:
            model_kwargs = {
                "torch_dtype": torch.float32 if device == "cpu" else torch.bfloat16,
            }
            
            pipe = pipeline("image-text-to-text", model="google/medgemma-4b-it", model_kwargs=model_kwargs)
            pipe.model = pipe.model.to(device)
            pipe.model.generation_config.do_sample = False
            print("Model loaded successfully with fallback method!")
            
        except Exception as e2:
            print(f"Failed to load model: {str(e2)}")
            raise e2

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_medical_image(prompt, image_path, system_prompt=None):
    """
    Analyze a medical image using MedGemma
    
    Args:
        prompt (str): What you want to analyze
        image_path (str): Path to the image file
        system_prompt (str, optional): Additional context for analysis
    
    Returns:
        str: Medical analysis report
    """
    try:
        # Load and prepare image
        image = Image.open(image_path)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            })
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        })
        
        # Generate analysis
        print("Analyzing image...")
        output = pipe(text=messages, max_new_tokens=1024)
        response = output[0]["generated_text"][-1]["content"]
        return response
        
    except Exception as e:
        return f"Error during analysis: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": device
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze medical image endpoint
    Expects JSON with:
    - image_path: path to the image file
    - prompt: analysis prompt (optional)
    - system_prompt: system context (optional)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({"error": "image_path is required"}), 400
        
        # Check if file exists
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image file not found: {image_path}"}), 404
        
        # Default prompt if not provided
        prompt = data.get('prompt', "Describe this medical image and identify any abnormalities or notable findings.")
        system_prompt = data.get('system_prompt')
        
        # Analyze the image
        result = analyze_medical_image(prompt, image_path, system_prompt)
        
        return jsonify({
            "success": True,
            "image_path": image_path,
            "prompt": prompt,
            "analysis": result,
            "device_used": device
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-upload', methods=['POST'])
def analyze_uploaded_image():
    """
    Analyze uploaded medical image
    Expects multipart/form-data with:
    - file: image file
    - prompt: analysis prompt (optional)
    - system_prompt: system context (optional)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"File type not allowed. Allowed types: {ALLOWED_EXTENSIONS}"}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get analysis parameters
        prompt = request.form.get('prompt', "Describe this medical image and identify any abnormalities or notable findings.")
        system_prompt = request.form.get('system_prompt')
        
        # Analyze the image
        result = analyze_medical_image(prompt, filepath, system_prompt)
        
        # Clean up uploaded file (optional)
        # os.remove(filepath)
        
        return jsonify({
            "success": True,
            "filename": filename,
            "prompt": prompt,
            "analysis": result,
            "device_used": device
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-base64', methods=['POST'])
def analyze_base64_image():
    """
    Analyze base64 encoded image
    Expects JSON with:
    - image_base64: base64 encoded image
    - prompt: analysis prompt (optional)
    - system_prompt: system context (optional)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_base64 = data.get('image_base64')
        if not image_base64:
            return jsonify({"error": "image_base64 is required"}), 400
        
        # Decode base64 image
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Invalid base64 image data: {str(e)}"}), 400
        
        # Save temporary file
        temp_filename = "temp_image.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_filepath)
        
        # Get analysis parameters
        prompt = data.get('prompt', "Describe this medical image and identify any abnormalities or notable findings.")
        system_prompt = data.get('system_prompt')
        
        # Analyze the image
        result = analyze_medical_image(prompt, temp_filepath, system_prompt)
        
        # Clean up temporary file
        os.remove(temp_filepath)
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "analysis": result,
            "device_used": device
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize model at startup
    initialize_model()
    
    # Run the Flask app
    print("\n🏥 MedGemma Medical Image Analysis API")
    print("="*50)
    print("Available endpoints:")
    print("- GET  /health - Health check")
    print("- POST /analyze - Analyze image by file path")
    print("- POST /analyze-upload - Analyze uploaded image")
    print("- POST /analyze-base64 - Analyze base64 encoded image")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)