from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import time
import threading
from datetime import datetime
import zipfile
import shutil
import base64
from io import BytesIO

# Import TensorFlow first
import tensorflow as tf
from tensorflow import keras
import numpy as np
from app.efficientnet_trainer import EfficientNetV2Trainer
from app.gradcam_generator import GradCAMGenerator
# Import our custom modules
from app.data_processor import DataProcessor
from app.model_builder import ModelBuilder
from app.trainer import ModelTrainer
from app.exporter import ModelExporter
from app.model_tester import ModelTester

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'trained_models'
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Global variables for training state
training_state = {
    'status': 'idle',  # idle, training, completed, error
    'progress': 0,
    'epoch': 0,
    'current_phase': 1,  # For EfficientNet two-phase training
    'history': [],
    'model': None,
    'results': None,
    'error_message': None,
    # إضافة معلومات الـ iterations
    'current_batch': 0,
    'total_batches': 0,
    'batch_loss': 0.0,
    'batch_accuracy': 0.0,
    'iteration_details': []
}
current_model_tester = None

# ====================================================
# UTILITY FUNCTIONS
# ====================================================

def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to Python native types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

# ====================================================
# API ROUTES
# ====================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Neural Network Trainer API is running'})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and data preprocessing"""
    import traceback
    
    try:
        print(f"\n🔍 Upload Debug - Request received at {datetime.now()}")
        
        # Check if file is in request
        if 'file' not in request.files:
            print("❌ Upload Debug - No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("❌ Upload Debug - Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📁 Upload Debug - File info: {file.filename}, {file.content_length} bytes, type: {file.content_type}")
        
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        print(f"💾 Upload Debug - Saving to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            print(f"❌ Upload Debug - File not saved: {filepath}")
            return jsonify({'error': 'Failed to save file'}), 500
            
        file_size = os.path.getsize(filepath)
        print(f"✅ Upload Debug - File saved successfully: {file_size} bytes")
        
        # Process the uploaded data
        print(f"🔧 Upload Debug - Starting data processing...")
        processor = DataProcessor()
        
        try:
            preview_data = processor.process_file(filepath, timestamp)
            print(f"✅ Upload Debug - Processing successful, type: {preview_data.get('type')}")
        except Exception as process_error:
            print(f"❌ Upload Debug - Processing failed: {str(process_error)}")
            print(f"🗜 Traceback: {traceback.format_exc()}")
            return jsonify({
                'error': f'Failed to process file: {str(process_error)}',
                'debug_info': {
                    'processing_error': str(process_error),
                    'file_path': filepath,
                    'file_size': file_size
                }
            }), 500
        
        # Add session_id to preview data
        preview_data['session_id'] = timestamp
        preview_data['upload_timestamp'] = timestamp
        
        # Store file info in session
        session_data = {
            'filepath': filepath,
            'original_filename': file.filename,
            'upload_timestamp': timestamp,
            'preview': preview_data,
            'file_size': file_size,
            'debug_info': {
                'processed_at': datetime.now().isoformat(),
                'file_type': file.content_type
            }
        }
        
        # Save session data to file
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_session.json")
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"🎉 Upload Debug - Complete! Session: {timestamp}")
        
        return jsonify({
            'message': 'File uploaded successfully',
            'session_id': timestamp,
            'preview': preview_data,
            'debug_info': {
                'file_size': file_size,
                'processing_time': 'N/A',
                'session_file': session_file
            }
        })
        
    except Exception as e:
        error_msg = str(e)
        print(f"💥 Upload Debug - Unexpected error: {error_msg}")
        print(f"🗜 Full traceback: {traceback.format_exc()}")
        
        return jsonify({
            'error': f'Upload failed: {error_msg}',
            'debug_info': {
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
        }), 500

# إضافة إلى app.py

@app.route('/api/stop-training', methods=['POST'])
def stop_training():
    """Stop current training"""
    global training_state
    
    try:
        if training_state['status'] == 'training':
            training_state['status'] = 'stopped'
            training_state['error_message'] = 'Training stopped by user'
            return jsonify({'message': 'Training stopped successfully'})
        else:
            return jsonify({'message': 'No training in progress'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset-training-state', methods=['POST'])
def reset_training_state():
    """Reset training state completely"""
    global training_state

    training_state = {
        'status': 'idle',
        'progress': 0,
        'epoch': 0,
        'current_phase': 1,
        'history': [],
        'model': None,
        'results': None,
        'error_message': None,
        'current_batch': 0,
        'total_batches': 0,
        'batch_loss': 0.0,
        'batch_accuracy': 0.0,
        'iteration_details': []
    }

    return jsonify({'message': 'Training state reset successfully'})

# تعديل دالة start_training
@app.route('/api/train', methods=['POST'])
def start_training():
    """Start model training"""
    global training_state
    
    try:
        data = request.get_json()
        config = data.get('config', {})
        data_info = data.get('dataInfo', {})
        
        # تحسين الفحص
        if training_state['status'] == 'training':
            return jsonify({
                'error': 'Training already in progress',
                'current_epoch': training_state.get('epoch', 0),
                'progress': training_state.get('progress', 0),
                'suggestion': 'Please wait for current training to complete or stop it first'
            }), 400
        
        # باقي الكود كما هو...
        
        # Validate model configuration
        model_type = config.get('modelType', 'mlp')
        if model_type not in ['perceptron', 'mlp', 'cnn']:
            return jsonify({'error': f'Unsupported model type: {model_type}'}), 400
        
        # Additional validation for perceptron
        if model_type == 'perceptron':
            task_type = config.get('taskType', 'classification')
            if task_type != 'classification':
                return jsonify({
                    'error': 'Perceptron model only supports classification tasks',
                    'suggestion': 'Please select classification or use MLP for regression'
                }), 400
        
        # Reset training state
        training_state = {
            'status': 'training',
            'progress': 0,
            'epoch': 0,
            'history': [],
            'model': None,
            'results': None,
            'error_message': None
        }
        
        # Start training in a separate thread
        training_thread = threading.Thread(
            target=train_model_background,
            args=(config, data_info)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'message': f'{model_type.upper()} training started successfully'})
        
    except Exception as e:
        training_state['status'] = 'error'
        training_state['error_message'] = str(e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-progress', methods=['GET'])
def get_training_progress():
    """Get current training progress"""
    # Create a JSON-serializable copy of training_state
    response_data = {
        'status': training_state['status'],
        'progress': convert_to_json_serializable(training_state['progress']),
        'epoch': convert_to_json_serializable(training_state['epoch']),
        'current_phase': convert_to_json_serializable(training_state.get('current_phase', 1)),
        'history': convert_to_json_serializable(training_state['history']),
        'results': convert_to_json_serializable(training_state['results']),
        'error_message': training_state['error_message'],
        # إضافة تفاصيل الـ iterations
        'current_batch': convert_to_json_serializable(training_state.get('current_batch', 0)),
        'total_batches': convert_to_json_serializable(training_state.get('total_batches', 0)),
        'batch_loss': convert_to_json_serializable(training_state.get('batch_loss', 0.0)),
        'batch_accuracy': convert_to_json_serializable(training_state.get('batch_accuracy', 0.0)),
        'iteration_details': convert_to_json_serializable(training_state.get('iteration_details', []))
    }
    # Note: We exclude the 'model' key as it's not JSON serializable
    return jsonify(response_data)

@app.route('/api/export-model', methods=['GET'])
def export_model():
    """Export trained model"""
    try:
        export_format = request.args.get('format', 'tensorflow')
        
        if training_state['status'] != 'completed' or training_state['model'] is None:
            return jsonify({'error': 'No trained model available'}), 400
        
        exporter = ModelExporter()
        export_path = exporter.export_model(
            training_state['model'], 
            export_format,
            app.config['MODELS_FOLDER']
        )
        
        return send_file(
            export_path,
            as_attachment=True,
            download_name=f"trained_model.{export_format if export_format == 'onnx' else 'zip'}"
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview-image/<session_id>/<filename>', methods=['GET'])
def preview_image(session_id, filename):
    """Serve preview images from extracted ZIP files or single images"""
    try:
        # Load session data to get the extract path
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found'}), 404
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Get the extract path from preview data
        extract_path = session_data['preview'].get('extract_path')
        if not extract_path or not os.path.exists(extract_path):
            return jsonify({'error': 'Extract path not found'}), 404
        
        # Find the image file in the extracted directory or direct path
        image_path = None
        
        # Check if it's a direct file path (for single images)
        direct_path = os.path.join(extract_path, filename)
        if os.path.exists(direct_path):
            image_path = direct_path
        else:
            # Search recursively for ZIP extracted files
            for root, dirs, files in os.walk(extract_path):
                if filename in files:
                    image_path = os.path.join(root, filename)
                    break
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Return the image file
        return send_file(image_path)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/refresh-preview/<session_id>', methods=['GET'])
def refresh_preview(session_id):
    """Generate a new random preview for an existing session"""
    try:
        # Load session data
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        if not os.path.exists(session_file):
            return jsonify({'error': 'Session not found'}), 404
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        filepath = session_data['filepath']
        
        # Re-process the file to get new random samples
        processor = DataProcessor()
        new_preview_data = processor.process_file(filepath, session_id)
        
        # Update session data with new preview
        session_data['preview'] = new_preview_data
        new_preview_data['session_id'] = session_id
        new_preview_data['upload_timestamp'] = session_data['upload_timestamp']
        
        # Save updated session data
        with open(session_file, 'w') as f:
            json.dump(session_data, f)
        
        return jsonify({
            'message': 'Preview refreshed successfully',
            'preview': new_preview_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Model Testing Endpoints
@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get list of available trained models"""
    try:
        tester = ModelTester(app.config['MODELS_FOLDER'], app.config['UPLOAD_FOLDER'])
        models = tester.get_available_models()
        
        return jsonify({
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-model/<model_id>', methods=['POST'])
def load_model_for_testing(model_id):
    """Load a specific model for testing"""
    global current_model_tester
    
    try:
        tester = ModelTester(app.config['MODELS_FOLDER'], app.config['UPLOAD_FOLDER'])
        result = tester.load_model(model_id)
        
        if result['success']:
            current_model_tester = tester
            
            return jsonify({
                'message': f'Model {model_id} loaded successfully',
                'model_info': result['model_info'],
                'input_shape': result['input_shape'],
                'session_id': model_id  # أضف هذا
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/test-csv', methods=['POST'])
def test_csv_input():
    """Test model with CSV input data"""
    global current_model_tester
    
    try:
        data = request.get_json()
        csv_data = data.get('csvData')
        
        if not csv_data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        # تحقق بسيط من وجود النموذج
        if current_model_tester is None:
            return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
        
        result = current_model_tester.test_csv_input(csv_data)
        
        if result['success']:
            return jsonify({
                'message': 'Prediction completed successfully',
                'result': result
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/test-image', methods=['POST'])
def test_image_input():
    """Test model with image input"""
    global current_model_tester
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # تحقق بسيط من وجود النموذج
        if current_model_tester is None:
            return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
        
        # باقي الكود كما هو...
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"test_{timestamp}_{image_file.filename}"
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(temp_image_path)
        
        try:
            result = current_model_tester.test_image_input(temp_image_path)
            
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            if result['success']:
                return jsonify({
                    'message': 'Image prediction completed successfully',
                    'result': result
                })
            else:
                return jsonify({'error': result['error']}), 400
                
        except Exception as e:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/model-summary', methods=['GET'])
def get_model_summary():
    """Get summary of currently loaded model"""
    try:
        if 'current_model_tester' not in globals():
            return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
        
        result = current_model_tester.get_model_summary()
        
        if result['success']:
            return jsonify({
                'summary': result
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-data-info/<model_id>', methods=['GET'])
def get_training_data_info(model_id):
    """Get training data information for better CSV input guidance"""
    try:
        if 'current_model_tester' not in globals():
            return jsonify({'error': 'No model loaded. Please load a model first.'}), 400
        
        result = current_model_tester.get_training_data_info(model_id)
        
        if result['success']:
            return jsonify({
                'training_info': result
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/validate-model-config', methods=['POST'])
def validate_model_config():
    """Validate model configuration before training"""
    try:
        data = request.get_json()
        config = data.get('config', {})
        data_info = data.get('dataInfo', {})
        
        # Import model analyzer
        from app.model_builder import ModelAnalyzer
        
        model_type = config.get('modelType', 'mlp')
        
        # Analyze perceptron suitability
        if model_type == 'perceptron':
            analysis = ModelAnalyzer.analyze_perceptron_suitability(data_info)
            if not analysis['suitable']:
                return jsonify({
                    'valid': False,
                    'warnings': analysis['warnings'],
                    'recommendations': analysis['recommendations']
                })
        
        # Estimate model complexity
        complexity = ModelAnalyzer.estimate_model_complexity(config, data_info)
        
        return jsonify({
            'valid': True,
            'complexity': complexity,
            'warnings': [],
            'recommendations': []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info/<model_type>', methods=['GET'])
def get_model_info(model_type):
    """Get detailed information about a specific model type"""
    try:
        model_info = {
            'perceptron': {
                'name': 'Single Layer Perceptron',
                'description': 'A single artificial neuron that learns linear decision boundaries',
                'use_cases': ['Binary classification', 'Linearly separable data'],
                'advantages': ['Simple', 'Fast training', 'Interpretable', 'Low memory'],
                'limitations': ['Linear only', 'Binary classification', 'Cannot solve XOR'],
                'parameters': 'n_features + 1 (bias)',
                'complexity': 'O(n_features)',
                'invented': '1943 (McCulloch-Pitts), 1957 (Rosenblatt)',
                'best_for': 'Simple binary classification problems'
            },
            'mlp': {
                'name': 'Multi-Layer Perceptron',
                'description': 'Multiple layers of neurons that can learn complex non-linear patterns',
                'use_cases': ['Classification', 'Regression', 'Pattern recognition'],
                'advantages': ['Non-linear', 'Universal approximator', 'Flexible', 'Versatile'],
                'limitations': ['Black box', 'Prone to overfitting', 'Requires tuning'],
                'parameters': 'Depends on architecture',
                'complexity': 'O(layers × neurons × features)',
                'invented': '1986 (Backpropagation)',
                'best_for': 'Complex tabular data with non-linear relationships'
            },
            'cnn': {
                'name': 'Convolutional Neural Network',
                'description': 'Specialized for processing grid-like data such as images',
                'use_cases': ['Image classification', 'Computer vision', 'Spatial data'],
                'advantages': ['Translation invariant', 'Feature hierarchy', 'Parameter sharing'],
                'limitations': ['Large data needed', 'Computationally intensive'],
                'parameters': 'Depends on filters and layers',
                'complexity': 'O(filters × kernel_size × image_size)',
                'invented': '1989 (LeCun)',
                'best_for': 'Image and spatial data processing'
            }
        }
        
        if model_type not in model_info:
            return jsonify({'error': f'Unknown model type: {model_type}'}), 400
        
        return jsonify({
            'model_type': model_type,
            'info': model_info[model_type]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system-status', methods=['GET'])
def get_system_status():
    """Get system status and statistics"""
    try:
        # Count available models
        model_count = 0
        if os.path.exists(app.config['MODELS_FOLDER']):
            model_count = len([d for d in os.listdir(app.config['MODELS_FOLDER']) 
                             if os.path.isdir(os.path.join(app.config['MODELS_FOLDER'], d))])
        
        # Get current training status
        current_training = training_state.copy()
        current_training.pop('model', None)  # Remove non-serializable model
        
        # System info
        system_info = {
            'tensorflow_version': tf.__version__,
            'supported_models': ['perceptron', 'mlp', 'cnn'],
            'max_file_size': f"{MAX_CONTENT_LENGTH / (1024*1024*1024):.1f} GB",
            'upload_folder': UPLOAD_FOLDER,
            'models_folder': MODELS_FOLDER,
            'total_models': model_count,
            'current_training': current_training
        }
        
        return jsonify({
            'status': 'healthy',
            'system_info': system_info,
            'capabilities': {
                'file_types': ['CSV', 'Images (PNG, JPG, JPEG)', 'ZIP archives'],
                'model_types': ['Perceptron', 'Multi-Layer Perceptron', 'CNN'],
                'export_formats': ['TensorFlow', 'ONNX', 'Keras H5', 'TensorFlow Lite']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def train_model_background(config, data_info):
    """Background training function with enhanced memory management - FIXED"""
    global training_state
    
    try:
        # Load session data to get file path
        timestamp = None
        
        # Try to get timestamp from different sources
        if 'preview' in data_info and 'session_id' in data_info['preview']:
            timestamp = data_info['preview']['session_id']
        elif 'preview' in data_info:
            timestamp = data_info['preview'].get('upload_timestamp')
        
        # If still no timestamp, get the most recent session file
        if not timestamp:
            session_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('_session.json')]
            if session_files:
                session_files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
                timestamp = session_files[0].split('_session.json')[0]
            else:
                raise Exception("No session data found. Please upload data first.")
        
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_session.json")
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        filepath = session_data['filepath']
        
        # Initialize components with memory optimization
        processor = DataProcessor(max_memory_gb=4.0)  # Adjust based on available RAM
        builder = ModelBuilder()
        trainer = ModelTrainer()
        
        print(f"🧠 Initializing training with memory optimization")
        
        # Load and preprocess data with memory management
        try:
            training_data = processor.prepare_training_data(
                filepath, 
                config.get('validationSplit', 0.2),
                config.get('taskType', 'classification')
            )
            
            # ✅ الإصلاح: تحديد نوع التدريب بشكل صحيح
            if len(training_data) == 5:
                X_train, X_val, y_train, y_val, data_info_processed = training_data
                
                # 🔍 فحص أفضل لنوع البيانات
                if hasattr(X_train, '__len__') and hasattr(X_train, '__getitem__') and hasattr(X_train, 'on_epoch_end'):
                    using_generators = True
                    print(f"🔄 Using memory-efficient generator-based training")
                    print(f"📊 Generator steps: Train={len(X_train)}, Val={len(X_val) if X_val else 'None'}")
                elif hasattr(X_train, 'shape'):
                    using_generators = False
                    print(f"📊 Using standard array-based training")
                    print(f"🔢 Training shape: {X_train.shape}")
                    if y_train is not None:
                        print(f"🔢 Training labels shape: {y_train.shape}")
                else:
                    # ✅ التعامل مع الحالات الاستثنائية
                    using_generators = True
                    print(f"🔄 Detected generator-like object, using generator mode")
                    print(f"📊 Training data type: {type(X_train).__name__}")
                
            else:
                # ❌ هذا لا يجب أن يحدث مع الكود الجديد
                raise Exception("Unexpected training data format returned from processor")
                
        except MemoryError as me:
            error_msg = f"Memory allocation failed: {str(me)}\n\n"
            error_msg += "💡 Memory Optimization Tips:\n"
            error_msg += "• Try reducing the image dataset size\n"
            error_msg += "• Use smaller image resolution (current: 224x224)\n"
            error_msg += "• Close other memory-intensive applications\n"
            error_msg += "• Consider using a machine with more RAM\n"
            error_msg += "• Try processing the dataset in smaller batches"
            
            training_state['status'] = 'error'
            training_state['error_message'] = error_msg
            return
            
        except Exception as e:
            training_state['status'] = 'error'
            training_state['error_message'] = f"Data preparation failed: {str(e)}"
            print(f"❌ Data preparation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Memory optimization for model building
        model_type = config.get('modelType', 'mlp')
        
        # Adjust batch size based on dataset size and memory
        original_batch_size = config.get('batchSize', 32)
        if using_generators:
            # For generators, we can use smaller batch sizes safely
            recommended_batch_size = min(original_batch_size, 16)
            config['batchSize'] = recommended_batch_size
            print(f"🎛️  Adjusted batch size for generator mode: {recommended_batch_size}")
        
        # Handle binary classification labels for perceptron and MLP
        if model_type in ['perceptron', 'mlp'] and data_info_processed.get('num_classes', 1) == 2:
            if not using_generators and y_train is not None:
                y_train = np.array(y_train, dtype=np.int32)
                y_val = np.array(y_val, dtype=np.int32) if y_val is not None else None
        
        # Build model with memory considerations
        try:
            model = builder.build_model(config, data_info_processed)
            print(f"🏗️  Model architecture built successfully")
            
            # Print model summary for debugging
            total_params = model.count_params()
            print(f"📊 Model parameters: {total_params:,}")
            
            # Estimate memory usage
            estimated_memory_mb = total_params * 4 / (1024*1024) * 4  # Rough estimate
            print(f"💾 Estimated model memory: {estimated_memory_mb:.1f} MB")
            
            if estimated_memory_mb > 4000:  # > 4GB
                print("⚠️  Large model detected - consider reducing complexity")
            
        except Exception as e:
            training_state['status'] = 'error'
            training_state['error_message'] = f"Model building failed: {str(e)}"
            print(f"❌ Model building error: {str(e)}")
            return
        
        # Compile the model
        try:
            model = builder.compile_model(model, config, data_info_processed)
            print(f"⚙️  Model compiled successfully")
        except Exception as e:
            training_state['status'] = 'error'
            training_state['error_message'] = f"Model compilation failed: {str(e)}"
            print(f"❌ Model compilation error: {str(e)}")
            return
        
        # Custom callback to update training state
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self):
                super().__init__()
                self.epoch_count = 0
                self.start_time = time.time()
                self.batch_count = 0

            def on_epoch_begin(self, epoch, logs=None):
                print(f"🚀 Starting epoch {epoch + 1}/{config.get('epochs', 50)}")
                self.batch_count = 0
                # Force garbage collection at the start of each epoch
                import gc
                gc.collect()

            def on_train_begin(self, logs=None):
                # حساب عدد الـ batches الإجمالي
                if using_generators and hasattr(X_train, '__len__'):
                    training_state['total_batches'] = len(X_train)
                elif hasattr(X_train, 'shape'):
                    batch_size = config.get('batchSize', 32)
                    training_state['total_batches'] = int(np.ceil(len(X_train) / batch_size))

            def on_train_batch_end(self, batch, logs=None):
                logs = logs or {}
                self.batch_count += 1

                # تحديث معلومات الـ batch الحالي
                training_state['current_batch'] = self.batch_count
                training_state['batch_loss'] = float(logs.get('loss', 0))
                training_state['batch_accuracy'] = float(logs.get('accuracy', 0))

                # الاحتفاظ بآخر 100 iteration فقط لتجنب استهلاك الذاكرة
                iteration_entry = {
                    'epoch': self.epoch_count + 1,
                    'batch': self.batch_count,
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0))
                }

                if len(training_state['iteration_details']) >= 100:
                    training_state['iteration_details'].pop(0)
                training_state['iteration_details'].append(iteration_entry)

                # طباعة كل 10 batches
                if self.batch_count % 10 == 0:
                    total_batches = training_state.get('total_batches', '?')
                    print(f"   📦 Batch {self.batch_count}/{total_batches}: "
                          f"loss={logs.get('loss', 0):.4f}, "
                          f"acc={logs.get('accuracy', 0):.4f}")

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                self.epoch_count += 1
                training_state['epoch'] = self.epoch_count
                training_state['progress'] = (self.epoch_count / config.get('epochs', 50)) * 100

                # إعادة تعيين batch counter
                self.batch_count = 0
                training_state['current_batch'] = 0

                # Add to history
                history_entry = {
                    'epoch': self.epoch_count,
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_accuracy': float(logs.get('val_accuracy', 0))
                }
                training_state['history'].append(history_entry)

                # Calculate ETA
                elapsed = time.time() - self.start_time
                epochs_remaining = config.get('epochs', 50) - self.epoch_count
                eta_seconds = (elapsed / self.epoch_count) * epochs_remaining if self.epoch_count > 0 else 0

                print(f"✅ Epoch {self.epoch_count}: "
                      f"loss={logs.get('loss', 0):.4f}, "
                      f"accuracy={logs.get('accuracy', 0):.4f}, "
                      f"val_loss={logs.get('val_loss', 0):.4f}, "
                      f"val_accuracy={logs.get('val_accuracy', 0):.4f}, "
                      f"ETA: {eta_seconds/60:.1f}min")

                # Periodic memory cleanup
                if self.epoch_count % 5 == 0:
                    import gc
                    gc.collect()
                    print(f"🧹 Memory cleanup performed at epoch {self.epoch_count}")
        
        # Train model with memory management
        start_time = time.time()
        try:
            print(f"\n🚀 Starting {model_type.upper()} training with memory optimization...")
            
            # ✅ طباعة معلومات صحيحة حسب نوع التدريب
            if using_generators:
                print(f"🔄 Generator mode active:")
                print(f"   📊 Training steps per epoch: {len(X_train)}")
                print(f"   📊 Validation steps per epoch: {len(X_val) if X_val else 0}")
                print(f"   🎛️  Batch size: {config.get('batchSize', 32)}")
                if hasattr(data_info_processed, 'get'):
                    total_train = data_info_processed.get('train_samples', 'Unknown')
                    total_val = data_info_processed.get('val_samples', 'Unknown')
                    print(f"   📈 Total training samples: {total_train}")
                    print(f"   📉 Total validation samples: {total_val}")
            else:
                print(f"📊 Array mode active:")
                print(f"   🔢 Training data shape: {X_train.shape}")
                print(f"   🔢 Training labels shape: {y_train.shape}")
                if X_val is not None:
                    print(f"   🔢 Validation data shape: {X_val.shape}")
                    print(f"   🔢 Validation labels shape: {y_val.shape}")
            
            # Use the memory-optimized trainer
            history = trainer.train_model(
                model, X_train, y_train, X_val, y_val, 
                config, ProgressCallback()
            )
            end_time = time.time()
            
            # Force memory cleanup after training
            import gc
            gc.collect()
            
            # Calculate final results
            final_results = {
                'model_type': model_type,
                'final_train_loss': float(history.history['loss'][-1]) if 'loss' in history.history else None,
                'final_train_accuracy': float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None,
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None,
                'final_val_accuracy': float(history.history['val_accuracy'][-1]) if 'val_accuracy' in history.history else None,
                'training_time': f"{int(end_time - start_time)} seconds",
                'total_params': int(model.count_params()) if hasattr(model, 'count_params') else None,
                'trainable_params': int(model.count_params()) if hasattr(model, 'count_params') else None,
                'model_size': f"{model.count_params() * 4 / (1024*1024):.2f} MB" if hasattr(model, 'count_params') else None,
                'convergence_epoch': len(history.history['loss']) if 'loss' in history.history else None,
                'epochs_completed': len(history.history['loss']) if 'loss' in history.history else 0,
                'memory_optimized': using_generators,
                'batch_size_used': config.get('batchSize', 32)
            }
            
            # Add model-specific insights
            if model_type == 'perceptron':
                final_results['model_insights'] = {
                    'type': 'Single Layer Perceptron',
                    'decision_boundary': 'Linear',
                    'suitable_for': 'Linearly separable binary classification',
                    'limitations': 'Cannot solve XOR-like problems',
                    'parameters': 'Minimal - just weights and bias',
                    'training_speed': 'Very fast'
                }
            elif model_type == 'mlp':
                final_results['model_insights'] = {
                    'type': 'Multi-Layer Perceptron',
                    'decision_boundary': 'Non-linear',
                    'suitable_for': 'Complex classification and regression',
                    'advantages': 'Universal function approximator',
                    'hidden_layers': len([l for l in config.get('layers', []) if l.get('type') == 'dense']) - 1,
                    'complexity': 'Medium to High'
                }
            elif model_type == 'cnn':
                final_results['model_insights'] = {
                    'type': 'Convolutional Neural Network',
                    'decision_boundary': 'Non-linear with spatial awareness',
                    'suitable_for': 'Image classification and computer vision',
                    'advantages': 'Translation invariant feature extraction',
                    'specialization': 'Spatial data processing',
                    'memory_efficient': using_generators
                }
            
            # Update training state
            training_state['status'] = 'completed'
            training_state['model'] = model
            training_state['results'] = final_results
            training_state['progress'] = 100
            training_state['epoch'] = config.get('epochs', 50)
            
            print(f"✅ Training completed successfully!")
            print(f"🎯 Model type: {model_type}")
            print(f"📊 Final validation accuracy: {final_results.get('final_val_accuracy', 'N/A')}")
            print(f"⏱️  Training time: {final_results.get('training_time')}")
            print(f"🧠 Memory optimized: {using_generators}")
            
            # Save model with enhanced metadata
            model_path = os.path.join(app.config['MODELS_FOLDER'], f"model_{timestamp}")
            if hasattr(model, 'save'):
                os.makedirs(model_path, exist_ok=True)
                model.save(model_path)
                print(f"💾 Model saved to: {model_path}")
                
                # Save comprehensive training metadata
                metadata = {
                    'session_id': timestamp,
                    'model_type': model_type,
                    'data_type': data_info_processed.get('data_type'),
                    'task_type': data_info_processed.get('task_type'),
                    'input_shape': data_info_processed.get('input_shape'),
                    'num_classes': data_info_processed.get('num_classes'),
                    'class_names': data_info_processed.get('class_names'),
                    'feature_names': data_info_processed.get('feature_names'),
                    'target_name': data_info_processed.get('target_name'),
                    'final_accuracy': final_results.get('final_val_accuracy'),
                    'final_loss': final_results.get('final_val_loss'),
                    'training_config': config,
                    'training_results': final_results,
                    'created_at': time.time(),
                    'framework': 'tensorflow',
                    'model_architecture': model_type.upper(),
                    'version': '2.0',
                    'memory_optimized': using_generators,
                    'memory_info': {
                        'used_generators': using_generators,
                        'batch_size_used': config.get('batchSize', 32),
                        'estimated_model_size_mb': final_results.get('model_size', 'unknown')
                    }
                }
                
                metadata_file = os.path.join(model_path, 'training_metadata.json')
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Save preprocessing objects if available
                try:
                    if hasattr(processor, 'scaler') and processor.scaler:
                        import pickle
                        scaler_path = os.path.join(model_path, 'scaler.pkl')
                        with open(scaler_path, 'wb') as f:
                            pickle.dump(processor.scaler, f)
                        print("✅ Scaler saved successfully")
                    
                    if hasattr(processor, 'label_encoder') and processor.label_encoder:
                        import pickle
                        encoder_path = os.path.join(model_path, 'label_encoder.pkl')
                        with open(encoder_path, 'wb') as f:
                            pickle.dump(processor.label_encoder, f)
                        print("✅ Label encoder saved successfully")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save preprocessing objects: {e}")
                
        except MemoryError as training_error:
            end_time = time.time()
            training_state['status'] = 'error'
            error_message = f"Training failed due to memory limitations: {str(training_error)}\n\n"
            error_message += "🧠 Memory Optimization Suggestions:\n"
            
            if model_type == 'cnn':
                error_message += "• Reduce batch size (try 8 or 16)\n"
                error_message += "• Use smaller image resolution (try 128x128)\n"
                error_message += "• Reduce number of filters in CNN layers\n"
                error_message += "• Enable data generators for large datasets\n"
            elif model_type == 'mlp':
                error_message += "• Reduce hidden layer sizes\n"
                error_message += "• Reduce batch size\n"
                error_message += "• Use fewer hidden layers\n"
            
            error_message += "• Close other applications to free up RAM\n"
            error_message += "• Consider using a machine with more memory\n"
            error_message += "• Try training on a smaller subset of data first"
            
            training_state['error_message'] = error_message
            print(f"❌ Memory error: {str(training_error)}")
            
        except Exception as training_error:
            end_time = time.time()
            training_state['status'] = 'error'
            error_message = f"Training failed: {str(training_error)}\n\n"
            
            # Add model-specific error context
            if config.get('modelType') == 'perceptron':
                error_message += "🔧 Perceptron Training Tips:\n"
                error_message += "• Ensure data is linearly separable\n"
                error_message += "• Use binary classification only\n"
                error_message += "• Try lower learning rates (0.01-0.1)\n"
                error_message += "• Consider using MLP for complex patterns"
            elif config.get('modelType') == 'mlp':
                error_message += "🔧 MLP Training Tips:\n"
                error_message += "• Try adjusting the number of hidden layers\n"
                error_message += "• Experiment with different activation functions\n"
                error_message += "• Consider adding dropout for regularization\n"
                error_message += "• Reduce learning rate if training is unstable\n"
                error_message += "• Try smaller batch sizes for memory issues"
            elif config.get('modelType') == 'cnn':
                error_message += "🔧 CNN Training Tips:\n"
                error_message += "• Ensure sufficient training data\n"
                error_message += "• Try data augmentation for better generalization\n"
                error_message += "• Adjust filter sizes and counts\n"
                error_message += "• Consider transfer learning for small datasets\n"
                error_message += "• Use memory-efficient generators for large datasets"
            
            training_state['error_message'] = error_message
            print(f"❌ Training error details: {str(training_error)}")
            import traceback
            traceback.print_exc()
        
        # Final memory cleanup
        finally:
            import gc
            gc.collect()
            print("🧹 Final memory cleanup completed")
        
    except Exception as e:
        training_state['status'] = 'error'
        error_message = f"Setup error: {str(e)}"
        
        # Add specific error handling for common memory issues
        if "memory" in str(e).lower() or "allocation" in str(e).lower():
            error_message += "\n\n💡 This appears to be a memory issue. Try:\n"
            error_message += "• Reducing the dataset size\n"
            error_message += "• Using smaller images (resize to 128x128)\n"
            error_message += "• Reducing batch size to 8 or 16\n"
            error_message += "• Closing other applications\n"
            error_message += "• Using a machine with more RAM"
        
        training_state['error_message'] = error_message
        print(f"💥 Setup error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Final cleanup on error
        import gc
        gc.collect()
@app.route('/api/debug/training-state', methods=['GET'])
def get_debug_training_state():
    """Get full training state for debugging (development only)"""
    if app.debug:
        debug_state = training_state.copy()
        debug_state.pop('model', None)  # Remove non-serializable model
        return jsonify({
            'debug': True,
            'training_state': debug_state,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Debug endpoint only available in development mode'}), 403

@app.route('/api/debug/clear-training', methods=['POST'])
def clear_training_state():
    """Clear training state (development only)"""
    if app.debug:
        global training_state
        training_state = {
            'status': 'idle',
            'progress': 0,
            'epoch': 0,
            'history': [],
            'model': None,
            'results': None,
            'error_message': None
        }
        return jsonify({'message': 'Training state cleared', 'debug': True})
    else:
        return jsonify({'error': 'Debug endpoint only available in development mode'}), 403

@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare multiple trained models"""
    try:
        data = request.get_json()
        model_ids = data.get('model_ids', [])
        
        if len(model_ids) < 2:
            return jsonify({'error': 'At least 2 models required for comparison'}), 400
        
        tester = ModelTester(app.config['MODELS_FOLDER'], app.config['UPLOAD_FOLDER'])
        models_info = []
        
        for model_id in model_ids:
            try:
                model_path = os.path.join(app.config['MODELS_FOLDER'], model_id)
                if os.path.exists(model_path):
                    metadata_file = os.path.join(model_path, 'training_metadata.json')
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        model_info = {
                            'id': model_id,
                            'model_type': metadata.get('model_type', 'unknown'),
                            'final_accuracy': metadata.get('final_accuracy'),
                            'final_loss': metadata.get('final_loss'),
                            'training_time': metadata.get('training_results', {}).get('training_time'),
                            'total_params': metadata.get('training_results', {}).get('total_params'),
                            'data_type': metadata.get('data_type'),
                            'task_type': metadata.get('task_type'),
                            'created_at': metadata.get('created_at')
                        }
                        models_info.append(model_info)
            except Exception as e:
                print(f"Error loading model {model_id}: {e}")
                continue
        
        if len(models_info) < 2:
            return jsonify({'error': 'Could not load enough models for comparison'}), 400
        
        # Sort by accuracy (highest first)
        models_info.sort(key=lambda x: x.get('final_accuracy', 0) or 0, reverse=True)
        
        comparison = {
            'models': models_info,
            'best_accuracy': models_info[0] if models_info else None,
            'fastest_training': min(models_info, key=lambda x: float(x.get('training_time', '999').split()[0])) if models_info else None,
            'smallest_model': min(models_info, key=lambda x: x.get('total_params', float('inf'))) if models_info else None,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        return jsonify(comparison)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train-efficientnet', methods=['POST'])
def train_efficientnet():
    """
    تدريب نموذج EfficientNetV2
    """
    global training_state
    
    try:
        data = request.get_json()
        config = data.get('config', {})
        data_info = data.get('dataInfo', {})
        
        # فحص صحة الـ config
        if not config or not data_info:
            return jsonify({'error': 'Missing configuration or data info'}), 400
        
        # التحقق من أن البيانات صور
        # Frontend may send the data type under different keys (data_type, type, or preview.type)
        data_type = data_info.get('data_type') or data_info.get('type') or (data_info.get('preview') or {}).get('type')
        if data_type != 'images':
            return jsonify({'error': 'EfficientNetV2 requires image data'}), 400
        
        # طبع بيانات الطلب للمساعدة في التصحيح
        print(f"📥 train_efficientnet request received. session preview keys: {list((data_info.get('preview') or {}).keys()) if data_info.get('preview') else 'no preview'}")
        print(f"📥 config keys: {list(config.keys())}")

        # تحقق مبكر من مسار الاستخراج لكي نُرجع خطأ واضح بدلاً من فشل خفي في الخيط الخلفي
        session_extract = (data_info.get('preview') or {}).get('extract_path') or data_info.get('extract_path')
        if not session_extract or not os.path.exists(session_extract):
            print(f"❌ train_efficientnet - extract_path missing or not accessible: {session_extract}")
            return jsonify({'error': 'Extract path not found or inaccessible on server', 'provided_extract_path': session_extract}), 400

        # إعادة تعيين حالة التدريب
        training_state = {
            'status': 'training',
            'progress': 0,
            'epoch': 0,
            'history': [],
            'model': None,
            'results': None,
            'error_message': None,
            'current_phase': 1,
            'current_batch': 0,
            'total_batches': 0,
            'batch_loss': 0.0,
            'batch_accuracy': 0.0,
            'iteration_details': []
        }
        
        # بدء التدريب في thread منفصل
        try:
            training_thread = threading.Thread(
                target=train_efficientnet_background,
                args=(config, data_info)
            )
            training_thread.daemon = True
            training_thread.start()
        except Exception as e:
            print(f"❌ Failed to start training thread: {e}")
            training_state['status'] = 'error'
            training_state['error_message'] = str(e)
            return jsonify({'error': 'Failed to start training', 'details': str(e)}), 500
        
        return jsonify({'message': 'EfficientNetV2 training started successfully'})
    
    except Exception as e:
        training_state['status'] = 'error'
        training_state['error_message'] = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/api/debug/training-state', methods=['GET'])
def debug_training_state_local():
    """Return training_state for local debugging only (restricted to localhost)."""
    remote = request.remote_addr or 'unknown'
    # Restrict to localhost for safety
    if remote not in ('127.0.0.1', '::1'):
        return jsonify({'error': 'Forbidden - debug endpoint available only from localhost', 'remote_addr': remote}), 403

    debug_state = training_state.copy()
    debug_state.pop('model', None)
    return jsonify({'debug': True, 'training_state': debug_state, 'timestamp': datetime.now().isoformat()})


@app.route('/api/compute-gradcam', methods=['POST'])
def compute_gradcam():
    """
    حساب Grad-CAM للصور
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        if training_state['status'] != 'completed' or training_state['model'] is None:
            return jsonify({'error': 'No trained model available'}), 400
        
        # حفظ النموذج مؤقتاً
        temp_model_path = os.path.join(app.config['MODELS_FOLDER'], f'temp_model_{session_id}')
        os.makedirs(temp_model_path, exist_ok=True)
        training_state['model'].save(temp_model_path)
        # حاول كتابة ملف metadata.json بجانب النموذج المؤقت بحيث يمكن لخلفية Grad-CAM قراءته
        try:
            temp_metadata = {}
            # حاول الحصول على المعلومات من training_state.results إن وجدت
            if training_state.get('results'):
                res = training_state['results']
                temp_metadata['class_names'] = res.get('class_names', [])
                temp_metadata['num_classes'] = res.get('num_classes')
                # إذا وُجد model_path في النتائج، لا نحاول نسخه الآن
            # حاول الحصول على extract_path من ملف الجلسة المحفوظ
            session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
            if os.path.exists(session_file):
                with open(session_file, 'r') as sf:
                    session_data = json.load(sf)
                extract_path = session_data.get('preview', {}).get('extract_path')
                if extract_path:
                    temp_metadata['extract_path'] = extract_path

            # اكتب metadata.json إذا جمعنا أي شيء مفيد
            if temp_metadata:
                with open(os.path.join(temp_model_path, 'metadata.json'), 'w') as mf:
                    json.dump(temp_metadata, mf, indent=2)
                print(f"✅ Wrote temporary metadata for Grad-CAM at {os.path.join(temp_model_path, 'metadata.json')}")
        except Exception as e:
            print(f"⚠️ Could not write temp metadata for Grad-CAM: {e}")
        
        # بدء حساب Grad-CAM في thread
        gradcam_thread = threading.Thread(
            target=compute_gradcam_background,
            args=(session_id, temp_model_path)
        )
        gradcam_thread.daemon = True
        gradcam_thread.start()
        
        return jsonify({
            'message': 'Grad-CAM computation started',
            'status': 'computing'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/gradcam-status/<session_id>', methods=['GET'])
def get_gradcam_status(session_id):
    """
    الحصول على حالة Grad-CAM
    """
    try:
        gradcam_dir = os.path.join(app.config['MODELS_FOLDER'], f'gradcam_{session_id}')
        json_path = os.path.join(gradcam_dir, 'gradcam_data.json')
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                gradcam_data = json.load(f)
            
            return jsonify({
                'status': 'completed',
                'data': gradcam_data
            })
        else:
            return jsonify({
                'status': 'computing',
                'message': 'Still computing Grad-CAM...'
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ====================================================
# BACKGROUND TRAINING FUNCTION
# ====================================================

def train_efficientnet_background(config, data_info):
    """
    دالة خلفية لتدريب EfficientNetV2
    """
    global training_state
    
    try:
        print("🚀 Starting EfficientNetV2 training...")
        
        # حصول على معلومات الجلسة
        session_id = data_info.get('session_id') or data_info.get('preview', {}).get('session_id')
        if not session_id:
            raise Exception("Session ID not found")
        
        # تحميل البيانات
        extract_path = data_info.get('preview', {}).get('extract_path')
        if not extract_path or not os.path.exists(extract_path):
            raise Exception("Extract path not found")

        # إصلاح المسار: إذا كان extract_path يحتوي على مجلد واحد فقط، استخدم ذلك المجلد
        subdirs = [d for d in os.listdir(extract_path) if os.path.isdir(os.path.join(extract_path, d))]
        if len(subdirs) == 1:
            # تحقق إذا كان المجلد الفرعي يحتوي على مجلدات أخرى (الأصناف)
            potential_path = os.path.join(extract_path, subdirs[0])
            sub_subdirs = [d for d in os.listdir(potential_path) if os.path.isdir(os.path.join(potential_path, d))]
            if len(sub_subdirs) > 1:
                print(f"📁 Correcting extract_path from {extract_path} to {potential_path}")
                extract_path = potential_path

        # إعداد البيانات
        img_size = (256, 256)
        batch_size = config.get('batchSize', 16)
        validation_split = config.get('validationSplit', 0.2)

        # تحميل البيانات
        train_ds = tf.keras.utils.image_dataset_from_directory(
            extract_path,
            validation_split=validation_split,
            subset="training",
            seed=42,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.utils.image_dataset_from_directory(
            extract_path,
            validation_split=validation_split,
            subset="validation",
            seed=42,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        class_names = train_ds.class_names
        num_classes = len(class_names)

        print(f"📊 Classes: {class_names}")
        print(f"📁 Training samples: {len(train_ds)}")
        print(f"📁 Validation samples: {len(val_ds)}")

        # التحقق من عدد الأصناف
        if num_classes < 2:
            raise Exception(f"❌ Found only {num_classes} class. Please ensure your data has the correct folder structure with multiple classes (e.g., glioma, meningioma, pituitary, notumor)")

        # تحضير البيانات
        trainer = EfficientNetV2Trainer(img_size=img_size, num_classes=num_classes)
        train_ds_augmented = trainer.prepare_dataset(train_ds, augment=True)
        val_ds_prepared = trainer.prepare_dataset(val_ds, augment=False)
        
        # بناء النموذج
        print("\n🏗️  Building model...")
        model, base_model = trainer.build_model()
        model = trainer.compile_model(model, learning_rate=config.get('learningRate', 1e-3))

        # حذف ملفات .h5 القديمة لتجنب مشاكل الملفات المفتوحة
        for old_file in ['best_model_phase1.h5', 'best_model_phase2.h5']:
            if os.path.exists(old_file):
                try:
                    os.remove(old_file)
                    print(f"🗑️  Removed old checkpoint: {old_file}")
                except Exception as e:
                    print(f"⚠️  Could not remove {old_file}: {e}")

        # Callback مخصص لتحديث الحالة
        class TrainingStateCallback(tf.keras.callbacks.Callback):
            def __init__(self, phase, phase1_epochs=0, train_ds=None):
                self.phase = phase
                self.phase1_epochs = phase1_epochs  # عدد epochs المرحلة 1 (لحساب الإجمالي في المرحلة 2)
                self.epoch_count = 0
                self.start_time = time.time()
                self.batch_count = 0
                self.train_ds = train_ds

            def on_epoch_begin(self, epoch, logs=None):
                self.batch_count = 0

            def on_train_begin(self, logs=None):
                # حساب عدد الـ batches الإجمالي
                if self.train_ds and hasattr(self.train_ds, '__len__'):
                    training_state['total_batches'] = len(self.train_ds)

            def on_train_batch_end(self, batch, logs=None):
                logs = logs or {}
                self.batch_count += 1

                # تحديث معلومات الـ batch الحالي
                training_state['current_batch'] = self.batch_count
                training_state['batch_loss'] = float(logs.get('loss', 0))
                training_state['batch_accuracy'] = float(logs.get('accuracy', 0))

                # الاحتفاظ بآخر 100 iteration فقط
                iteration_entry = {
                    'epoch': self.epoch_count + 1,
                    'phase': self.phase,
                    'batch': self.batch_count,
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0))
                }

                if len(training_state['iteration_details']) >= 100:
                    training_state['iteration_details'].pop(0)
                training_state['iteration_details'].append(iteration_entry)

                # طباعة كل 10 batches
                if self.batch_count % 10 == 0:
                    total_batches = training_state.get('total_batches', '?')
                    print(f"   📦 Phase {self.phase} Batch {self.batch_count}/{total_batches}: "
                          f"loss={logs.get('loss', 0):.4f}, "
                          f"acc={logs.get('accuracy', 0):.4f}")

            def on_epoch_end(self, epoch, logs=None):
                self.epoch_count += 1
                logs = logs or {}

                # إعادة تعيين batch counter
                self.batch_count = 0
                training_state['current_batch'] = 0

                # حساب الـ epoch الإجمالي
                if self.phase == 1:
                    total_epoch = self.epoch_count
                else:  # phase 2
                    total_epoch = self.phase1_epochs + self.epoch_count

                training_state['epoch'] = total_epoch
                training_state['current_phase'] = self.phase

                # حساب التقدم
                phase1_total = config.get('epochs', 30)
                phase2_total = config.get('phase2_epochs', 25)
                total_epochs = phase1_total + phase2_total

                training_state['progress'] = (total_epoch / total_epochs) * 100

                # إضافة إلى history
                history_entry = {
                    'epoch': training_state['epoch'],
                    'phase': self.phase,
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_accuracy': float(logs.get('val_accuracy', 0)),
                    'top2_acc': float(logs.get('top2_acc', 0)),
                    'val_top2_acc': float(logs.get('val_top2_acc', 0))
                }
                training_state['history'].append(history_entry)

                print(f"✅ Phase {self.phase} Epoch {self.epoch_count}: "
                      f"loss={logs.get('loss', 0):.4f} acc={logs.get('accuracy', 0):.4f} "
                      f"val_loss={logs.get('val_loss', 0):.4f} val_acc={logs.get('val_accuracy', 0):.4f}")
        
        # المرحلة 1: تدريب الرأس
        print("\n" + "="*70)
        print("🔹 PHASE 1: Training classification head")
        print("="*70)
        
        epochs_phase1 = config.get('epochs', 30)
        callbacks_phase1 = trainer.create_callbacks("best_model_phase1")
        callbacks_phase1.append(TrainingStateCallback(phase=1, train_ds=train_ds_augmented))
        
        history1 = trainer.train_phase1(
            model,
            train_ds_augmented,
            val_ds_prepared,
            epochs=epochs_phase1
        )
        
        training_state['model'] = model
        
        # المرحلة 2: Fine-tuning
        print("\n" + "="*70)
        print("🔹 PHASE 2: Fine-tuning base model")
        print("="*70)
        
        epochs_phase2 = config.get('phase2_epochs', 25)
        callbacks_phase2 = trainer.create_callbacks("best_model_phase2")
        callbacks_phase2.append(TrainingStateCallback(phase=2, phase1_epochs=epochs_phase1, train_ds=train_ds_augmented))
        
        history2 = trainer.train_phase2(
            model,
            base_model,
            train_ds_augmented,
            val_ds_prepared,
            epochs=epochs_phase2
        )
        
        # دمج الـ histories
        combined_history = EfficientNetV2Trainer.combine_histories(history1, history2)

        # Update training_state with combined history
        training_state['history'] = []
        for i in range(len(combined_history['accuracy'])):
            training_state['history'].append({
                'epoch': i + 1,
                'loss': float(combined_history['loss'][i]),
                'accuracy': float(combined_history['accuracy'][i]),
                'val_loss': float(combined_history['val_loss'][i]),
                'val_accuracy': float(combined_history['val_accuracy'][i])
            })

        # تقييم النموذج
        print("\n📈 Evaluating model...")
        eval_results = trainer.evaluate_model(model, val_ds_prepared)
        
        # حفظ النتائج
        model_save_path = os.path.join(app.config['MODELS_FOLDER'], f'efficientnetv2_{session_id}')
        os.makedirs(model_save_path, exist_ok=True)
        model.save(os.path.join(model_save_path, 'model'))
        
        # حفظ metadata
        metadata = {
            'session_id': session_id,
            'model_type': 'efficientnetv2',
            'class_names': list(class_names),
            'num_classes': int(num_classes),
            'img_size': list(img_size),
            'extract_path': extract_path,
            'config': config,
            'eval_results': convert_to_json_serializable(eval_results),
            'created_at': time.time()
        }

        with open(os.path.join(model_save_path, 'metadata.json'), 'w') as f:
            json.dump(convert_to_json_serializable(metadata), f, indent=2)
        
        # الحصول على أفضل epoch
        best_val_acc_idx = int(np.argmax(combined_history['val_accuracy']))

        final_results = {
            'model_type': 'EfficientNetV2',
            'final_train_accuracy': float(combined_history['accuracy'][-1]),
            'final_val_accuracy': float(combined_history['val_accuracy'][-1]),
            'final_train_loss': float(combined_history['loss'][-1]),
            'final_val_loss': float(combined_history['val_loss'][-1]),
            'best_epoch': int(best_val_acc_idx + 1),
            'best_accuracy': float(max(combined_history['val_accuracy'])),
            'epochs_completed': int(len(combined_history['accuracy'])),
            'class_names': list(class_names),
            'num_classes': int(num_classes),
            'model_path': str(model_save_path),
            'has_gradcam': False  # سيتم تحديثه بعد حساب Grad-CAM
        }
        
        training_state['results'] = final_results
        training_state['status'] = 'completed'
        training_state['progress'] = 100
        training_state['epoch'] = int(len(combined_history['accuracy']))  # Total epochs completed

        print("✅ Training completed successfully!")
        print(f"📊 Final Validation Accuracy: {final_results['final_val_accuracy']:.2%}")
        print(f"🎯 Best Accuracy: {final_results['best_accuracy']:.2%} at epoch {final_results['best_epoch']}")
        
        # حفظ النموذج النهائي
        training_state['model'] = model
        
        # تنظيف الذاكرة
        import gc
        gc.collect()
        # بعد الإنهاء، شغّل حساب Grad-CAM في خلفية آليًا
        try:
            gradcam_thread = threading.Thread(
                target=compute_gradcam_background,
                args=(session_id, os.path.join(model_save_path, 'model'))
            )
            gradcam_thread.daemon = True
            gradcam_thread.start()
            print(f"🔮 Grad-CAM background thread started for session {session_id}")
        except Exception as e:
            print(f"⚠️ Failed to start Grad-CAM background thread: {e}")
        
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        training_state['status'] = 'error'
        training_state['error_message'] = str(e)


def compute_gradcam_background(session_id, model_path):
    """
    دالة خلفية لحساب Grad-CAM
    """
    try:
        print("🔮 Computing Grad-CAM...")
        
        # تحميل النموذج
        model = tf.keras.models.load_model(model_path)
        
        # تحميل metadata — حاول داخل model_path، ثم المجلد الأب، ثم مستوى أعلى كاحتياط
        metadata_path = os.path.join(model_path, 'metadata.json')
        if not os.path.exists(metadata_path):
            parent_dir = os.path.dirname(model_path)
            alt_metadata_path = os.path.join(parent_dir, 'metadata.json')
            if os.path.exists(alt_metadata_path):
                metadata_path = alt_metadata_path
            else:
                # حاول مستوى أعلى (مثلاً عندما يكون model_path endswith 'model')
                grandparent_dir = os.path.dirname(parent_dir)
                alt2 = os.path.join(grandparent_dir, 'metadata.json')
                if os.path.exists(alt2):
                    metadata_path = alt2
                else:
                    raise Exception("Metadata not found")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"🔎 Using metadata file: {metadata_path}")

        class_names = metadata.get('class_names', [])
        extract_path = metadata.get('extract_path')

        # إذا لم توجد أسماء الأصناف في metadata، حاول الحصول عليها من training_state أو من مجلد النماذج النهائي
        if not class_names:
            alt_names = None
            try:
                alt_names = training_state.get('results', {}).get('class_names')
            except Exception:
                alt_names = None

            if alt_names:
                class_names = alt_names
                print("ℹ️ Filled class_names from training_state.results")
            else:
                # حاول تحميل metadata من trained_models/efficientnetv2_{session_id}/metadata.json
                trained_meta = os.path.join(app.config['MODELS_FOLDER'], f'efficientnetv2_{session_id}', 'metadata.json')
                if os.path.exists(trained_meta):
                    try:
                        with open(trained_meta, 'r') as tfm:
                            tm = json.load(tfm)
                        class_names = tm.get('class_names', [])
                        if class_names:
                            print(f"ℹ️ Filled class_names from {trained_meta}")
                    except Exception:
                        pass
        
        # إذا لم يكن موجود، ابحث عن البيانات الأصلية
        session_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_session.json")
        if os.path.exists(session_file):
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            extract_path = session_data.get('preview', {}).get('extract_path')
        
        if not extract_path or not os.path.exists(extract_path):
            raise Exception("Extract path not found for Grad-CAM")
        
        # جمع صورة واحدة من كل فئة
        sample_images = []
        for class_name in class_names:
            class_path = os.path.join(extract_path, class_name)
            if os.path.exists(class_path):
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    sample_images.append(os.path.join(class_path, images[0]))

        # Fallback: إذا لم نعثر على أي صور من خلال بنية المجلد المتوقعة، قم ببحث تكراري داخل extract_path
        if not sample_images:
            print("⚠️ No images found in class subfolders — performing recursive search in extract_path...")
            found = {}
            for root, dirs, files in os.walk(extract_path):
                for fn in files:
                    if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
                        full = os.path.join(root, fn)
                        # محاولة مطابقة اسم الفئة من مسار الدليل الأب
                        parent = os.path.basename(os.path.dirname(full))
                        # normalise comparison
                        for cname in class_names:
                            if cname.lower() == parent.lower():
                                if cname not in found:
                                    found[cname] = full
                        # احتفظ ببعض الصور العامة إذا لم نجد مطابقة
                        if len(found) >= len(class_names):
                            break
                if len(found) >= len(class_names):
                    break

            # إذا وجدنا مطابقات، أضفها إلى sample_images
            if found:
                for cname in class_names:
                    if cname in found:
                        sample_images.append(found[cname])

            # كحل أخير، إذا ما زال فارغًا، التقط بعض الصور من أي مكان داخل extract_path
            if not sample_images:
                any_images = []
                for root, dirs, files in os.walk(extract_path):
                    for fn in files:
                        if fn.lower().endswith(('.jpg', '.png', '.jpeg')):
                            any_images.append(os.path.join(root, fn))
                    if any_images:
                        break
                # اختر حتى  min( len(class_names), len(any_images) ) صور
                if any_images:
                    limit = min(len(class_names) if class_names else 4, len(any_images))
                    sample_images = any_images[:limit]

        if not sample_images:
            raise Exception("No sample images found")
        
        print(f"📸 Found {len(sample_images)} sample images")
        
        # إنشاء Grad-CAM generator
        gradcam_gen = GradCAMGenerator(model, class_names, img_size=(256, 256))
        
        # إنشاء مجلد الإخراج
        output_dir = os.path.join(app.config['MODELS_FOLDER'], f'gradcam_{session_id}')
        os.makedirs(output_dir, exist_ok=True)
        
        # حساب Grad-CAM
        gradcam_data = gradcam_gen.generate_gradcam_samples(sample_images, output_dir)
        
        if gradcam_data:
            print(f"✅ Grad-CAM computed successfully! {gradcam_data['num_samples']} samples")
        else:
            print("⚠️ Grad-CAM computation had issues")
        
        # تحديث حالة التدريب
        if training_state['results']:
            training_state['results']['has_gradcam'] = True
            training_state['results']['gradcam_path'] = output_dir
        
    except Exception as e:
        print(f"❌ Grad-CAM error: {str(e)}")
        import traceback
        traceback.print_exc()









@app.route('/api/model-recommendations', methods=['POST'])
def get_model_recommendations():
    """Get model recommendations based on data characteristics"""
    try:
        data = request.get_json()
        data_info = data.get('dataInfo', {})
        
        from app.model_tester import ModelCompatibilityChecker
        
        recommendations = ModelCompatibilityChecker.get_model_recommendations(data_info)
        
        # Add specific guidance
        guidance = {
            'data_analysis': {
                'type': data_info.get('data_type', 'unknown'),
                'task': data_info.get('task_type', 'unknown'),
                'complexity': 'unknown'
            },
            'recommendations': recommendations,
            'tips': []
        }
        
        # Add tips based on data characteristics
        if data_info.get('data_type') == 'tabular':
            guidance['tips'].append("For tabular data, start with MLP for versatility")
            guidance['tips'].append("Consider Perceptron only if data is linearly separable")
        elif data_info.get('data_type') == 'images':
            guidance['tips'].append("CNN is strongly recommended for image data")
            guidance['tips'].append("Ensure sufficient training images (>100 per class)")
        
        return jsonify(guidance)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Neural Network Trainer API...")
    print("=" * 60)
    print("📊 Supported Model Types:")
    print("   🔹 Perceptron     - Single neuron for binary classification")
    print("   🔹 MLP           - Multi-layer for complex patterns")
    print("   🔹 CNN           - Convolutional for image processing")
    print("=" * 60)
    print("⚙️  System Configuration:")
    print(f"   💾 Upload folder: {UPLOAD_FOLDER}")
    print(f"   🤖 Models folder: {MODELS_FOLDER}")
    print(f"   📏 Max file size: {MAX_CONTENT_LENGTH / (1024*1024*1024):.1f} GB")
    print(f"   🔧 TensorFlow version: {tf.__version__}")
    print("=" * 60)
    print("🌐 API Endpoints:")
    print("   • /api/health           - Health check")
    print("   • /api/upload           - File upload")
    print("   • /api/train            - Start training")
    print("   • /api/training-progress - Training status")
    print("   • /api/models           - Available models")
    print("   • /api/test-csv         - Test with CSV")
    print("   • /api/test-image       - Test with images")
    print("   • /api/model-info/<type> - Model information")
    print("   • /api/system-status    - System statistics")
    print("=" * 60)
    print("🚀 Server starting on http://0.0.0.0:5000")
    print("📚 Ready to train Perceptrons, MLPs, and CNNs!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)