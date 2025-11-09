import pandas as pd
import numpy as np
import os
import zipfile
from PIL import Image
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import gc

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: OpenCV not available. Image processing will be limited.")
# أضف هذا الكود في app/data_processor.py بعد الـ imports

class ImageDataLoaderEfficientNet:
    """
    محمل بيانات محسّن لـ EfficientNetV2
    """

    def __init__(self, img_size=(256, 256), batch_size=16):
        self.img_size = img_size
        self.batch_size = batch_size

    def clean_image_directory(self, data_dir):
        """
        تنظيف المجلد من الصور التالفة أو غير المدعومة
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
        removed_count = 0

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # تخطي الملفات المخفية
                if file.startswith('.'):
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"Removed hidden file: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
                    continue

                # حذف الملفات غير المدعومة
                if file_ext not in valid_extensions:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"Removed unsupported file: {file_path}")
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
                    continue

                # التحقق من صحة الصورة بشكل شامل
                try:
                    # فتح الصورة
                    img = Image.open(file_path)
                    # تحويل الصورة لـ RGB (يكشف الأخطاء المخفية)
                    img = img.convert('RGB')
                    # محاولة تحميل البيانات فعلياً
                    img.load()
                    # التحقق من الأبعاد
                    if img.size[0] < 10 or img.size[1] < 10:
                        raise Exception("Image too small")
                    img.close()
                except Exception as e:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                        print(f"Removed corrupted image: {file_path} - {e}")
                    except Exception as remove_error:
                        print(f"Error removing {file_path}: {remove_error}")

        if removed_count > 0:
            print(f"✅ Cleaned {removed_count} invalid/corrupted files")
        else:
            print(f"✅ No corrupted files found")
        return removed_count

    def load_image_dataset(self, data_dir, validation_split=0.2, seed=42):
        """
        تحميل مجموعة صور من مجلد
        """
        try:
            # تنظيف المجلد من الصور التالفة
            print("🧹 Cleaning image directory...")
            self.clean_image_directory(data_dir)

            # تحميل مجموعة التدريب
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset="training",
                seed=seed,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode='categorical'
            )
            
            # تحميل مجموعة التحقق
            val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                subset="validation",
                seed=seed,
                image_size=self.img_size,
                batch_size=self.batch_size,
                label_mode='categorical'
            )
            
            class_names = train_ds.class_names
            
            return train_ds, val_ds, class_names
        
        except Exception as e:
            raise Exception(f"Error loading image dataset: {str(e)}")
    
    def apply_augmentation(self, dataset):
        """
        تطبيق Data Augmentation
        """
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.2),
            tf.keras.layers.RandomBrightness(0.2)
        ])
        
        def augment_batch(images, labels):
            return data_augmentation(images, training=True), labels
        
        dataset = dataset.map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def prepare_dataset(self, dataset, augment=False):
        """
        تحضير dataset نهائي
        """
        if augment:
            dataset = self.apply_augmentation(dataset)
        else:
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset

# ============================================================
# تعديل في دالة process_file الموجودة
# ============================================================

# إضافة هذا الشرط في البداية:
def process_file(self, filepath, session_id=None):
    """
    معالجة الملف المرفوع
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    if file_extension == '.csv':
        return self._process_csv(filepath)
    elif file_extension in ['.zip']:
        return self._process_image_zip(filepath, session_id)
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        return self._process_single_image(filepath, session_id)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# ============================================================
# إصلاح _process_image_zip
# ============================================================

# تأكد من أن _process_image_zip يحفظ المسار:
def _process_image_zip(self, filepath, session_id=None):
    """معالجة ZIP يحتوي على صور - محسّن"""
    try:
        extract_path = filepath.replace('.zip', '_extracted')
        os.makedirs(extract_path, exist_ok=True)
        
        # فك الضغط
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # البحث عن الصور
        image_files = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            raise Exception("No image files found in ZIP archive")
        
        # الكشف عن الفئات
        classes = set()
        for img_path in image_files:
            parent_dir = os.path.basename(os.path.dirname(img_path))
            if parent_dir != os.path.basename(extract_path):
                classes.add(parent_dir)
        
        if session_id is None:
            session_id = os.path.basename(filepath).split('_')[0]
        
        # اختيار عينات عشوائية
        sample_images = []
        max_samples = min(8, len(image_files))
        
        if classes and len(classes) > 1:
            images_by_class = {}
            for img_path in image_files:
                class_name = os.path.basename(os.path.dirname(img_path))
                if class_name != os.path.basename(extract_path):
                    if class_name not in images_by_class:
                        images_by_class[class_name] = []
                    images_by_class[class_name].append(img_path)
            
            images_per_class = max(1, max_samples // len(classes))
            for class_name, class_images in images_by_class.items():
                sample_size = min(images_per_class, len(class_images))
                selected = random.sample(class_images, sample_size)
                
                for img_path in selected:
                    if len(sample_images) >= max_samples:
                        break
                    filename = os.path.basename(img_path)
                    sample_images.append({
                        'url': f'/api/preview-image/{session_id}/{filename}',
                        'label': class_name
                    })
        else:
            selected_paths = random.sample(image_files, max_samples)
            for i, img_path in enumerate(selected_paths):
                filename = os.path.basename(img_path)
                class_label = os.path.basename(os.path.dirname(img_path)) if classes else f'Image {i+1}'
                sample_images.append({
                    'url': f'/api/preview-image/{session_id}/{filename}',
                    'label': class_label
                })
        
        preview_data = {
            'type': 'images',
            'count': len(image_files),
            'classes': len(classes) if classes else None,
            'class_names': list(classes) if classes else None,
            'samples': sample_images,
            'extract_path': extract_path,  # ✅ حفظ المسار
            'use_generator': len(image_files) > self.max_images_in_memory,
            'memory_info': {
                'max_images_in_memory': self.max_images_in_memory,
                'estimated_memory_gb': len(image_files) * 256 * 256 * 3 * 4 / (1024**3),
                'will_use_generator': len(image_files) > self.max_images_in_memory
            }
        }
        
        return preview_data
    
    except Exception as e:
        raise Exception(f"Error processing image ZIP: {str(e)}")
class ImageDataGenerator(Sequence):
    """Memory-efficient image data generator for large datasets"""
    
    def __init__(self, image_paths, labels, batch_size=32, target_size=(224, 224), 
                 shuffle=True, augmentation=None, validation=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.validation = validation
        self.indices = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        # Load and process images
        batch_images = []
        for path in batch_paths:
            try:
                img = self._load_and_preprocess_image(path)
                batch_images.append(img)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                # Create a blank image as fallback
                blank_img = np.zeros((*self.target_size, 3), dtype=np.float32)
                batch_images.append(blank_img)
        
        X = np.array(batch_images, dtype=np.float32)
        y = np.array(batch_labels, dtype=np.int32)
        
        # Apply augmentation for training data
        if self.augmentation and not self.validation:
            X = self._apply_augmentation(X)
        
        return X, y
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image efficiently"""
        try:
            # Try OpenCV first (faster)
            if cv2 is not None:
                img = cv2.imread(image_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.target_size)
                    img = img.astype(np.float32) / 255.0
                    return img
            
            # Fallback to PIL
            with Image.open(image_path) as pil_img:
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                pil_img = pil_img.resize(self.target_size, Image.Resampling.LANCZOS)
                img = np.array(pil_img, dtype=np.float32) / 255.0
                return img
                
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.float32)
    
    def _apply_augmentation(self, batch_images):
        """Apply simple augmentation to batch"""
        if self.augmentation is None:
            return batch_images
        
        # Simple random horizontal flip
        for i in range(len(batch_images)):
            if np.random.random() > 0.5:
                batch_images[i] = np.fliplr(batch_images[i])
        
        return batch_images
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

class DataProcessor:
    def __init__(self, max_memory_gb=4.0):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.max_memory_gb = max_memory_gb
        self.max_images_in_memory = self._calculate_max_images()
    
    def _calculate_max_images(self):
        """Calculate maximum number of images that can fit in memory"""
        # Assume 224x224x3 float32 images (4 bytes per pixel)
        bytes_per_image = 224 * 224 * 3 * 4
        max_bytes = self.max_memory_gb * 1024**3
        max_images = int(max_bytes * 0.7 / bytes_per_image)  # Use 70% of available memory
        return max(100, max_images)  # At least 100 images
    
    def process_file(self, filepath, session_id=None):
        """Process uploaded file and return preview data"""
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.csv':
            return self._process_csv(filepath)
        elif file_extension in ['.zip']:
            return self._process_image_zip(filepath, session_id)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            return self._process_single_image(filepath, session_id)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _process_csv(self, filepath):
        """Process CSV file"""
        try:
            # Read CSV with error handling
            df = pd.read_csv(filepath)
            
            # Basic info
            rows, columns = df.shape
            
            # Sample data for preview
            sample_data = df.head(10).fillna('').to_dict('records')
            
            # Detect data types and potential target column
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            preview_data = {
                'type': 'csv',
                'rows': rows,
                'columns': columns,
                'sample': sample_data,
                'numeric_columns': numeric_columns,
                'categorical_columns': categorical_columns,
                'column_names': df.columns.tolist()
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing CSV file: {str(e)}")
    
    def _process_image_zip(self, filepath, session_id=None):
        """Process ZIP file containing images with memory optimization"""
        try:
            extract_path = filepath.replace('.zip', '_extracted')
            os.makedirs(extract_path, exist_ok=True)
            
            # Extract ZIP file
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find image files
            image_files = []
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
            
            for root, dirs, files in os.walk(extract_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise Exception("No image files found in ZIP archive")
            
            # Memory check
            if len(image_files) > self.max_images_in_memory:
                print(f"⚠️  Large dataset detected: {len(image_files)} images")
                print(f"🧠 Will use generator-based training to manage memory")
            
            # Detect class structure
            classes = set()
            for img_path in image_files:
                parent_dir = os.path.basename(os.path.dirname(img_path))
                if parent_dir != os.path.basename(extract_path):
                    classes.add(parent_dir)
            
            # Extract session_id from filepath if not provided
            if session_id is None:
                session_id = os.path.basename(filepath).split('_')[0]
            
            # Sample images for preview (load only a few for memory efficiency)
            sample_images = []
            max_samples = min(8, len(image_files))
            
            if classes and len(classes) > 1:
                # Balanced sampling from classes
                images_by_class = {}
                for img_path in image_files:
                    class_name = os.path.basename(os.path.dirname(img_path))
                    if class_name != os.path.basename(extract_path):
                        if class_name not in images_by_class:
                            images_by_class[class_name] = []
                        images_by_class[class_name].append(img_path)
                
                # Sample from each class
                images_per_class = max(1, max_samples // len(classes))
                for class_name, class_images in images_by_class.items():
                    sample_size = min(images_per_class, len(class_images))
                    selected = random.sample(class_images, sample_size)
                    
                    for img_path in selected:
                        if len(sample_images) >= max_samples:
                            break
                        filename = os.path.basename(img_path)
                        sample_images.append({
                            'url': f'/api/preview-image/{session_id}/{filename}',
                            'label': class_name
                        })
            else:
                # Random sampling
                selected_paths = random.sample(image_files, max_samples)
                for i, img_path in enumerate(selected_paths):
                    filename = os.path.basename(img_path)
                    class_label = os.path.basename(os.path.dirname(img_path)) if classes else f'Image {i+1}'
                    sample_images.append({
                        'url': f'/api/preview-image/{session_id}/{filename}',
                        'label': class_label
                    })
            
            preview_data = {
                'type': 'images',
                'count': len(image_files),
                'classes': len(classes) if classes else None,
                'class_names': list(classes) if classes else None,
                'samples': sample_images,
                'extract_path': extract_path,
                'use_generator': len(image_files) > self.max_images_in_memory,
                'memory_info': {
                    'max_images_in_memory': self.max_images_in_memory,
                    'estimated_memory_gb': len(image_files) * 224 * 224 * 3 * 4 / (1024**3),
                    'will_use_generator': len(image_files) > self.max_images_in_memory
                }
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing image ZIP: {str(e)}")
    
    def _process_single_image(self, filepath, session_id=None):
        """Process single image file"""
        try:
            img = Image.open(filepath)
            width, height = img.size
            
            if session_id is None:
                session_id = os.path.basename(filepath).split('_')[0]
            
            preview_data = {
                'type': 'images',
                'count': 1,
                'classes': None,
                'samples': [{
                    'url': f'/api/preview-image/{session_id}/{os.path.basename(filepath)}',
                    'label': 'Single Image'
                }],
                'dimensions': f"{width}x{height}",
                'mode': img.mode,
                'extract_path': os.path.dirname(filepath),
                'use_generator': False
            }
            
            return preview_data
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def prepare_training_data(self, filepath, validation_split=0.2, task_type='classification'):
        """Prepare data for training with memory optimization"""
        preview_data = self.process_file(filepath)
        
        if preview_data['type'] == 'csv':
            return self._prepare_csv_data(filepath, validation_split, task_type)
        elif preview_data['type'] == 'images':
            if preview_data.get('use_generator', False):
                return self._prepare_image_data_generator(preview_data, validation_split, task_type)
            else:
                return self._prepare_image_data_memory(preview_data, validation_split, task_type)
        else:
            raise ValueError("Unsupported data type for training")
    
    def _prepare_csv_data(self, filepath, validation_split, task_type):
        """Prepare CSV data for training"""
        df = pd.read_csv(filepath)
        
        feature_columns = df.columns[:-1]
        target_column = df.columns[-1]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X.loc[:, col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle labels
        original_class_names = None
        if task_type == 'classification':
            if y.dtype == 'object':
                original_class_names = sorted(y.unique().tolist())
                y = self.label_encoder.fit_transform(y)
            else:
                original_class_names = sorted(y.unique().tolist())
            
            unique_classes = np.unique(y)
            num_classes = len(unique_classes)
            
            y_mapped = np.zeros_like(y)
            for new_idx, old_class in enumerate(unique_classes):
                y_mapped[y == old_class] = new_idx
            y = y_mapped.astype(np.int32)
        else:
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().sum() > 0:
                valid_indices = ~y.isna()
                X = X[valid_indices]
                y = y[valid_indices]
            y = y.values.astype(np.float32)
            num_classes = 1
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, 
            stratify=y if task_type == 'classification' else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        data_info = {
            'input_shape': X_train_scaled.shape[1],
            'num_classes': num_classes,
            'task_type': task_type,
            'data_type': 'tabular',
            'class_names': original_class_names,
            'feature_names': feature_columns.tolist(),
            'target_name': target_column
        }
        
        return X_train_scaled, X_val_scaled, y_train, y_val, data_info
    
    def _prepare_image_data_generator(self, preview_data, validation_split, task_type):
        """Prepare image data using generators for large datasets"""
        extract_path = preview_data['extract_path']
        
        # Collect all image paths and labels
        image_paths = []
        labels = []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        if preview_data.get('class_names'):
            label_to_idx = {label: idx for idx, label in enumerate(preview_data['class_names'])}
            
            for root, dirs, files in os.walk(extract_path):
                class_name = os.path.basename(root)
                if class_name in label_to_idx:
                    for file in files:
                        if os.path.splitext(file)[1].lower() in image_extensions:
                            img_path = os.path.join(root, file)
                            image_paths.append(img_path)
                            labels.append(label_to_idx[class_name])
        
        if not image_paths:
            raise Exception("No valid images found for training")
        
        # Convert to numpy arrays
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        # Split indices
        indices = np.arange(len(image_paths))
        train_idx, val_idx = train_test_split(
            indices, test_size=validation_split, random_state=42,
            stratify=labels if task_type == 'classification' else None
        )
        
        # Create generators
        train_generator = ImageDataGenerator(
            image_paths[train_idx], labels[train_idx],
            batch_size=32, shuffle=True, validation=False
        )
        
        val_generator = ImageDataGenerator(
            image_paths[val_idx], labels[val_idx],
            batch_size=32, shuffle=False, validation=True
        )
        
        data_info = {
            'input_shape': (224, 224, 3),
            'num_classes': len(np.unique(labels)) if task_type == 'classification' else 1,
            'task_type': task_type,
            'data_type': 'images',
            'class_names': preview_data.get('class_names'),
            'use_generator': True,
            'total_samples': len(image_paths),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        
        return train_generator, val_generator, None, None, data_info
    
    def _prepare_image_data_memory(self, preview_data, validation_split, task_type):
        """Prepare image data loading all into memory (for smaller datasets)"""
        extract_path = preview_data['extract_path']
        
        # Load images and labels
        images = []
        labels = []
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        target_size = (224, 224)
        
        if preview_data.get('class_names'):
            label_to_idx = {label: idx for idx, label in enumerate(preview_data['class_names'])}
            
            for root, dirs, files in os.walk(extract_path):
                class_name = os.path.basename(root)
                if class_name in label_to_idx:
                    for file in files:
                        if os.path.splitext(file)[1].lower() in image_extensions:
                            img_path = os.path.join(root, file)
                            try:
                                if cv2 is not None:
                                    img = cv2.imread(img_path)
                                    img = cv2.resize(img, target_size)
                                    img = img.astype(np.float32) / 255.0
                                else:
                                    img = Image.open(img_path)
                                    img = img.resize(target_size)
                                    img = np.array(img, dtype=np.float32) / 255.0
                                    if len(img.shape) == 2:
                                        img = np.stack([img] * 3, axis=-1)
                                
                                images.append(img)
                                labels.append(label_to_idx[class_name])
                                
                            except Exception as e:
                                print(f"Error processing image {img_path}: {e}")
                                continue
        
        if not images:
            raise Exception("No valid images found for training")
        
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        # Force garbage collection
        del images
        gc.collect()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        data_info = {
            'input_shape': X_train.shape[1:],
            'num_classes': len(np.unique(y)) if task_type == 'classification' else 1,
            'task_type': task_type,
            'data_type': 'images',
            'class_names': preview_data.get('class_names'),
            'use_generator': False
        }
        
        return X_train, X_val, y_train, y_val, data_info