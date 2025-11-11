import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class EfficientNetV2Trainer:
    """
    🧠 EfficientNetV2 Trainer for Brain Tumor Classification
    """
    
    def __init__(self, img_size=(256, 256), num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
    
    def create_data_augmentation(self):
        """
        إنشاء Data Augmentation
        """
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2)
        ])
    
    def prepare_dataset(self, dataset, augment=False):
        """
        تحضير dataset مع augmentation
        """
        data_augmentation = self.create_data_augmentation() if augment else None
        
        def prepare_batch(images, labels):
            if augment and data_augmentation:
                images = data_augmentation(images, training=True)
            return images, labels
        
        if augment:
            dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    
    def build_model(self):
        """
        بناء نموذج EfficientNetV2B0 مع رأس تصنيف

        ملاحظة مهمة: لا نستخدم preprocessing داخل النموذج لتمكين Grad-CAM.
        Preprocessing يتم تطبيقه في preprocessing pipeline خارجياً.
        """
        logger.info("Building EfficientNetV2B0 model...")

        # تحميل Base Model - B0 أصغر بكثير من L
        base_model = tf.keras.applications.EfficientNetV2B0(
            weights="imagenet",
            include_top=False,
            input_shape=self.img_size + (3,)
        )

        # تجميد الطبقات في البداية
        base_model.trainable = False

        # بناء النموذج الكامل
        inputs = keras.Input(shape=self.img_size + (3,))

        # ❌ لا نستخدم preprocess_input هنا!
        # ✅ سيتم التطبيق في data pipeline
        # x = tf.keras.applications.efficientnet_v2.preprocess_input(inputs)

        # Rescaling manual (بدلاً من preprocess_input)
        # EfficientNet preprocessing: scale to [-1, 1]
        x = layers.Rescaling(scale=1./127.5, offset=-1)(inputs)

        # Base model
        x = base_model(x, training=False)

        # Global Average Pooling
        x = layers.GlobalAveragePooling2D()(x)

        # رأس التصنيف
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

        logger.info("Model built successfully (Grad-CAM compatible)!")
        return model, base_model
    
    def compile_model(self, model, learning_rate=1e-3):
        """
        ضبط النموذج
        """
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
        )
        logger.info(f"Model compiled with learning_rate={learning_rate}")
        return model
    
    def create_callbacks(self, model_name="best_model"):
        """
        إنشاء callbacks للتدريب
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5,
                restore_best_weights=True,
                monitor='val_accuracy',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                monitor='val_loss',
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                f'{model_name}.h5',
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1,
                save_format='h5'
            )
        ]
        return callbacks
    
    def train_phase1(self, model, train_ds, val_ds, epochs=30):
        """
        المرحلة الأولى: تدريب الرأس فقط
        """
        logger.info("="*70)
        logger.info("🔹 PHASE 1: Training classification head (base frozen)...")
        logger.info("="*70)
        
        callbacks = self.create_callbacks("best_model_phase1")
        
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Phase 1 completed!")
        return history1
    
    def train_phase2(self, model, base_model, train_ds, val_ds, epochs=25, fine_tune_at=None):
        """
        المرحلة الثانية: Fine-tuning آخر طبقات Base Model
        """
        logger.info("="*70)
        logger.info("🔹 PHASE 2: Fine-tuning last layers...")
        logger.info("="*70)
        
        # فك التجميد
        base_model.trainable = True
        
        # تجميد الطبقات الأولى فقط
        if fine_tune_at is None:
            fine_tune_at = len(base_model.layers) - 30  # فك آخر 30 طبقة (B0 أصغر من L)
        
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        logger.info(f"Unfreezing {len(base_model.layers) - fine_tune_at} layers")
        
        # إعادة ضبط بمعدل تعلم أقل
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
        )
        
        callbacks = self.create_callbacks("best_model_phase2")
        
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Phase 2 completed!")
        return history2
    
    def evaluate_model(self, model, val_ds):
        """
        تقييم النموذج على بيانات التحقق
        """
        logger.info("Evaluating model...")
        results = model.evaluate(val_ds, verbose=1)
        
        return {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'top2_accuracy': float(results[2]) if len(results) > 2 else None
        }
    
    @staticmethod
    def combine_histories(history1, history2):
        """
        دمج histories من المرحلتين
        """
        combined = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'top2_acc': history1.history.get('top2_acc', []) + history2.history.get('top2_acc', []),
            'val_top2_acc': history1.history.get('val_top2_acc', []) + history2.history.get('val_top2_acc', []),
        }
        return combined