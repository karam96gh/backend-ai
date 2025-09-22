import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import gc

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
    
    def train_model(self, model, X_train, y_train, X_val, y_val, config, progress_callback=None):
        """Train the model with given data and configuration - supports both arrays and generators"""
        
        # Detect if we're using generators
        using_generators = hasattr(X_train, '__len__') and hasattr(X_train, '__getitem__') and hasattr(X_train, 'on_epoch_end')
        
        if using_generators:
            return self._train_with_generators(model, X_train, X_val, config, progress_callback)
        else:
            return self._train_with_arrays(model, X_train, y_train, X_val, y_val, config, progress_callback)
    
    def _train_with_arrays(self, model, X_train, y_train, X_val, y_val, config, progress_callback):
        """Traditional training with arrays in memory"""
        
        # Get training parameters
        epochs = config.get('epochs', 50)
        batch_size = config.get('batchSize', 32)
        validation_split = config.get('validationSplit', 0.2) if X_val is None else None
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks(progress_callback)
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_data = None
            validation_split = config.get('validationSplit', 0.2)
        
        # Train the model
        try:
            print(f"🚀 Starting training with arrays in memory")
            print(f"📊 Training shape: {X_train.shape}, Validation shape: {X_val.shape if X_val is not None else 'None'}")
            
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.model = model
            self.history = history
            
            # Clean up memory
            gc.collect()
            
            return history
            
        except Exception as e:
            raise Exception(f"Array-based training failed: {str(e)}")
    
    def _train_with_generators(self, model, train_generator, val_generator, config, progress_callback):
        """Memory-efficient training with data generators"""
        
        # Get training parameters
        epochs = config.get('epochs', 50)
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks(progress_callback)
        
        # Add memory management callback
        memory_callback = MemoryManagementCallback()
        callbacks.append(memory_callback)
        
        try:
            print(f"🚀 Starting generator-based training for memory efficiency")
            print(f"📊 Training samples: {getattr(train_generator, 'total_samples', 'Unknown')}")
            print(f"📊 Validation samples: {getattr(val_generator, 'total_samples', 'Unknown')}")
            print(f"🔄 Steps per epoch: {len(train_generator)}")
            
            history = model.fit(
                train_generator,
                epochs=epochs,
                validation_data=val_generator,
                callbacks=callbacks,
                verbose=0,
                use_multiprocessing=False,  # Avoid multiprocessing issues
                workers=1  # Single worker for stability
            )
            
            self.model = model
            self.history = history
            
            # Force memory cleanup
            gc.collect()
            
            return history
            
        except Exception as e:
            raise Exception(f"Generator-based training failed: {str(e)}")
    
    def _prepare_callbacks(self, progress_callback):
        """Prepare training callbacks with memory optimization"""
        callbacks = []
        
        # Early stopping with more patience for large datasets
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience for large datasets
            restore_best_weights=True,
            verbose=0
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint - save best model
        checkpoint = keras.callbacks.ModelCheckpoint(
            'temp_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0,
            save_weights_only=False
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_reducer = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,  # Increased patience
            min_lr=1e-8,
            verbose=0
        )
        callbacks.append(lr_reducer)
        
        # Custom progress callback
        if progress_callback:
            custom_callback = CustomProgressCallback(progress_callback)
            callbacks.append(custom_callback)
        
        return callbacks
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        if self.model is None:
            raise Exception("No trained model available")
        
        # Handle generator vs array evaluation
        if hasattr(X_test, '__len__') and hasattr(X_test, '__getitem__'):
            results = self.model.evaluate(X_test, verbose=0)
        else:
            results = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Return structured results
        metrics = {}
        if len(results) >= 2:
            metrics['loss'] = results[0]
            metrics['accuracy'] = results[1] if len(results) > 1 else None
        
        return metrics
    
    def predict(self, X):
        """Make predictions with the trained model"""
        if self.model is None:
            raise Exception("No trained model available")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            raise Exception("No trained model available")
        
        # Capture model summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        
        return '\n'.join(summary_lines)


class CustomProgressCallback(keras.callbacks.Callback):
    """Enhanced progress callback with memory monitoring"""
    
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training"""
        self.start_time = time.time()
        print("🏁 Training started")
    
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the beginning of each epoch"""
        # Force garbage collection at the start of each epoch
        gc.collect()
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        if self.progress_callback and hasattr(self.progress_callback, 'on_epoch_end'):
            self.progress_callback.on_epoch_end(epoch, logs)
        
        # Print progress with memory info
        if logs:
            elapsed = time.time() - self.start_time if self.start_time else 0
            print(f"⏱️  Epoch {epoch + 1} completed in {elapsed:.1f}s - "
                  f"Loss: {logs.get('loss', 0):.4f} - "
                  f"Val Loss: {logs.get('val_loss', 0):.4f}")
        
        # Periodic memory cleanup
        if (epoch + 1) % 5 == 0:
            gc.collect()
    
    def on_train_end(self, logs=None):
        """Called at the end of training"""
        total_time = time.time() - self.start_time if self.start_time else 0
        print(f"🏆 Training completed in {total_time:.1f} seconds")
        gc.collect()


class MemoryManagementCallback(keras.callbacks.Callback):
    """Callback to manage memory during training"""
    
    def __init__(self, cleanup_frequency=5):
        super().__init__()
        self.cleanup_frequency = cleanup_frequency
        self.epoch_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Cleanup memory periodically"""
        self.epoch_count += 1
        
        if self.epoch_count % self.cleanup_frequency == 0:
            print(f"🧹 Performing memory cleanup at epoch {epoch + 1}")
            gc.collect()
            
            # Try to get memory info if psutil is available
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                print(f"💾 Current memory usage: {memory_mb:.1f} MB")
            except ImportError:
                pass
    
    def on_train_begin(self, logs=None):
        """Initial memory cleanup"""
        gc.collect()
        print("🧠 Memory management initialized")


class TrainingMonitor:
    """Enhanced training monitor with memory optimization insights"""
    
    def __init__(self):
        self.training_history = []
    
    def analyze_training(self, history, using_generators=False):
        """Analyze training history and provide insights"""
        insights = {
            'converged': self._check_convergence(history),
            'overfitting': self._check_overfitting(history),
            'best_epoch': self._find_best_epoch(history),
            'training_stability': self._assess_stability(history),
            'memory_efficient': using_generators,
            'recommendations': []
        }
        
        # Generate recommendations based on analysis
        insights['recommendations'] = self._generate_recommendations(insights, history, using_generators)
        
        return insights
    
    def _check_convergence(self, history):
        """Check if training has converged"""
        if 'loss' not in history.history:
            return False
        
        losses = history.history['loss']
        if len(losses) < 5:
            return False
        
        # Check if loss has stabilized in the last 5 epochs
        recent_losses = losses[-5:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Convergence if standard deviation is less than 1% of mean
        return loss_std < 0.01 * loss_mean
    
    def _check_overfitting(self, history):
        """Check for signs of overfitting"""
        if 'loss' not in history.history or 'val_loss' not in history.history:
            return False
        
        train_losses = history.history['loss']
        val_losses = history.history['val_loss']
        
        if len(train_losses) < 5:
            return False
        
        # Check if validation loss is consistently higher and increasing
        recent_train = np.mean(train_losses[-5:])
        recent_val = np.mean(val_losses[-5:])
        
        train_trend = np.polyfit(range(5), train_losses[-5:], 1)[0]
        val_trend = np.polyfit(range(5), val_losses[-5:], 1)[0]
        
        # Overfitting if val_loss > train_loss and val_loss trending up
        return recent_val > recent_train * 1.1 and val_trend > 0
    
    def _find_best_epoch(self, history):
        """Find the epoch with best validation performance"""
        if 'val_loss' not in history.history:
            return len(history.history['loss']) - 1
        
        val_losses = history.history['val_loss']
        return np.argmin(val_losses)
    
    def _assess_stability(self, history):
        """Assess training stability"""
        if 'loss' not in history.history:
            return 'unknown'
        
        losses = history.history['loss']
        if len(losses) < 10:
            return 'insufficient_data'
        
        # Calculate coefficient of variation for recent epochs
        recent_losses = losses[-10:]
        cv = np.std(recent_losses) / np.mean(recent_losses)
        
        if cv < 0.05:
            return 'very_stable'
        elif cv < 0.1:
            return 'stable'
        elif cv < 0.2:
            return 'moderately_stable'
        else:
            return 'unstable'
    
    def _generate_recommendations(self, insights, history, using_generators):
        """Generate training recommendations including memory optimization tips"""
        recommendations = []
        
        if insights['overfitting']:
            recommendations.append({
                'type': 'overfitting',
                'message': 'Model is overfitting. Consider adding regularization, dropout, or reducing model complexity.',
                'actions': ['Add dropout layers', 'Reduce learning rate', 'Add early stopping', 'Get more training data']
            })
        
        if not insights['converged']:
            recommendations.append({
                'type': 'convergence',
                'message': 'Model has not converged. Consider training for more epochs or adjusting learning rate.',
                'actions': ['Increase epochs', 'Adjust learning rate', 'Check data preprocessing']
            })
        
        if insights['training_stability'] == 'unstable':
            recommendations.append({
                'type': 'stability',
                'message': 'Training is unstable. Consider reducing learning rate or batch size.',
                'actions': ['Reduce learning rate', 'Use learning rate scheduler', 'Increase batch size']
            })
        
        # Memory optimization recommendations
        if using_generators:
            recommendations.append({
                'type': 'memory_optimization',
                'message': 'Using memory-efficient training with data generators. This allows training on large datasets.',
                'actions': ['Generators automatically handle memory', 'Consider reducing batch size if memory issues persist', 'Monitor system memory usage']
            })
        else:
            # Check if dataset might benefit from generators
            if 'val_accuracy' in history.history:
                recommendations.append({
                    'type': 'memory_optimization',
                    'message': 'For larger datasets, consider using data generators to reduce memory usage.',
                    'actions': ['Enable generator mode for large datasets', 'Reduce image resolution if needed', 'Use batch processing']
                })
        
        # Performance-based recommendations
        if 'val_accuracy' in history.history:
            final_acc = history.history['val_accuracy'][-1]
            if final_acc < 0.7:
                recommendations.append({
                    'type': 'performance',
                    'message': 'Model accuracy is below 70%. Consider improving the model architecture or data quality.',
                    'actions': ['Try different architecture', 'Increase model capacity', 'Improve data preprocessing', 'Data augmentation']
                })
        
        return recommendations


class DataAugmentation:
    """Memory-efficient data augmentation utilities"""
    
    @staticmethod
    def get_image_augmentation(config):
        """Get image data augmentation based on config"""
        aug_config = config.get('augmentation', {})
        
        if not aug_config.get('enabled', False):
            return None
        
        # Use TensorFlow's more memory-efficient augmentation
        augmentation_layers = []
        
        if aug_config.get('horizontal_flip', True):
            augmentation_layers.append(
                keras.layers.RandomFlip("horizontal")
            )
        
        if aug_config.get('rotation', 0) > 0:
            augmentation_layers.append(
                keras.layers.RandomRotation(aug_config.get('rotation', 20) / 360.0)
            )
        
        if aug_config.get('zoom', 0) > 0:
            zoom_factor = aug_config.get('zoom', 0.2)
            augmentation_layers.append(
                keras.layers.RandomZoom((-zoom_factor, zoom_factor))
            )
        
        if aug_config.get('width_shift', 0) > 0 or aug_config.get('height_shift', 0) > 0:
            width_factor = aug_config.get('width_shift', 0.2)
            height_factor = aug_config.get('height_shift', 0.2)
            augmentation_layers.append(
                keras.layers.RandomTranslation(height_factor, width_factor)
            )
        
        if augmentation_layers:
            return keras.Sequential(augmentation_layers)
        
        return None
    
    @staticmethod
    def apply_augmentation_to_model(model, augmentation):
        """Apply augmentation layers to the beginning of a model"""
        if augmentation is None:
            return model
        
        # Create new model with augmentation at the beginning
        inputs = model.input
        x = augmentation(inputs)
        
        # Get all layers except the input layer
        for layer in model.layers[1:]:
            x = layer(x)
        
        augmented_model = keras.Model(inputs, x)
        return augmented_model


class ModelOptimizer:
    """Memory-efficient model optimization utilities"""
    
    def __init__(self, max_memory_gb=4.0):
        self.best_params = None
        self.best_score = None
        self.max_memory_gb = max_memory_gb
    
    def optimize_for_memory(self, model, target_memory_mb=2048):
        """Optimize model architecture for memory constraints"""
        
        # Get current model memory usage estimate
        current_params = model.count_params()
        estimated_memory_mb = self._estimate_model_memory(model)
        
        recommendations = {
            'current_params': current_params,
            'estimated_memory_mb': estimated_memory_mb,
            'within_budget': estimated_memory_mb <= target_memory_mb,
            'optimizations': []
        }
        
        if estimated_memory_mb > target_memory_mb:
            # Suggest optimizations
            excess_memory = estimated_memory_mb - target_memory_mb
            reduction_needed = excess_memory / estimated_memory_mb
            
            recommendations['optimizations'].extend([
                f"Reduce model parameters by ~{reduction_needed*100:.1f}%",
                "Consider using smaller filter sizes in CNN layers",
                "Reduce the number of filters in convolutional layers",
                "Use depthwise separable convolutions",
                "Implement gradient checkpointing",
                "Use mixed precision training (float16)"
            ])
        
        return recommendations
    
    def _estimate_model_memory(self, model):
        """Estimate model memory usage in MB"""
        # Rough estimation based on parameters and layer types
        total_params = model.count_params()
        
        # Base memory for parameters (4 bytes per float32 parameter)
        param_memory = total_params * 4
        
        # Additional memory for activations, gradients, etc.
        # This is a rough estimate and can vary significantly
        overhead_multiplier = 4  # Conservative estimate
        
        total_memory_bytes = param_memory * overhead_multiplier
        total_memory_mb = total_memory_bytes / (1024 * 1024)
        
        return total_memory_mb
    
    def suggest_batch_size(self, model, input_shape, available_memory_gb=4.0):
        """Suggest optimal batch size based on model and memory constraints"""
        
        # Estimate memory per sample
        input_size_bytes = np.prod(input_shape) * 4  # float32
        
        # Estimate model memory
        model_memory_mb = self._estimate_model_memory(model)
        
        # Available memory for batch processing (leave some headroom)
        available_batch_memory_mb = (available_memory_gb * 1024 * 0.6) - model_memory_mb
        available_batch_memory_bytes = available_batch_memory_mb * 1024 * 1024
        
        # Estimate memory per sample including gradients and activations
        memory_per_sample = input_size_bytes * 3  # Input + gradients + activations
        
        # Calculate max batch size
        max_batch_size = max(1, int(available_batch_memory_bytes / memory_per_sample))
        
        # Suggest powers of 2 for efficiency
        suggested_batch_size = min(max_batch_size, 128)  # Cap at 128
        suggested_batch_size = max(1, 2 ** int(np.log2(suggested_batch_size)))
        
        return {
            'suggested_batch_size': suggested_batch_size,
            'max_theoretical_batch_size': max_batch_size,
            'model_memory_mb': model_memory_mb,
            'memory_per_sample_bytes': memory_per_sample,
            'reasoning': f"Based on {available_memory_gb}GB available memory and model size"
        }


# Utility functions for memory management
def get_memory_usage():
    """Get current memory usage if psutil is available"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024
        }
    except ImportError:
        return None

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    import gc
    
    # Run garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Clear TensorFlow session if needed
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    print("🧹 Aggressive memory cleanup completed")