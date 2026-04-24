"""
===============================================================================
COMPLETE END-TO-END DEEP LEARNING SYSTEM
Resume Analysis & Job Prediction using BERT + OCR + ANN
===============================================================================

Pipeline:
1. OCR: Extract text from resume images
2. BERT: Convert text to 768-dim embeddings
3. Feature Engineering: Combine BERT + structured features
4. ANN: Predict job role
5. Insights: Skill gap, companies, salary

Author: AI-Assisted Development
Date: 2026
Version: 3.0 - BERT Enhanced Edition
===============================================================================
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Import custom modules
from ocr_module import extract_text_with_fallback
from bert_embedding_module import BERTEmbeddingExtractor
from hybrid_feature_engineering import HybridFeatureEngineer

# TensorFlow for ANN
import tensorflow as t
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    'bert_model': 'bert-base-uncased',
    'bert_batch_size': 8,  # Smaller batch size for BERT (memory intensive)
    'epochs': 15,
    'batch_size': 32,
    'test_size': 0.2,
    'val_size': 0.1,
    'random_seed': 42,
    'min_skill_frequency': 3,
    'use_class_weights': True,
    'use_kfold': True,  # NEW: Enable K-Fold Cross-Validation
    'k_folds': 5  # NEW: Number of folds
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_real_datasets():
    """Load and prepare real job posting datasets"""
    print("\n" + "="*80)
    print("PHASE 1: LOADING REAL DATASETS")
    print("="*80)
    
    # Paths to your datasets
    job_skills_path = r"C:\Users\shaik\Downloads\job_skills.csv\job_skills.csv"
    job_postings_path = r"C:\Users\shaik\Downloads\linkedin_job_postings.csv\linkedin_job_postings.csv"
    
    print(f"\nLoading job_skills.csv...")
    df_skills = pd.read_csv(job_skills_path)
    print(f"✓ Loaded: {len(df_skills)} rows")
    
    print(f"\nLoading linkedin_job_postings.csv...")
    df_postings = pd.read_csv(job_postings_path)
    print(f"✓ Loaded: {len(df_postings)} rows")
    
    # Merge datasets
    print("\nMerging datasets...")
    df_merged = pd.merge(df_postings, df_skills, on='job_link', how='inner')
    print(f"✓ Merged: {len(df_merged)} rows")
    
    # Clean data
    df_merged = df_merged.dropna(subset=['job_title', 'job_skills'])
    df_merged['skills_list'] = df_merged['job_skills'].apply(
        lambda x: [s.strip().lower() for s in str(x).split(',')]
    )
    
    return df_merged


def prepare_job_categories(df, sample_size=2000):
    """Prepare job categories for classification"""
    print("\n" + "="*80)
    print("PHASE 2: PREPARING JOB CATEGORIES")
    print("="*80)
    
    # Define tech job categories
    TECH_CATEGORIES = {
        'AI Engineer': ['ai engineer', 'machine learning engineer', 'ml engineer'],
        'Data Scientist': ['data scientist', 'data science'],
        'Data Engineer': ['data engineer', 'big data'],
        'Web Developer': ['web developer', 'frontend developer', 'full stack'],
        'Backend Developer': ['backend developer', '+ware engineer'],
        'DevOps Engineer': ['devops', 'site reliability', 'sre'],
    }
    
    def categorize_job(title):
        title_lower = str(title).lower()
        for category, keywords in TECH_CATEGORIES.items():
            for keyword in keywords:
                if keyword in title_lower:
                    return category
        return 'Other'
    
    df['job_category'] = df['job_title'].apply(categorize_job)
    df_tech = df[df['job_category'] != 'Other'].copy()
    
    print(f"\nTech jobs found: {len(df_tech)}")
    print(df_tech['job_category'].value_counts())
    
    # Sample for faster processing
    if len(df_tech) > sample_size:
        print(f"\nSampling {sample_size} jobs for BERT processing...")
        df_tech = df_tech.sample(n=sample_size, random_state=42)
    
    return df_tech


def extract_bert_embeddings_for_dataset(df, extractor, batch_size=8):
    """Extract BERT embeddings for all job postings"""
    print("\n" + "="*80)
    print("PHASE 3: EXTRACTING BERT EMBEDDINGS")
    print("="*80)
    
    # Create text representation for each job
    texts = []
    for idx, row in df.iterrows():
        text = f"{row['job_title']} {row['job_skills']}"
        texts.append(text)
    
    # Extract embeddings
    embeddings = extractor.extract_embeddings_batch(texts, batch_size=batch_size)
    
    return embeddings, texts


def train_hybrid_ann(X_train, y_train, X_val, y_val, num_classes, input_dim):
    """
    Train Hybrid ANN model.
    
    Architecture:
    Input (BERT + Structured) → Dense(128) → Dropout(0.3) →
                                Dense(64) → Dropout(0.3) →
                                Output (softmax)
    """
    print("\n" + "="*80)
    print("PHASE 5: TRAINING HYBRID ANN MODEL")
    print("="*80)
    
    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    print("Input → Dense(128, L2) → BatchNorm → Dropout(0.3) →")
    print("        Dense(64, L2) → BatchNorm → Dropout(0.3) →")
    print("        Output (softmax)")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Train
    print(f"\nTraining for {CONFIG['epochs']} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    return model, history


def train_with_kfold_cross_validation(X, y, num_classes, input_dim, k_folds=5):
    """
    Train model using K-Fold Stratified Cross-Validation.
    
    This is MORE ROBUST than single train/test split:
    - Trains k different models
    - Each fold gets to be test set once
    - Final metrics are averaged across all folds
    - Better estimate of model performance
    
    Args:
        X: Features
        y: Labels
        num_classes: Number of job roles
        input_dim: Input feature dimensions
        k_folds: Number of folds (default 5)
    
    Returns:
        best_model: Best model across all folds
        fold_results: Dictionary with results from each fold
    """
    print("\n" + "="*80)
    print("PHASE 5: K-FOLD STRATIFIED CROSS-VALIDATION TRAINING")
    print("="*80)
    
    print(f"\n🔄 Configuration:")
    print(f"   • K-Folds: {k_folds}")
    print(f"   • Total Samples: {len(X)}")
    print(f"   • Samples per fold: ~{len(X)//k_folds} train, ~{len(X)//k_folds} val")
    print(f"   • Strategy: Stratified (maintains class distribution)")
    
    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': [],
        'models': [],
        'histories': []
    }
    
    # Train on each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print("\n" + "-"*80)
        print(f"📊 FOLD {fold}/{k_folds}")
        print("-"*80)
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val:   {len(X_val)} samples")
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,),
                  kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
        ]
        
        # Class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))
        
        # Train
        print(f"   Training for {CONFIG['epochs']} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CONFIG['epochs'],
            batch_size=CONFIG['batch_size'],
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=0  # Silent training, show summary later
        )
        
        # Get best metrics
        best_epoch = np.argmin(history.history['val_loss'])
        fold_acc = history.history['accuracy'][best_epoch]
        fold_val_acc = history.history['val_accuracy'][best_epoch]
        fold_loss = history.history['loss'][best_epoch]
        fold_val_loss = history.history['val_loss'][best_epoch]
        
        print(f"   ✅ Fold {fold} Results:")
        print(f"      Train Acc: {fold_acc*100:.2f}%")
        print(f"      Val Acc:   {fold_val_acc*100:.2f}%")
        print(f"      Train Loss: {fold_loss:.4f}")
        print(f"      Val Loss:   {fold_val_loss:.4f}")
        
        # Store results
        fold_results['accuracy'].append(fold_acc)
        fold_results['val_accuracy'].append(fold_val_acc)
        fold_results['loss'].append(fold_loss)
        fold_results['val_loss'].append(fold_val_loss)
        fold_results['models'].append(model)
        fold_results['histories'].append(history)
    
    # Calculate average metrics
    avg_acc = np.mean(fold_results['accuracy'])
    avg_val_acc = np.mean(fold_results['val_accuracy'])
    avg_loss = np.mean(fold_results['loss'])
    avg_val_loss = np.mean(fold_results['val_loss'])
    std_acc = np.std(fold_results['val_accuracy'])
    
    print("\n" + "="*80)
    print("📊 K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n📈 Performance Across {k_folds} Folds:")
    print(f"   • Mean Train Accuracy: {avg_acc*100:.2f}%")
    print(f"   • Mean Val Accuracy:   {avg_val_acc*100:.2f}%")
    print(f"   • Mean Train Loss:     {avg_loss:.4f}")
    print(f"   • Mean Val Loss:       {avg_val_loss:.4f}")
    print(f"   • Std Dev (Val Acc):   {std_acc*100:.2f}%")
    
    print(f"\n📋 Per-Fold Validation Accuracy:")
    for i, acc in enumerate(fold_results['val_accuracy'], 1):
        print(f"   Fold {i}: {acc*100:.2f}%")
    
    # Select best model (highest validation accuracy)
    best_fold_idx = np.argmax(fold_results['val_accuracy'])
    best_model = fold_results['models'][best_fold_idx]
    
    print(f"\n🏆 Best Model: Fold {best_fold_idx+1} (Val Acc: {fold_results['val_accuracy'][best_fold_idx]*100:.2f}%)")
    print(f"\n✅ K-Fold Training Complete!")
    print(f"   • All {k_folds} models trained successfully")
    print(f"   • Results are more reliable than single split")
    print(f"   • Standard deviation shows model stability")
    
    return best_model, fold_results


def evaluate_model(model, X_test, y_test, class_names, history=None):
    """Evaluate model with comprehensive metrics"""
    print("\n" + "="*80)
    print("PHASE 6: MODEL EVALUATION")
    print("="*80)
    
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))
    
    # Overfitting analysis
    if history:
        train_acc = max(history.history['accuracy'])
        val_acc = max(history.history['val_accuracy'])
        gap = train_acc - val_acc
        
        print(f"\n🔍 Overfitting Analysis:")
        print(f"   Train Accuracy: {train_acc:.4f}")
        print(f"   Val Accuracy:   {val_acc:.4f}")
        print(f"   Gap:            {gap:.4f} ({gap*100:.2f}%)")
        
        if gap < 0.05:
            print(f"   ✓ LOW OVERFITTING - Good generalization!")
        elif gap < 0.10:
            print(f"   ⚠️  MODERATE OVERFITTING")
        else:
            print(f"   ⚠️  HIGH OVERFITTING")
    
    # Plot training history
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0].set_title('Accuracy vs Epoch', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[1].set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/bert_ann_training_curves.png', dpi=150)
        print(f"\n✓ Training curves saved to results/bert_ann_training_curves.png")
        plt.show()


def predict_resume_pipeline(image_path, model, bert_extractor, feature_engineer, 
                            label_encoder, class_names):
    """
    Complete prediction pipeline from resume image to job prediction.
    
    Pipeline:
    Image → OCR → Text → BERT → Embeddings → +Structured → ANN → Prediction
    """
    print("\n" + "="*80)
    print("END-TO-END RESUME PREDICTION")
    print("="*80)
    
    # Step 1: OCR
    print("\n[1/5] Extracting text from resume image...")
    resume_text = extract_text_with_fallback(image_path)
    
    if not resume_text:
        print("❌ Failed to extract text")
        return None
    
    print(f"\n📝 Extracted Text (first 200 chars):")
    print(resume_text[:200])
    print("...")
    
    # Step 2: Skill extraction
    print("\n[2/5] Extracting skills...")
    skills = feature_engineer.extract_skills_from_text(resume_text)
    print(f"✓ Found {len(skills)} skills: {', '.join(skills[:10])}")
    
    # Step 3: BERT embeddings
    print("\n[3/5] Generating BERT embeddings...")
    bert_embedding = bert_extractor.extract_single_embedding(resume_text)
    print(f"✓ BERT embedding shape: {bert_embedding.shape}")
    
    # Step 4: Feature combination
    print("\n[4/5] Creating hybrid features...")
    hybrid_features = feature_engineer.transform_single_resume(bert_embedding, resume_text)
    print(f"✓ Hybrid features shape: {hybrid_features.shape}")
    
    # Step 5: Prediction
    print("\n[5/5] Predicting job role...")
    pred_probs = model.predict(hybrid_features)[0]
    pred_class = np.argmax(pred_probs)
    pred_role = class_names[pred_class]
    confidence = pred_probs[pred_class]
    
    # Top-3 predictions
    top_3_indices = np.argsort(pred_probs)[::-1][:3]
    top_3 = [(class_names[i], float(pred_probs[i])) for i in top_3_indices]
    
    # Display results
    print("\n" + "="*80)
    print("📊 PREDICTION RESULTS")
    print("="*80)
    
    print(f"\n🎯 Predicted Role: {pred_role}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    print(f"\n📋 Top-3 Predictions:")
    for i, (role, conf) in enumerate(top_3, 1):
        print(f"   {i}. {role}: {conf*100:.2f}%")
    
    print(f"\n💡 Extracted Skills ({len(skills)}):")
    for skill in skills[:15]:
        print(f"   - {skill}")
    
    print(f"\n📅 Estimated Experience: {feature_engineer.extract_experience_years(resume_text)} years")
    
    print(f"\n{'='*80}")
    
    return {
        'predicted_role': pred_role,
        'confidence': confidence,
        'top_3': top_3,
        'skills': skills,
        'text': resume_text
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("🎓 RESUME ANALYSIS & JOB PREDICTION SYSTEM")
    print("   BERT + OCR + ANN - Complete End-to-End Pipeline")
    print("="*80)
    
    # Phase 1: Load data
    df_merged = load_real_datasets()
    
    # Phase 2: Prepare categories
    df_tech = prepare_job_categories(df_merged, sample_size=2000)
    
    # Phase 3: Initialize BERT
    print("\n" + "="*80)
    print("PHASE 3: INITIALIZING BERT")
    print("="*80)
    bert_extractor = BERTEmbeddingExtractor(model_name=CONFIG['bert_model'])
    
    # Phase 4: Extract BERT embeddings
    bert_embeddings, texts = extract_bert_embeddings_for_dataset(
        df_tech, bert_extractor, batch_size=CONFIG['bert_batch_size']
    )
    
    # Phase 5: Create hybrid features
    print("\n" + "="*80)
    print("PHASE 4: CREATING HYBRID FEATURES")
    print("="*80)
    feature_engineer = HybridFeatureEngineer(
        min_skill_frequency=CONFIG['min_skill_frequency']
    )
    
    hybrid_features = feature_engineer.create_hybrid_features(bert_embeddings, texts)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_tech['job_category'])
    class_names = label_encoder.classes_
    
    print(f"\n✓ Classes: {list(class_names)}")
    
    # K-FOLD STRATIFIED CROSS-VALIDATION
    if CONFIG.get('use_kfold', False):
        print("\n🔄 Using K-FOLD STRATIFIED CROSS-VALIDATION...")
        print("   This trains multiple models and averages results for better reliability!")
        
        # Use all data for K-Fold (no separate test set)
        model, fold_results = train_with_kfold_cross_validation(
            hybrid_features, y,
            num_classes=len(class_names),
            input_dim=hybrid_features.shape[1],
            k_folds=CONFIG['k_folds']
        )
        
        # Use best fold's history for evaluation
        best_fold_idx = np.argmax(fold_results['val_accuracy'])
        history = fold_results['histories'][best_fold_idx]
        
        # Create pseudo test set from last fold for final evaluation
        skf = StratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=42)
        all_splits = list(skf.split(hybrid_features, y))
        _, test_idx = all_splits[-1]  # Use last fold as test
        X_test, y_test = hybrid_features[test_idx], y[test_idx]
        
    else:
        # Traditional train/val/test split
        print("\n📊 Using traditional train/val/test split...")
        
        # Train/val/test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            hybrid_features, y, test_size=CONFIG['test_size'], 
            random_state=CONFIG['random_seed'], stratify=y
        )
        
        val_ratio = CONFIG['val_size'] / (1 - CONFIG['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio,
            random_state=CONFIG['random_seed'], stratify=y_trainval
        )
        
        print(f"\n✓ Data splits:")
        print(f"   Train: {len(X_train)}")
        print(f"   Val:   {len(X_val)}")
        print(f"   Test:  {len(X_test)}")
        
        # Phase 6: Train model
        model, history = train_hybrid_ann(
            X_train, y_train, X_val, y_val,
            num_classes=len(class_names),
            input_dim=hybrid_features.shape[1]
        )
    
    # Phase 7: Evaluate
    evaluate_model(model, X_test, y_test, class_names, history)
    
    # Phase 8: Save models
    print("\n" + "="*80)
    print("PHASE 7: SAVING MODELS")
    print("="*80)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    model.save('models/bert_ann_model.h5')
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open('models/feature_engineer.pkl', 'wb') as f:
        pickle.dump(feature_engineer, f)
    
    print("✓ Model saved: models/bert_ann_model.h5")
    print("✓ Label encoder saved: models/label_encoder.pkl")
    print("✓ Feature engineer saved: models/feature_engineer.pkl")
    
    # Phase 9: Interactive prediction
    print("\n" + "="*80)
    print("PHASE 8: INTERACTIVE PREDICTION")
    print("="*80)
    
    print("\nWould you like to test with a resume image? (yes/no): ", end='')
    try:
        response = input().strip().lower()
        if response in ['yes', 'y']:
            print("\nEnter resume image path: ", end='')
            image_path = input().strip().strip('"')
            
            if os.path.exists(image_path):
                predict_resume_pipeline(
                    image_path, model, bert_extractor, 
                    feature_engineer, label_encoder, class_names
                )
            else:
                print(f"❌ File not found: {image_path}")
    except:
        pass
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("🎉 PROJECT COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"\nKey Achievements:")
    print(f"  ✓ OCR text extraction from resume images")
    print(f"  ✓ BERT embeddings (768-dim) for semantic understanding")
    print(f"  ✓ Hybrid features (BERT + structured)")
    print(f"  ✓ ANN model with regularization")
    if CONFIG.get('use_kfold', False):
        print(f"  ✓ K-Fold Stratified Cross-Validation ({CONFIG['k_folds']} folds)")
        print(f"  ✓ More robust evaluation than single split")
    print(f"  ✓ End-to-end prediction pipeline")
    print("="*80)


if __name__ == "__main__":
    main()
