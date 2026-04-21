"""
===============================================================================
INTELLIGENT CAREER GUIDANCE SYSTEM - CONTINUOUS LOOP
FULLY DATA-DRIVEN: Powered by real-world job market data

Pipeline:
OCR → BERT → Hybrid Features → ANN → Data-Driven Insights

What changed from hardcoded to data-driven?
============================================
BEFORE (Hardcoded):
- Skills, companies, salaries were manually defined in ROLE_SKILLS dict
- Limited to pre-defined knowledge
- Doesn't reflect actual job market trends

AFTER (Data-Driven):
- Insights LEARNED from job_skills.csv + linkedin_job_postings.csv
- Reflects REAL employer requirements from thousands of job postings
- Automatically adapts to market trends
- More accurate and realistic recommendations

Features:
- Continuous resume analysis loop
- Top-3 predictions with confidence bars
- DATA-DRIVEN skill gap analysis (from real job postings)
- DATA-DRIVEN company recommendations (from actual hiring patterns)
- DATA-DRIVEN salary insights (from dataset or market estimates)
- Auto-save reports
- Session tracking

Usage:
    python intelligent_career_guidance.py
===============================================================================
"""

import os
import sys
import time
import numpy as np
import pickle
import torch
from datetime import datetime

# Import modules
from ocr_module import extract_text_with_fallback
from bert_embedding_module import BERTEmbeddingExtractor
from hybrid_feature_engineering import HybridFeatureEngineer
from data_driven_insights import DataDrivenInsights

# TensorFlow for model
import tensorflow as tf
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = 'models'
RESULTS_DIR = 'results'
RESUMES_DIR = 'resumes'
DATA_DIR = 'data_insights'

# Dataset paths (update these to your actual dataset locations)
JOB_SKILLS_PATH = r"C:\Users\shaik\Downloads\job_skills.csv\job_skills.csv"
JOB_POSTINGS_PATH = r"C:\Users\shaik\Downloads\linkedin_job_postings.csv\linkedin_job_postings.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_models():
    """
    Load trained models and data-driven insights engine.
    
    What's loaded:
    --------------
    1. ANN model (bert_ann_model.h5) - Predicts job role from features
    2. Label encoder (label_encoder.pkl) - Maps indices to role names
    3. Feature engineer (feature_engineer.pkl) - Processes resume text
    4. BERT extractor - Generates semantic embeddings
    5. Data-driven insights - REAL job market data for skill/company analysis
    """
    print("\n" + "="*80)
    print("🎓 LOADING MODELS & DATA-DRIVEN INSIGHTS...")
    print("="*80)
    
    # Load ANN model
    model_path = os.path.join(MODEL_DIR, 'bert_ann_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_model(model_path)
    print("✓ ANN model loaded")
    
    # Load label encoder
    with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"✓ Label encoder: {list(label_encoder.classes_)}")
    
    # Load feature engineer
    with open(os.path.join(MODEL_DIR, 'feature_engineer.pkl'), 'rb') as f:
        feature_engineer = pickle.load(f)
    print("✓ Feature engineer loaded")
    
    # Initialize BERT
    bert_extractor = BERTEmbeddingExtractor(model_name='bert-base-uncased')
    print("✓ BERT extractor initialized")
    
    # Initialize data-driven insights (from real job market data)
    print("\n" + "-"*80)
    insights = DataDrivenInsights(data_dir=DATA_DIR)
    insights.load_and_prepare_data(JOB_SKILLS_PATH, JOB_POSTINGS_PATH)
    print("✓ Data-driven insights loaded (REAL job market data)")
    
    return model, label_encoder, feature_engineer, bert_extractor, insights


def extract_skills(text, feature_engineer):
    """Extract skills from resume text"""
    skills = feature_engineer.extract_skills_from_text(text)
    return skills


def generate_predictions(model, features, label_encoder):
    """Generate predictions with confidence scores"""
    probabilities = model.predict(features, verbose=0)[0]
    
    # Get top-3 predictions
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    
    top_3 = []
    for idx in top_3_indices:
        top_3.append({
            'role': label_encoder.classes_[idx],
            'confidence': float(probabilities[idx] * 100)
        })
    
    return top_3, probabilities


def analyze_skill_gaps(skills, predicted_role, insights):
    """
    Compute skill gap using DATA-DRIVEN insights from real job postings.
    
    HOW THIS WORKS (vs old hardcoded approach):
    ===========================================
    OLD (Hardcoded):
      required_skills = ROLE_SKILLS[predicted_role]['essential']
      → Manually defined, may not reflect actual market
    
    NEW (Data-Driven):
      required_skills = insights.get_top_skills(predicted_role, top_n=15)
      → Learned from thousands of real job postings
      → Reflects ACTUAL employer requirements
    
    Example:
      If 847 out of 1000 Web Developer jobs require JavaScript,
      the system KNOWS JavaScript is essential (not hardcoded).
    
    Args:
        skills: Skills extracted from resume
        predicted_role: ANN-predicted job category
        insights: DataDrivenInsights object with real job market data
    
    Returns:
        Dictionary with skill gap analysis
    """
    # Use data-driven insights to compute skill gap
    skill_gap = insights.compute_skill_gap(predicted_role, skills)
    
    return skill_gap


def get_companies_and_salary(predicted_role, insights):
    """
    Get data-driven company recommendations and salary insights.
    
    HOW THIS WORKS (vs old hardcoded approach):
    ===========================================
    OLD (Hardcoded):
      companies = ROLE_SKILLS[predicted_role]['companies']
      → Manually curated list, may be outdated
    
    NEW (Data-Driven):
      companies = insights.get_top_companies(predicted_role, top_n=5)
      → Learned from actual job posting frequency in dataset
      → Companies that ARE ACTUALLY hiring for this role
    
    SALARY INSIGHTS:
    ================
    If dataset contains salary data: Uses real salary information
    Else: Uses market research estimates (2024 US averages)
    
    Args:
        predicted_role: ANN-predicted job category
        insights: DataDrivenInsights object
    
    Returns:
        Tuple: (companies_list, salary_string, company_context)
    """
    # Get top companies from dataset (sorted by hiring frequency)
    companies_with_counts = insights.get_top_companies(predicted_role, top_n=5)
    
    # Format companies with job count (shows market demand)
    companies = []
    for company, count in companies_with_counts:
        companies.append((company, f"{count} job postings in dataset"))
    
    # Get salary info
    salary = insights.format_salary_range(predicted_role)
    
    # Generate context (why these companies?)
    company_context = insights.get_role_description(predicted_role)
    
    return companies, salary, company_context


def create_confidence_bar(percentage, width=30):
    """Create visual confidence bar"""
    filled = int(width * percentage / 100)
    empty = width - filled
    return '█' * filled + '░' * empty


def generate_report(result, skills, report_num):
    """Generate comprehensive career guidance report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append(f"🎓 INTELLIGENT CAREER GUIDANCE REPORT #{report_num}")
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Extracted skills
    report_lines.append("💡 EXTRACTED SKILLS:")
    report_lines.append("-"*80)
    for skill in skills:
        report_lines.append(f"   • {skill}")
    report_lines.append("")
    
    # Top-3 predictions
    report_lines.append("🎯 TOP-3 PREDICTIONS:")
    report_lines.append("-"*80)
    for i, pred in enumerate(result['top_3'], 1):
        bar = create_confidence_bar(pred['confidence'])
        report_lines.append(f"   {i}. {pred['role']:<20}: {pred['confidence']:6.2f}% {bar}")
    report_lines.append("")
    
    # Final prediction
    report_lines.append("🏆 FINAL PREDICTED ROLE:")
    report_lines.append("-"*80)
    report_lines.append(f"   Role: {result['predicted_role']}")
    report_lines.append(f"   Confidence: {result['confidence']:.2f}%")
    report_lines.append(f"   Expected Salary: {result['salary']}")
    report_lines.append("")
    
    # Skill gap analysis
    if result['skill_gap']:
        report_lines.append("🔍 SKILL GAP ANALYSIS:")
        report_lines.append("-"*80)
        report_lines.append(f"   Skill Match: {result['skill_gap']['match_percentage']:.1f}%")
        
        # Readiness level
        match = result['skill_gap']['match_percentage']
        if match >= 80:
            readiness = "✅ Job Ready"
        elif match >= 60:
            readiness = "⚠️  Nearly Ready (learn essentials)"
        else:
            readiness = "❌ Needs Significant Upskilling"
        
        report_lines.append(f"   Readiness: {readiness}")
        report_lines.append("")
        
        if result['skill_gap']['missing_essential']:
            report_lines.append(f"   ⚠️  Missing Essential Skills ({len(result['skill_gap']['missing_essential'])}):")
            for skill in result['skill_gap']['missing_essential']:
                report_lines.append(f"      • {skill}")
            report_lines.append("")
        
        if result['skill_gap']['missing_preferred']:
            report_lines.append(f"   💡 Missing Preferred Skills ({len(result['skill_gap']['missing_preferred'])}):")
            for skill in result['skill_gap']['missing_preferred']:
                report_lines.append(f"      • {skill}")
            report_lines.append("")
        
        if result['skill_gap']['learning_priority']:
            report_lines.append(f"   📚 LEARNING PRIORITY:")
            report_lines.append(f"      Focus on: {', '.join(result['skill_gap']['learning_priority'])}")
            report_lines.append("")
    
    # Company recommendations
    report_lines.append("🏢 SUGGESTED COMPANIES:")
    report_lines.append("-"*80)
    report_lines.append(f"   Why these companies? {result['company_context']}")
    report_lines.append("")
    for i, (company, desc) in enumerate(result['companies'], 1):
        report_lines.append(f"   {i}. {company}")
        report_lines.append(f"      {desc}")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("✅ Report generated by BERT + OCR + ANN Career Guidance System")
    report_lines.append("="*80)
    
    return '\n'.join(report_lines)


def analyze_resume(image_path, model, label_encoder, feature_engineer, bert_extractor, insights, report_num):
    """
    Complete resume analysis pipeline with DATA-DRIVEN insights.
    
    Pipeline:
    =========
    1. OCR: Extract text from resume image
    2. Skills: Extract technical skills from text
    3. BERT: Generate 768-dim semantic embeddings
    4. Features: Create hybrid features (BERT + structured)
    5. ANN: Predict job role using trained model
    6. INSIGHTS: Data-driven skill gap, companies, salary (from real job data)
    """
    print(f"\n{'='*80}")
    print(f"📄 ANALYZING RESUME #{report_num}: {image_path}")
    print(f"{'='*80}")
    
    # Step 1: OCR
    print("\n[1/6] Extracting text from resume (OCR)...")
    resume_text = extract_text_with_fallback(image_path)
    print(f"✅ OCR Complete: Extracted {len(resume_text)} characters")
    
    # Step 2: Extract skills
    print("\n[2/6] Extracting skills...")
    skills = extract_skills(resume_text, feature_engineer)
    print(f"✓ Found {len(skills)} skills: {', '.join(skills[:8])}")
    
    # Step 3: BERT embeddings
    print("\n[3/6] Generating BERT embeddings...")
    bert_embedding = bert_extractor.extract_single_embedding(resume_text)
    print(f"✓ BERT embedding: {bert_embedding.shape}")
    
    # Step 4: Create hybrid features
    print("\n[4/6] Creating hybrid features...")
    # Use transform_single_resume for inference (preserves training dimensions)
    features = feature_engineer.transform_single_resume(bert_embedding, resume_text)
    print(f"✓ Hybrid features: {features.shape}")
    
    # Step 5: Generate predictions
    print("\n[5/6] Generating predictions...")
    top_3, probabilities = generate_predictions(model, features, label_encoder)
    
    predicted_role = top_3[0]['role']
    confidence = top_3[0]['confidence']
    print(f"✓ Top prediction: {predicted_role}")
    
    # Step 6: DATA-DRIVEN skill gap & companies
    print("\n[6/6] Computing DATA-DRIVEN insights (from real job market data)...")
    skill_gap = analyze_skill_gaps(skills, predicted_role, insights)
    companies, salary, company_context = get_companies_and_salary(predicted_role, insights)
    
    match_pct = skill_gap['match_percentage'] if skill_gap else 0
    print(f"✓ Skill match: {match_pct:.1f}% (based on {skill_gap['total_required']} top required skills)")
    print(f"✓ Companies: {len(companies)} suggested (from dataset)")
    print(f"✓ Salary: {salary}")
    
    # Build result
    result = {
        'predicted_role': predicted_role,
        'confidence': confidence,
        'top_3': top_3,
        'skills': skills,
        'skill_gap': skill_gap,
        'companies': companies,
        'company_context': company_context,
        'salary': salary,
        'text': resume_text
    }
    
    # Generate and save report
    report = generate_report(result, skills, report_num)
    
    os.makedirs(RESUMES_DIR, exist_ok=True)
    report_path = os.path.join(RESUMES_DIR, f'resume_{report_num:03d}_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print report to console
    print(f"\n{report}")
    print(f"\n✓ Report saved to: {report_path}")
    
    return result


# ============================================================
# MAIN CONTINUOUS LOOP
# ============================================================

def main():
    """Main continuous loop for resume analysis with data-driven insights"""
    print("\n" + "="*80)
    print("🎓 INTELLIGENT CAREER GUIDANCE SYSTEM")
    print("   FULLY DATA-DRIVEN: Powered by Real Job Market Data")
    print("="*80)
    
    # Load models and data-driven insights
    model, label_encoder, feature_engineer, bert_extractor, insights = load_models()
    
    print("\n" + "="*80)
    print("📋 SYSTEM READY FOR CONTINUOUS ANALYSIS")
    print("="*80)
    print("\nInstructions:")
    print("  • Enter resume image path to analyze")
    print("  • Press Enter to use default: my_resume.png")
    print("  • Type 'exit', 'quit', 'q', or 'e' to stop")
    print("="*80)
    
    report_num = 1
    results = []
    
    while True:
        print(f"\n{'='*80}")
        print(f"📝 Resume #{report_num}")
        print(f"{'='*80}")
        
        try:
            # Get image path
            image_path = input("\nEnter resume image path (or press Enter for default): ").strip().strip('"')
            
            # Check for exit
            if image_path.lower() in ['exit', 'quit', 'q', 'e']:
                print(f"\n{'='*80}")
                print(f"📊 SESSION SUMMARY: Analyzed {report_num - 1} resume(s)")
                print(f"{'='*80}")
                break
            
            # Use default if empty
            if not image_path:
                image_path = 'my_resume.png'
            
            # Check if file exists
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                print("   Please check the path and try again.")
                continue
            
            # Analyze resume
            result = analyze_resume(image_path, model, label_encoder, feature_engineer, bert_extractor, insights, report_num)
            results.append(result)
            
            report_num += 1
            
        except KeyboardInterrupt:
            print(f"\n\n{'='*80}")
            print(f"📊 SESSION SUMMARY: Analyzed {report_num - 1} resume(s)")
            print(f"{'='*80}")
            print("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("   Please try again with a different resume.")
            continue
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 THANK YOU FOR USING THE DATA-DRIVEN CAREER GUIDANCE SYSTEM!")
    print("="*80)
    print(f"\nTotal resumes analyzed: {report_num - 1}")
    print(f"Reports saved to: {RESUMES_DIR}/")
    print("\n💡 All insights are data-driven from real job market data!")
    print("="*80)


if __name__ == "__main__":
    main()
