# 🎓 INTELLIGENT CAREER GUIDANCE SYSTEM
## Domain-Aware Deep Learning Resume Analysis

---

## 📋 PROJECT OVERVIEW

This is an **AI-powered career guidance system** that analyzes resume images and predicts the most suitable job role using deep learning, while providing data-driven skill gap analysis and company recommendations.

### Key Features:
✅ **OCR-based resume parsing** (Tesseract)  
✅ **Deep learning classification** (BERT + ANN)  
✅ **Domain-aware calibration** (Fixes model bias)  
✅ **Data-driven insights** (Real job market data)  
✅ **Skill gap analysis** (Domain-filtered, relevant skills)  
✅ **Hybrid profile detection** (Identifies mixed skill sets)  
✅ **Company recommendations** (Based on actual hiring data)  

---

## 🗂️ PROJECT STRUCTURE

```
INTELLIGENT-CAREER-GUIDANCE/
│
├── 📂 CORE MODULES (System Components)
│   ├── ocr_module.py                    # Text extraction from images
│   ├── bert_embedding_module.py         # Semantic feature extraction
│   ├── hybrid_feature_engineering.py    # Feature combination
│   ├── data_driven_insights.py          # Job market analysis
│   ├── domain_calibration.py            # Probability calibration
│   ├── domain_aware_enhancements.py     # Domain-aware logic
│   └── intelligent_career_guidance.py   # Main application
│
├── 📂 TRAINING & MODELS
│   ├── main_bert_ann.py                 # Training script
│   └── models/                          # Trained models
│       ├── bert_ann_model.h5            # ANN classifier
│       ├── feature_engineer.pkl         # Feature processor
│       ├── label_encoder.pkl            # Role label encoder
│       ├── scaler.pkl                   # Feature scaler
│       └── skill_binarizer.pkl          # Skill vectorizer
│
├── 📂 DATA & INSIGHTS
│   ├── data/                            # Raw datasets (CSV)
│   │   ├── job_skills.csv               # Job postings with skills
│   │   └── linkedin_job_postings.csv    # LinkedIn job data
│   └── data_insights/
│       └── insights_cache.pkl           # Precomputed insights
│
├── 📂 RESULTS & OUTPUTS
│   ├── results/                         # Training reports
│   └── resumes/                         # Analysis reports
│
└── 📂 SUPPORT FILES
    ├── venv/                            # Python virtual environment
    ├── .gitignore                       # Git ignore rules
    └── PROJECT_OVERVIEW.md              # This file
```

---

## 🔬 HOW THE SYSTEM WORKS

### 📊 COMPLETE PIPELINE (6 Steps)

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: OCR - TEXT EXTRACTION                               │
│ ─────────────────────────────────                           │
│ Input: Resume image (PNG, JPG, WEBP)                        │
│ Tool: Tesseract OCR                                         │
│ Output: Raw text (2000-3000 characters)                     │
│ Example: "Experienced Python developer with expertise..."   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: SKILL EXTRACTION & NORMALIZATION                    │
│ ─────────────────────────────────────────                   │
│ Input: Raw text                                             │
│ Process:                                                    │
│   • Extract technical skills using keyword matching         │
│   • Normalize abbreviations (js→javascript, k8s→kubernetes) │
│   • Expand skills (react→javascript, html→frontend)         │
│   • Remove duplicates                                       │
│ Output: List of normalized skills                           │
│ Example: [python, java, javascript, sql, react, docker]     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: BERT EMBEDDINGS - SEMANTIC FEATURES                 │
│ ─────────────────────────────────────────────               │
│ Input: Resume text                                          │
│ Model: bert-base-uncased (FROZEN, not trained)              │
│ Process:                                                    │
│   • Tokenize text                                           │
│   • Pass through BERT                                       │
│   • Extract 768-dimensional embedding                       │
│   • Represents semantic meaning of entire resume            │
│ Output: 768-dim vector                                      │
│ Example: [0.234, -0.567, 0.891, ...] (768 numbers)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: HYBRID FEATURE ENGINEERING                          │
│ ───────────────────────────────────────────                 │
│ Input: BERT embedding + Skills + Text metadata              │
│ Process:                                                    │
│   • BERT embedding: 768 dimensions                          │
│   • Skill vector: 74 dimensions (binary, one per skill)     │
│   • Structured features: 3 dimensions                       │
│     - Years of experience (extracted from text)             │
│     - Number of skills found                                │
│     - Resume text length                                    │
│   • Total: 768 + 74 + 3 = 845 features                      │
│   • Apply StandardScaler (normalization)                    │
│ Output: 845-dim feature vector (scaled)                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: ANN PREDICTION + DOMAIN CALIBRATION                 │
│ ───────────────────────────────────────────────             │
│ Input: 845-dim feature vector                               │
│ Model: Artificial Neural Network                            │
│   • Layer 1: Dense(128) + BatchNorm + Dropout(0.3)          │
│   • Layer 2: Dense(64) + BatchNorm + Dropout(0.3)           │
│   • Output: Dense(7) with Softmax                           │
│   • Classes: 7 job roles                                    │
│                                                             │
│ Raw Prediction (from ANN):                                  │
│   • Web Developer: 76.7%                                    │
│   • Backend Developer: 15.2%                                │
│   • Data Analyst: 5.8%                                      │
│                                                             │
│ ⚡ DOMAIN CALIBRATION (Fixes bias):                         │
│   • Compute domain scores from skills                       │
│   • backend: 0.46, devops: 0.31, data: 0.15, frontend: 0.08│
│   • Identify dominant domain (backend)                      │
│   • Apply strong boost to backend roles                     │
│   • Apply penalty to non-backend roles                      │
│   • Add domain prior (soft influence)                       │
│                                                             │
│ Calibrated Prediction:                                      │
│   • Backend Developer: 48.7% ✅ (corrected!)                │
│   • Web Developer: 40.5%                                    │
│   • Data Analyst: 10.8%                                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: DATA-DRIVEN INSIGHTS                                │
│ ───────────────────────────────────                         │
│ Input: Predicted role + Extracted skills                    │
│ Process:                                                    │
│   • Query insights cache (from real job data)               │
│   • Get top 15 skills required for role                     │
│   • Filter to domain-relevant skills only                   │
│   • Compare with resume skills                              │
│   • Calculate match percentage                              │
│   • Get companies that hire for this role                   │
│   • Get salary range from dataset                           │
│                                                             │
│ Output:                                                     │
│   • Skill match: 65%                                        │
│   • Missing skills: [Docker, Kubernetes, AWS]               │
│   • Top companies: [Google, Microsoft, Amazon]              │
│   • Salary range: $80K - $140K                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    📊 FINAL REPORT
```

---

## 🎯 WHAT MAKES THIS SYSTEM "DATA-DRIVEN"?

### 1️⃣ **DATASET-BASED SKILL EXTRACTION**

**Problem with traditional systems:**
- Hardcoded skill lists (outdated, incomplete)
- Manual maintenance required
- Not reflective of current job market

**Our data-driven approach:**
```python
# Skills come from REAL job postings, not hardcoded
skills = feature_engineer.frequent_skills  # Top 74 skills from dataset

# Example: If "React" appears in 80% of Web Developer jobs in dataset,
# it becomes a key skill for that role
```

**How it works:**
- Dataset analyzed to find most frequent skills per role
- Only skills that appear ≥5 times in dataset are used
- Automatically updated when dataset changes

---

### 2️⃣ **DATASET-BASED ROLE MAPPING**

**Traditional approach:**
```python
# Hardcoded (BAD)
if "react" in skills:
    predict "Web Developer"
```

**Our data-driven approach:**
```python
# Learned from data (GOOD)
# Model trained on 5000 real job postings
# Learns patterns like:
# - Web Developer jobs mention: React, Angular, HTML, CSS (85% of time)
# - Backend Developer jobs mention: Java, Spring, API, SQL (78% of time)
# - Data Analyst jobs mention: SQL, Python, Excel, Tableau (90% of time)
```

---

### 3️⃣ **DATASET-BASED SKILL GAP ANALYSIS**

**Traditional approach:**
- Generic skill recommendations
- Same for everyone
- Not role-specific

**Our data-driven approach:**
```python
# Get top 15 skills ACTUALLY required by employers
required_skills = insights.get_top_skills(role, top_n=15)

# Filter to domain-relevant only (no noise)
relevant_skills = filter_by_domain(role, required_skills)

# Example:
# Web Developer → [JavaScript, React, HTML, CSS, TypeScript]
# NOT → [Python, SQL, Docker] (irrelevant for frontend!)
```

**Why this is better:**
- Recommendations based on what employers ACTUALLY ask for
- Filtered by domain (no irrelevant skills)
- Updated when dataset changes

---

### 4️⃣ **DATASET-BASED COMPANY RECOMMENDATIONS**

**Traditional approach:**
- Static company list
- Not personalized

**Our data-driven approach:**
```python
# Analyze which companies hire for this role in dataset
companies = insights.get_companies_for_role(role, top_n=5)

# Example from real data:
# Backend Developer → [Google, Microsoft, Amazon, Meta, Netflix]
# Data Analyst → [Deloitte, Accenture, IBM, TCS, Infosys]
```

---

### 5️⃣ **DATASET-BASED SALARY INSIGHTS**

**Traditional approach:**
- Generic salary ranges
- Not role-specific

**Our data-driven approach:**
```python
# Calculate from actual job postings in dataset
salary_range = insights.get_salary_range(role)

# Example:
# Backend Developer: $80K - $140K (avg: $105K)
# Data Analyst: $60K - $110K (avg: $80K)
```

---

## 📊 DATASET DESCRIPTION

### **Dataset 1: job_skills.csv**

**Source:** Kaggle / Job postings aggregator  
**Size:** ~10,000+ job postings  
**Columns:**
- `job_title`: Job role (e.g., "Senior Backend Developer")
- `skills`: Required skills (comma-separated)
- `company`: Company name
- `location`: Job location
- `salary`: Salary range (if available)

**Sample Data:**
```
job_title,skills,company,location,salary
Backend Developer,"Java,Spring,Python,SQL,API",Google,New York,$120K
Data Analyst,"SQL,Python,Excel,Tableau",Deloitte,Chicago,$85K
Web Developer,"JavaScript,React,HTML,CSS",Microsoft,Seattle,$110K
```

**How it's used:**
1. **Role normalization:** Maps variations to standard names
   - "Frontend Dev" → "Web Developer"
   - "Backend Engineer" → "Backend Developer"
2. **Skill frequency analysis:** Finds top skills per role
3. **Company analysis:** Identifies top hiring companies
4. **Salary analysis:** Computes salary ranges

---

### **Dataset 2: linkedin_job_postings.csv**

**Source:** LinkedIn job postings  
**Size:** ~5,000+ postings  
**Columns:**
- `job_title`: Job role
- `description`: Full job description
- `skills`: Required skills
- `company`: Company name
- `location`: Location
- `experience_required`: Experience level

**Sample Data:**
```
job_title,description,skills,company
Senior Data Scientist,"We are looking for...","Python,ML,SQL,Statistics",Amazon
Full Stack Developer,"Join our team...","React,Node.js,MongoDB",StartupXYZ
```

**How it's used:**
1. **Skill extraction:** Additional skill data
2. **Role validation:** Confirms role-skill relationships
3. **Market trends:** Identifies emerging skills
4. **Combined with Dataset 1:** Creates comprehensive insights

---

### **DATASET PROCESSING PIPELINE**

```
Raw Datasets (job_skills.csv + linkedin_job_postings.csv)
                    ↓
         Role Normalization
         "Frontend Dev" → "Web Developer"
         "Backend Eng" → "Backend Developer"
                    ↓
         Skill Frequency Analysis
         Count skills per role
         Keep top 74 most frequent
                    ↓
         Company Analysis
         Count companies per role
         Keep top 5 per role
                    ↓
         Salary Calculation
         Compute min/max/avg per role
                    ↓
         Insights Cache (insights_cache.pkl)
         Precomputed for fast querying
```

---

## 📁 DETAILED FILE DESCRIPTIONS

### **CORE MODULES**

#### 1. `ocr_module.py`
**Purpose:** Extract text from resume images  
**How it works:**
- Uses Tesseract OCR engine
- Multiple preprocessing techniques (grayscale, threshold, blur)
- Fallback mechanisms if initial extraction fails
- Returns clean text (2000-3000 characters)

**Key function:**
```python
extract_text_with_fallback(image_path) → text
```

---

#### 2. `bert_embedding_module.py`
**Purpose:** Convert resume text to semantic embeddings  
**How it works:**
- Loads pre-trained BERT model (bert-base-uncased)
- Tokenizes input text
- Passes through BERT layers
- Extracts 768-dimensional embedding
- Model is FROZEN (not trained, just used as feature extractor)

**Key function:**
```python
extract_bert_embedding(text) → 768-dim vector
```

---

#### 3. `hybrid_feature_engineering.py`
**Purpose:** Combine BERT embeddings with structured features  
**How it works:**
- Takes BERT embedding (768 dims)
- Extracts skill vector (74 dims, binary)
- Extracts structured features (3 dims):
  - Years of experience
  - Number of skills
  - Text length
- Concatenates: 768 + 74 + 3 = 845 features
- Applies StandardScaler for normalization

**Key function:**
```python
transform_single_resume(bert_embedding, text) → 845-dim vector
```

---

#### 4. `data_driven_insights.py`
**Purpose:** Analyze job market data and generate insights  
**How it works:**
- Loads datasets (job_skills.csv + linkedin_job_postings.csv)
- Normalizes role names
- Computes skill frequencies per role
- Identifies top companies per role
- Calculates salary ranges
- Saves insights to cache (insights_cache.pkl)

**Key functions:**
```python
get_top_skills(role, top_n=15) → List of skills
get_companies_for_role(role, top_n=5) → List of companies
get_salary_range(role) → (min, max, avg)
```

---

#### 5. `domain_calibration.py`
**Purpose:** Fix ANN prediction bias using domain knowledge  
**How it works:**
- Groups skills into 4 domains: frontend, backend, data, devops
- Computes domain scores from resume skills
- Identifies dominant domain (highest score)
- Applies STRONG boost to matching roles (2x multiplier)
- Applies penalty to non-matching roles (0.75x)
- Adds domain prior (soft influence)
- Normalizes probabilities

**Key function:**
```python
apply_domain_calibration(model_probs, domain_scores, alpha=0.5) → calibrated_probs
```

---

#### 6. `domain_aware_enhancements.py`
**Purpose:** Advanced domain-aware logic (hybrid detection, skill filtering)  
**How it works:**
- Normalizes skills (abbreviations → standard names)
- Expands skills (adds parent categories)
- Detects hybrid profiles (2+ strong domains)
- Filters skill gaps to domain-relevant skills only
- Provides complete prediction pipeline

**Key functions:**
```python
normalize_skills(skills) → normalized list
expand_skills(skills) → expanded list
detect_hybrid(domain_scores) → (is_hybrid, domains)
filter_relevant_skills(role, skills) → filtered list
```

---

#### 7. `intelligent_career_guidance.py` ⭐ **MAIN APPLICATION**
**Purpose:** Orchestrates complete resume analysis pipeline  
**How it works:**
1. Loads all modules and trained models
2. Accepts resume image path from user
3. Runs 6-step pipeline (OCR → Skills → BERT → Features → Prediction → Insights)
4. Generates detailed report
5. Saves report to file
6. Loops for multiple resumes

**Key function:**
```python
analyze_resume(image_path, ...) → result dictionary
```

---

### **TRAINING & MODELS**

#### 8. `main_bert_ann.py`
**Purpose:** Train the BERT + ANN model  
**What it does:**
- Loads datasets
- Preprocesses data (normalization, stratified sampling)
- Extracts BERT embeddings for all samples
- Engineers hybrid features
- Trains ANN classifier
- Saves trained models

**Model Architecture:**
```
Input (845 features)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.3)
    ↓
Dense(7) + Softmax
    ↓
Output: 7 job roles
```

**Training Configuration:**
- Epochs: 100 (with early stopping)
- Batch size: 32
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Class weights: Balanced (handles imbalance)

---

#### 9. `models/` Directory
**Contents:**
- `bert_ann_model.h5` - Trained ANN classifier
- `feature_engineer.pkl` - Feature engineering pipeline
- `label_encoder.pkl` - Maps roles to indices (and back)
- `scaler.pkl` - StandardScaler for feature normalization
- `skill_binarizer.pkl` - Converts skills to binary vector

**Why saved as .pkl?**
- Faster loading than retraining
- Preserves exact state
- Portable across systems

---

### **DATA & INSIGHTS**

#### 10. `data/` Directory
**Contents:**
- `job_skills.csv` - Job postings with skills
- `linkedin_job_postings.csv` - LinkedIn job data

**How to update:**
- Replace CSV files with new data
- Run `data_driven_insights.py` to rebuild cache
- System automatically uses new insights

---

#### 11. `data_insights/insights_cache.pkl`
**Purpose:** Precomputed job market insights  
**Contents:**
- Top skills per role (from dataset)
- Top companies per role (from dataset)
- Salary ranges per role (from dataset)
- Skill-company associations

**Why cached?**
- Fast loading (no re-computation)
- Reduces memory usage
- Speeds up prediction

---

### **RESULTS & OUTPUTS**

#### 12. `results/` Directory
**Contents:**
- Training reports
- Performance metrics
- Confusion matrices
- Training curves

---

#### 13. `resumes/` Directory
**Contents:**
- Analysis reports for each resume
- Format: `resume_XXX_report.txt`
- Contains full predictions, skill gaps, recommendations

---

## 🚀 HOW TO USE THE SYSTEM

### **Step 1: Activate Environment**
```bash
cd "c:\Users\shaik\Downloads\new'"
.\venv\Scripts\activate
```

### **Step 2: Run Application**
```bash
python intelligent_career_guidance.py
```

### **Step 3: Enter Resume Image**
```
Enter resume image path: my_resume.png
```

### **Step 4: View Results**
```
📊 PREDICTION RESULTS:
   Predicted Role: Backend Developer
   Confidence: 48.7%

🔍 SKILL GAP ANALYSIS:
   Skill Match: 65%
   Missing: Docker, Kubernetes, AWS

🏢 SUGGESTED COMPANIES:
   1. Google - "Top tech company hiring backend developers"
   2. Microsoft - "Cloud services division"
   ...
```

---

## 🎓 ACADEMIC HIGHLIGHTS

### **Why This System is Innovative:**

1. **Hybrid Deep Learning:**
   - Combines BERT (semantic understanding) with ANN (classification)
   - Better than either alone

2. **Domain-Aware Calibration:**
   - Fixes model bias without hard rules
   - Probabilistic, explainable approach
   - Mathematically sound

3. **Data-Driven Insights:**
   - All recommendations from real job data
   - No hardcoded lists
   - Reflects actual job market

4. **Hybrid Profile Detection:**
   - Identifies candidates with mixed skills
   - Adjusts confidence appropriately
   - More realistic than forced single-class

5. **Domain-Filtered Skill Gaps:**
   - Only relevant skills recommended
   - No noise (Web Developer doesn't need Python)
   - Actionable recommendations

---

## 📊 SYSTEM ARCHITECTURE DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│                    USER INPUT                                │
│              Resume Image (PNG/JPG/WEBP)                     │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                  OCR MODULE                                  │
│            (ocr_module.py)                                   │
│  Tesseract → Preprocessing → Text Extraction → Clean Text   │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│              SKILL EXTRACTION                                │
│        (hybrid_feature_engineering.py)                       │
│  Keyword Matching → Normalization → Expansion → Skill List  │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│              BERT EMBEDDINGS                                 │
│          (bert_embedding_module.py)                          │
│  Tokenization → BERT Model (Frozen) → 768-dim Embedding     │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│           HYBRID FEATURE ENGINEERING                         │
│        (hybrid_feature_engineering.py)                       │
│  BERT(768) + Skills(74) + Metadata(3) = 845 Features        │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                 ANN PREDICTION                               │
│            (bert_ann_model.h5)                               │
│  Dense(128) → Dense(64) → Dense(7,Softmax) → Raw Probs      │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│            DOMAIN CALIBRATION                                │
│          (domain_calibration.py)                             │
│  Domain Scores → Boost/Penalty → Prior → Calibrated Probs   │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│            DATA-DRIVEN INSIGHTS                              │
│          (data_driven_insights.py)                           │
│  Skill Gap + Companies + Salary (from real job data)        │
└──────────────────────┬───────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────────┐
│                    FINAL REPORT                              │
│        (intelligent_career_guidance.py)                      │
│  Prediction + Confidence + Skill Gap + Companies + Salary   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔑 KEY INNOVATIONS SUMMARY

| Feature | Traditional Systems | Our System |
|---------|-------------------|------------|
| **Skill Lists** | Hardcoded, static | Dataset-driven, dynamic |
| **Role Mapping** | Manual rules | Learned from data |
| **Skill Gaps** | Generic | Domain-filtered, relevant |
| **Companies** | Static list | Based on hiring data |
| **Salaries** | Generic ranges | Computed from dataset |
| **Hybrid Profiles** | Ignored | Detected & explained |
| **Model Bias** | Unaddressed | Domain-calibrated |
| **Explainability** | Black box | Transparent, auditable |

---

## 📝 CONCLUSION

This system represents a **modern, data-driven approach** to career guidance that:

✅ Uses **real job market data** (not assumptions)  
✅ Combines **deep learning** with **domain knowledge**  
✅ Provides **actionable, relevant** recommendations  
✅ Is **transparent and explainable**  
✅ Scales **automatically** with new data  

**Perfect for:**
- Job seekers (skill gap analysis)
- Career counselors (data-driven guidance)
- HR professionals (resume screening)
- Academic projects (demonstrates ML + data integration)

---

## 🎯 FUTURE ENHANCEMENTS

- [ ] Add more datasets (Indeed, Glassdoor)
- [ ] Implement real-time web scraping
- [ ] Add user feedback loop
- [ ] Create web interface
- [ ] Add resume score calculation
- [ ] Integrate with job APIs

---

**Built with ❤️ using Python, TensorFlow, BERT, and Real Job Market Data**
