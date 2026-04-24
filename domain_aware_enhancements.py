"""
===============================================================================
DOMAIN-AWARE RESUME ANALYSIS ENHANCEMENTS
Data-driven logic for skill filtering, hybrid detection, and role normalization

Improvements:
1. Domain-based skill gap filtering (no irrelevant skills)
2. Hybrid profile detection (frontend + backend = full stack)
3. Role normalization for dataset consistency
4. Skill expansion and grouping
5. Clean pipeline integration
===============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================
# IMPROVEMENT 1: DOMAIN-BASED SKILL FILTERING
# ============================================================

# Role to domain mapping (abstraction layer, not hardcoded rules)
ROLE_DOMAIN = {
    'Web Developer': 'frontend',
    'Backend Developer': 'backend',
    'Data Analyst': 'data',
    'Data Scientist': 'data',
    'Data Engineer': 'data',
    'DevOps Engineer': 'devops',
    'AI Engineer': 'data'
}

# Domain to allowed skills (filters irrelevant skills from gap analysis)
# This ensures Web Developer doesn't get Python/SQL recommendations
DOMAIN_SKILLS = {
    'frontend': [
        'javascript', 'react', 'vue', 'angular', 'css', 'html', 'typescript',
        'sass', 'less', 'bootstrap', 'tailwind', 'webpack', 'babel',
        'jquery', 'redux', 'next.js', 'nuxt.js', 'frontend'
    ],
    'backend': [
        'java', 'python', 'node.js', 'sql', 'api', 'spring', 'express',
        'django', 'flask', 'fastapi', 'ruby', 'rails', 'php', 'laravel',
        'c#', '.net', 'rest api', 'graphql', 'microservices', 'serverless',
        'go', 'golang', 'rust', 'kotlin', 'scala'
    ],
    'data': [
        'python', 'sql', 'pandas', 'machine learning', 'statistics',
        'data analysis', 'data science', 'deep learning', 'tensorflow',
        'pytorch', 'keras', 'numpy', 'scikit-learn', 'tableau', 'power bi',
        'excel', 'r language', 'matplotlib', 'seaborn', 'analytics',
        'visualization', 'data engineering', 'etl', 'big data', 'spark'
    ],
    'devops': [
        'docker', 'kubernetes', 'aws', 'linux', 'ci/cd', 'jenkins',
        'terraform', 'ansible', 'bash', 'shell', 'git', 'azure', 'gcp',
        'devops', 'site reliability', 'sre', 'cloud', 'infrastructure',
        'monitoring', 'prometheus', 'grafana', 'nginx'
    ]
}


def filter_relevant_skills(role: str, required_skills: List[str]) -> List[str]:
    """
    Filter required skills to only include domain-relevant skills.
    
    Problem this solves:
    --------------------
    Without filtering, Web Developer might show "Python, SQL" as missing skills
    (because they appear in dataset), but those aren't relevant for frontend roles.
    
    How it works:
    -------------
    1. Get domain for the role (e.g., Web Developer → frontend)
    2. Get allowed skills for that domain
    3. Filter required_skills to only include domain-relevant ones
    
    Why this is NOT hardcoding:
    ---------------------------
    - Required skills STILL come from dataset (top 10 most frequent)
    - We're just FILTERING out irrelevant ones based on domain
    - Domain abstraction makes it scalable (add new roles easily)
    
    Example:
        Role: Web Developer
        Domain: frontend
        Required skills from dataset: [javascript, react, python, sql, html, css]
        
        After filtering:
        [javascript, react, html, css]  # Python, SQL removed (not frontend)
    
    Args:
        role: Predicted job role
        required_skills: Top skills from dataset (before filtering)
    
    Returns:
        Filtered list of domain-relevant skills
    """
    # Get domain for this role
    domain = ROLE_DOMAIN.get(role, None)
    
    if domain is None:
        # Unknown role - return all skills (fallback)
        return required_skills
    
    # Get allowed skills for this domain
    allowed = DOMAIN_SKILLS.get(domain, [])
    
    # Filter required skills to only include domain-relevant ones
    filtered = []
    for skill in required_skills:
        # Check if skill matches any allowed skill (partial match)
        if any(allowed_skill in skill.lower() or skill.lower() in allowed_skill 
               for allowed_skill in allowed):
            filtered.append(skill)
    
    return filtered


# ============================================================
# IMPROVEMENT 2: HYBRID PROFILE DETECTION
# ============================================================

def detect_hybrid(domain_scores: Dict[str, float], threshold: float = 0.3) -> Tuple[bool, List[str]]:
    """
    Detect hybrid profiles (e.g., frontend + backend = full stack).
    
    Problem this solves:
    --------------------
    Mixed profiles (e.g., knows React AND Java) are often misclassified
    because model forces single-class prediction.
    
    How it works:
    -------------
    1. Check domain scores (frontend, backend, data, devops)
    2. Count how many domains have strong signals (>30%)
    3. If 2+ domains are strong → hybrid profile detected
    
    Example:
        Domain scores:
        - frontend: 0.45  (React, Angular, HTML)
        - backend:  0.40  (Java, Spring, API)
        - data:     0.10
        - devops:   0.05
        
        Result: Hybrid detected! (frontend + backend)
        Interpretation: Full Stack Developer
    
    What we do with this info:
    --------------------------
    - DO NOT override prediction (model still decides)
    - REDUCE confidence by 15% (acknowledge ambiguity)
    - ADD explanation to report ("Hybrid profile: frontend + backend")
    
    Args:
        domain_scores: Computed domain scores from skills
        threshold: Minimum score to consider a domain "strong" (default 0.3)
    
    Returns:
        Tuple: (is_hybrid, strong_domains)
    """
    # Find domains with strong signals
    strong_domains = [domain for domain, score in domain_scores.items() 
                      if score > threshold]
    
    # If 2+ domains are strong, it's a hybrid profile
    is_hybrid = len(strong_domains) >= 2
    
    return is_hybrid, strong_domains


# ============================================================
# IMPROVEMENT 3: DATASET ROLE NORMALIZATION
# ============================================================

def normalize_role(role: str) -> Optional[str]:
    """
    Normalize inconsistent role names from dataset.
    
    Problem this solves:
    --------------------
    Dataset contains inconsistent role names:
    - "Frontend Developer", "Frontend Dev", "UI Developer" → should be "Web Developer"
    - "Backend Dev", "Server-side Developer" → should be "Backend Developer"
    - "Data Analyst", "SQL Analyst", "Business Analyst" → should be "Data Analyst"
    
    This normalization happens BEFORE training to ensure clean labels.
    
    How it works:
    -------------
    1. Convert role to lowercase
    2. Check for keywords
    3. Map to standardized role name
    4. Return None if role doesn't match any category (will be filtered out)
    
    Why this is important:
    ----------------------
    - Ensures consistent class labels
    - Prevents model from learning duplicate categories
    - Improves accuracy by reducing noise
    
    Example:
        "Senior Frontend Developer" → "Web Developer"
        "Full Stack Engineer" → "Backend Developer"
        "Business Intelligence Analyst" → "Data Analyst"
    
    Args:
        role: Original role name from dataset
    
    Returns:
        Normalized role name, or None if unmappable
    """
    role = str(role).lower().strip()
    
    # Web Developer (frontend roles)
    if any(kw in role for kw in ['frontend', 'front-end', 'ui developer', 'web developer']):
        return 'Web Developer'
    
    # Backend Developer (includes full stack)
    if any(kw in role for kw in ['backend', 'back-end', 'full stack', 'software engineer']):
        return 'Backend Developer'
    
    # Data Analyst (all analyst roles except data scientist)
    if 'analyst' in role and 'data scientist' not in role:
        return 'Data Analyst'
    
    # Data Scientist (must be explicit)
    if any(kw in role for kw in ['data scientist', 'ml scientist', 'ai scientist']):
        return 'Data Scientist'
    
    # DevOps Engineer
    if any(kw in role for kw in ['devops', 'site reliability', 'sre', 'cloud engineer']):
        return 'DevOps Engineer'
    
    # Data Engineer
    if any(kw in role for kw in ['data engineer', 'big data', 'etl developer']):
        return 'Data Engineer'
    
    # AI Engineer
    if any(kw in role for kw in ['ai engineer', 'ml engineer', 'machine learning engineer']):
        return 'AI Engineer'
    
    # Unknown role - will be filtered out
    return None


# ============================================================
# IMPROVEMENT 4: SKILL NORMALIZATION + EXPANSION
# ============================================================

# Skill grouping: base skill → variants
SKILL_GROUPS = {
    'javascript': ['react', 'vue', 'angular', 'node.js', 'jquery', 'redux'],
    'frontend': ['html', 'css', 'sass', 'less', 'bootstrap', 'tailwind'],
    'backend': ['api', 'rest api', 'graphql', 'microservices', 'serverless'],
    'database': ['sql', 'postgresql', 'mysql', 'mongodb', 'nosql', 'redis'],
    'cloud': ['aws', 'azure', 'gcp', 'cloud'],
    'ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'keras'],
    'data_processing': ['pandas', 'numpy', 'scikit-learn', 'spark'],
    'visualization': ['tableau', 'power bi', 'matplotlib', 'seaborn', 'plotly'],
    'devops': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'terraform', 'ansible']
}


def expand_skills(skills: List[str]) -> List[str]:
    """
    Expand skills by adding base/parent skills when variants are present.
    
    Problem this solves:
    --------------------
    If resume has "React, Angular" but not "JavaScript", skill gap analysis
    might say "JavaScript missing" (even though React IS JavaScript).
    
    How it works:
    -------------
    1. For each skill group, check if any variant is present
    2. If yes, add the base skill to the list
    3. Remove duplicates
    
    Example:
        Input: ['react', 'angular', 'html', 'css']
        
        Processing:
        - 'react' in javascript variants → add 'javascript'
        - 'angular' in javascript variants → 'javascript' already added
        - 'html' in frontend variants → add 'frontend'
        - 'css' in frontend variants → 'frontend' already added
        
        Output: ['react', 'angular', 'html', 'css', 'javascript', 'frontend']
    
    IMPORTANT:
    - Keeps original skills (never removes)
    - Only adds parent/base skills
    - Avoids duplicates
    
    Args:
        skills: List of extracted skills
    
    Returns:
        Expanded list with base skills added
    """
    expanded = set(skills)  # Use set to avoid duplicates
    
    # Check each skill group
    for base_skill, variants in SKILL_GROUPS.items():
        # If any variant is present, add the base skill
        if any(variant in skills for variant in variants):
            expanded.add(base_skill)
    
    return list(expanded)


def normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalize individual skills to standard names.
    
    Examples:
    - 'js' → 'javascript'
    - 'ts' → 'typescript'
    - 'k8s' → 'kubernetes'
    - 'ml' → 'machine learning'
    
    Args:
        skills: List of extracted skills
    
    Returns:
        Normalized skills list
    """
    # Common abbreviations and aliases
    NORMALIZATION_MAP = {
        'js': 'javascript',
        'ts': 'typescript',
        'k8s': 'kubernetes',
        'ml': 'machine learning',
        'dl': 'deep learning',
        'cv': 'computer vision',
        'nlp': 'natural language processing',
        'db': 'database',
        'aws': 'aws',
        'gcp': 'gcp',
    }
    
    normalized = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        # Use normalized name if exists, otherwise keep original
        normalized.append(NORMALIZATION_MAP.get(skill_lower, skill))
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for skill in normalized:
        if skill not in seen:
            seen.add(skill)
            unique.append(skill)
    
    return unique


# ============================================================
# COMPLETE ENHANCED PREDICTION PIPELINE
# ============================================================

def enhanced_predict_with_domain_awareness(model, features, label_encoder, 
                                           skills, resume_text, insights, 
                                           alpha=0.5):
    """
    Complete prediction pipeline with stronger domain-aware calibration.
    
    Pipeline Order:
    ===============
    1. Normalize skills (abbreviations → standard names)
    2. Expand skills (add base skills from groups)
    3. Compute domain scores from expanded skills
    4. Get raw model predictions (BERT + ANN)
    5. Apply STRONG domain calibration (alpha=0.5)
    6. Detect hybrid profiles
    7. Adjust confidence if hybrid
    8. Get required skills from dataset
    9. Filter to domain-relevant skills only
    10. Compute skill gap on filtered skills
    
    Args:
        model: Trained ANN model
        features: Hybrid features
        label_encoder: Label encoder
        skills: Raw extracted skills
        resume_text: Full resume text
        insights: DataDrivenInsights object
        alpha: Calibration strength (default 0.5, increased from 0.2)
    
    Returns:
        Dictionary with enhanced predictions
    """
    # Step 1: Normalize skills
    skills_normalized = normalize_skills(skills)
    
    # Step 2: Expand skills (add base skills)
    skills_expanded = expand_skills(skills_normalized)
    
    # Step 3: Compute domain scores
    from domain_calibration import compute_domain_scores
    domain_scores = compute_domain_scores(skills_expanded)
    
    # Step 4: Raw model predictions
    raw_probs = model.predict(features, verbose=0)[0]
    
    # Step 5: Apply soft domain calibration
    from domain_calibration import apply_domain_calibration, calibrate_confidence
    calibrated_probs = apply_domain_calibration(
        raw_probs, label_encoder, domain_scores, alpha=alpha
    )
    
    # Step 6: Detect hybrid profiles
    is_hybrid, strong_domains = detect_hybrid(domain_scores, threshold=0.3)
    
    # Step 7: Adjust confidence if hybrid
    if is_hybrid:
        calibrated_probs = calibrate_confidence(calibrated_probs, factor=0.85)
    
    # Step 8: Get top-3 predictions
    top_3_indices = np.argsort(calibrated_probs)[::-1][:3]
    top_3 = []
    for idx in top_3_indices:
        top_3.append({
            'role': label_encoder.classes_[idx],
            'confidence': float(calibrated_probs[idx] * 100)
        })
    
    # Step 9: Get predicted role
    predicted_role = top_3[0]['role']
    
    # Step 10: Get required skills from dataset
    from domain_calibration import get_technical_skills_only
    required_skills = get_technical_skills_only(predicted_role, insights, top_n=15)
    
    # Step 11: Filter to domain-relevant skills
    relevant_skills = filter_relevant_skills(predicted_role, required_skills)
    
    # Step 12: Compute skill gap on filtered skills
    skills_lower = [s.lower() for s in skills_expanded]
    missing_skills = [s for s in relevant_skills if s.lower() not in skills_lower]
    matched_count = len(relevant_skills) - len(missing_skills)
    
    match_percentage = (matched_count / len(relevant_skills) * 100) if relevant_skills else 0
    
    return {
        'raw_probs': raw_probs,
        'domain_scores': domain_scores,
        'calibrated_probs': calibrated_probs,
        'top_3': top_3,
        'predicted_role': predicted_role,
        'confidence': top_3[0]['confidence'],
        'is_hybrid': is_hybrid,
        'strong_domains': strong_domains,
        'skills_normalized': skills_normalized,
        'skills_expanded': skills_expanded,
        'required_skills': relevant_skills,  # Filtered!
        'skill_gap': {
            'match_percentage': match_percentage,
            'total_required': len(relevant_skills),
            'matched_count': matched_count,
            'missing_skills': missing_skills[:5],  # Top 5
            'learning_priority': missing_skills[:2]  # Top 2
        }
    }
