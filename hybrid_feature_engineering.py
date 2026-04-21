"""
===============================================================================
HYBRID FEATURE ENGINEERING
Combines BERT embeddings with structured features (skills, experience)
===============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from collections import Counter


class HybridFeatureEngineer:
    """
    Creates hybrid features by combining:
    1. BERT embeddings (768-dim text features)
    2. Structured features (skills, experience, etc.)
    
    Why Hybrid Approach?
    ====================
    Earlier, deep ANN alone failed because:
    - Raw text features (TF-IDF) were too sparse (38K features)
    - Model couldn't capture semantic meaning
    - Overfitting due to high dimensionality
    
    Now with hybrid approach:
    - BERT provides dense, semantic text features (768-dim)
    - Structured features add explicit signals (skills, experience)
    - Combined: Best of both worlds!
    """
    
    def __init__(self, min_skill_frequency=5):
        """
        Initialize feature engineer.
        
        Args:
            min_skill_frequency: Minimum frequency for skill inclusion
        """
        self.min_skill_frequency = min_skill_frequency
        self.skill_binarizer = None
        self.scaler = StandardScaler()
        self.frequent_skills = None
        
        # Common tech skills for extraction
        self.TECH_SKILLS = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby',
            'go', 'rust', 'swift', 'kotlin', 'sql', 'nosql', 'mongodb', 'postgresql',
            'mysql', 'redis', 'elasticsearch', 'react', 'angular', 'vue', 'node.js',
            'express', 'django', 'flask', 'fastapi', 'spring', 'hibernate',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
            'matplotlib', 'seaborn', 'machine learning', 'deep learning', 'nlp',
            'computer vision', 'data science', 'data analysis', 'statistics',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'ci/cd',
            'git', 'agile', 'scrum', 'linux', 'bash', 'shell',
            'html', 'css', 'sass', 'less', 'bootstrap', 'tailwind',
            'rest api', 'graphql', 'microservices', 'serverless',
            'tableau', 'power bi', 'excel', 'sap', 'salesforce',
            'project management', 'leadership', 'communication', 'teamwork'
        ]
    
    def extract_skills_from_text(self, text):
        """
        Extract skills from resume text using keyword matching.
        
        Why keyword matching instead of complex NLP?
        - Fast and efficient
        - Works well for technical skills
        - Easy to understand and maintain
        - No additional model training needed
        
        Args:
            text: Resume text
        
        Returns:
            List of extracted skills
        """
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.TECH_SKILLS:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_experience_years(self, text):
        """
        Extract years of experience from text.
        
        Args:
            text: Resume text
        
        Returns:
            Estimated years of experience (int)
        """
        import re
        
        if not text:
            return 0
        
        # Look for patterns like "5 years", "10+ years", "3 yrs experience"
        patterns = [
            r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp)\s*:\s*(\d+)\s*(?:years?|yrs?)',
            r'(\d+)\s*\+\s*years?\s*(?:of\s*)?(?:experience|exp)'
        ]
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                years = [int(m) for m in matches if m.isdigit()]
                if years:
                    max_years = max(max_years, max(years))
        
        return min(max_years, 30)  # Cap at 30 years
    
    def prepare_structured_features(self, texts):
        """
        Prepare structured features from texts.
        
        Features:
        1. Skills (multi-hot encoded)
        2. Experience years
        3. Skill count
        4. Text length (proxy for resume completeness)
        
        Args:
            texts: List of resume texts
        
        Returns:
            Numpy array of structured features
        """
        print("\n" + "="*80)
        print("PREPARING STRUCTURED FEATURES")
        print("="*80)
        
        # Extract skills
        print("\n[1/4] Extracting skills from texts...")
        all_skills = [self.extract_skills_from_text(text) for text in texts]
        
        # Filter frequent skills
        print("[2/4] Filtering frequent skills...")
        skill_counts = Counter()
        for skills in all_skills:
            skill_counts.update(skills)
        
        self.frequent_skills = {skill for skill, count in skill_counts.items() 
                               if count >= self.min_skill_frequency}
        
        print(f"  Total unique skills found: {len(skill_counts)}")
        print(f"  Skills with frequency >= {self.min_skill_frequency}: {len(self.frequent_skills)}")
        
        # Filter skills
        filtered_skills = [[s for s in skills if s in self.frequent_skills] 
                          for skills in all_skills]
        
        # Multi-hot encode skills
        print("[3/4] Encoding skills...")
        self.skill_binarizer = MultiLabelBinarizer()
        skills_matrix = self.skill_binarizer.fit_transform(filtered_skills)
        
        # Extract experience
        print("[4/4] Extracting experience and creating features...")
        experience_years = np.array([self.extract_experience_years(text) for text in texts])
        skill_counts = np.array([len(skills) for skills in filtered_skills])
        text_lengths = np.array([len(text) for text in texts])
        
        # Combine structured features
        structured_features = np.column_stack([
            skills_matrix,              # Multi-hot encoded skills
            experience_years.reshape(-1, 1),
            skill_counts.reshape(-1, 1),
            text_lengths.reshape(-1, 1)
        ])
        
        print(f"\n✓ Structured features shape: {structured_features.shape}")
        print(f"  - Skills: {skills_matrix.shape[1]}")
        print(f"  - Experience years: 1")
        print(f"  - Skill count: 1")
        print(f"  - Text length: 1")
        print(f"  - Total: {structured_features.shape[1]}")
        
        return structured_features
    
    def create_hybrid_features(self, bert_embeddings, texts):
        """
        Create hybrid features by combining BERT embeddings with structured features.
        
        Architecture:
        ┌─────────────────────────────────────┐
        │ Resume Text                         │
        ├──────────────┬──────────────────────┤
        │ BERT Model   │ Feature Extraction   │
        ├──────────────┼──────────────────────┤
        │ 768-dim      │ Skills, Experience   │
        │ Embeddings   │ (50-100 features)    │
        └──────────────┴──────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │  Combine &    │
                │  Scale        │
                └───────────────┘
                        │
                        ▼
                Final Feature Vector
                (~850-900 dimensions)
        
        Args:
            bert_embeddings: BERT embeddings (n_samples, 768)
            texts: List of resume texts
        
        Returns:
            Combined feature matrix
        """
        print("\n" + "="*80)
        print("CREATING HYBRID FEATURES (BERT + STRUCTURED)")
        print("="*80)
        
        # Prepare structured features
        structured_features = self.prepare_structured_features(texts)
        
        # Combine BERT embeddings with structured features
        print("\nCombining BERT embeddings with structured features...")
        hybrid_features = np.hstack([bert_embeddings, structured_features])
        
        print(f"\n✓ Hybrid features shape: {hybrid_features.shape}")
        print(f"  - BERT embeddings: 768")
        print(f"  - Structured features: {structured_features.shape[1]}")
        print(f"  - Total: {hybrid_features.shape[1]}")
        
        # Scale features (important for neural networks)
        print("\nScaling features...")
        hybrid_features_scaled = self.scaler.fit_transform(hybrid_features)
        
        print(f"✓ Features scaled and ready for training")
        
        return hybrid_features_scaled
    
    def transform_single_resume(self, bert_embedding, text):
        """
        Transform a single resume into hybrid features.
        
        CRITICAL: This method MUST produce the exact same number of features
        as create_hybrid_features used during training.
        
        Args:
            bert_embedding: BERT embedding for the resume (768-dim)
            text: Resume text
        
        Returns:
            Scaled feature vector
        """
        # Extract structured features
        skills = self.extract_skills_from_text(text)
        filtered_skills = [s for s in skills if s in self.frequent_skills]
        
        # CRITICAL FIX: Create skill vector with EXACT same dimensions as training
        # The skill_binarizer was fitted during training with specific classes
        # We must ALWAYS create a vector of that same length, even if no skills match
        num_skill_classes = len(self.skill_binarizer.classes_)
        skills_vector = np.zeros(num_skill_classes)
        
        for skill in filtered_skills:
            if skill in self.skill_binarizer.classes_:
                idx = list(self.skill_binarizer.classes_).index(skill)
                skills_vector[idx] = 1
        
        # Other features
        exp_years = self.extract_experience_years(text)
        skill_count = len(filtered_skills)
        text_length = len(text)
        
        # Combine structured features
        structured = np.concatenate([
            skills_vector,  # This will ALWAYS have the correct number of dimensions
            [exp_years, skill_count, text_length]
        ])
        
        # Combine BERT + structured (same as training)
        hybrid = np.concatenate([bert_embedding, structured])
        
        # Scale the entire hybrid vector
        hybrid_scaled = self.scaler.transform(hybrid.reshape(1, -1))
        
        return hybrid_scaled


# Example usage
if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "Experienced Python developer with 5 years of experience in machine learning, deep learning, tensorflow, and pytorch",
        "Frontend web developer skilled in React, JavaScript, HTML, CSS with 3 years experience",
        "Data scientist with expertise in Python, pandas, numpy, scikit-learn, statistics, and data analysis"
    ]
    
    # Simulate BERT embeddings (random for testing)
    bert_embeddings = np.random.randn(3, 768)
    
    # Create hybrid features
    engineer = HybridFeatureEngineer(min_skill_frequency=1)
    hybrid_features = engineer.create_hybrid_features(bert_embeddings, sample_texts)
    
    print(f"\nFinal hybrid features shape: {hybrid_features.shape}")
