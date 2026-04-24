"""
===============================================================================
DATA-DRIVEN CAREER INSIGHTS MODULE
Replaces hardcoded ROLE_SKILLS dictionary with real-world job market data

Why Data-Driven Approach?
==========================
Earlier: Skills, companies, and salaries were HARDCODED manually
  - Limited to pre-defined knowledge
  - Doesn't reflect actual job market trends
  - Requires manual updates when market changes

Now: Insights are LEARNED from real job posting datasets
  - Reflects actual employer requirements
  - Automatically adapts to market trends
  - More accurate and realistic recommendations
  - Scales to new roles without manual intervention

Datasets Used:
- job_skills.csv: Skills required for different job postings
- linkedin_job_postings.csv: Company names, job titles, salaries

Pipeline:
1. Load and merge datasets
2. Build role → skills mapping (frequency-based)
3. Build role → company mapping (frequency-based)
4. Compute salary ranges per role (if available)
5. Provide real-time insights during prediction
===============================================================================
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import pickle
from typing import Dict, List, Tuple, Optional


# ============================================================
# DATA-DRIVEN INSIGHTS CLASS
# ============================================================

class DataDrivenInsights:
    """
    Builds and provides data-driven career insights from real job market data.
    
    Why this class?
    ---------------
    Instead of manually defining what skills each role needs, we LEARN it from 
    thousands of real job postings. This makes the system:
    
    1. MORE ACCURATE: Based on actual employer requirements, not assumptions
    2. UP-TO-DATE: Reflects current market trends
    3. SCALABLE: Automatically works for any role in the dataset
    4. REALISTIC: Company suggestions come from actual hiring patterns
    
    Example:
        Earlier: "Web Developers need JavaScript" (hardcoded assumption)
        Now: "Web Developers need JavaScript" (learned from 847 job postings)
    """
    
    def __init__(self, data_dir: str = 'data_insights'):
        """
        Initialize data-driven insights engine.
        
        Args:
            data_dir: Directory to cache precomputed mappings
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Mappings (will be built from data)
        self.role_skill_map = {}       # role → {skill: frequency}
        self.role_company_map = {}     # role → {company: frequency}
        self.role_salary_map = {}      # role → {min, max, avg}
        self.role_descriptions = {}    # role → description
        
        # Dataset cache
        self.df_merged = None
        
    def load_and_prepare_data(self, 
                              job_skills_path: str,
                              job_postings_path: str,
                              force_rebuild: bool = False) -> None:
        """
        Load datasets and build all mappings.
        
        Why precompute?
        ---------------
        Building mappings takes time (parsing CSVs, counting frequencies).
        We do it ONCE and save to disk. Future runs load instantly.
        
        Args:
            job_skills_path: Path to job_skills.csv
            job_postings_path: Path to linkedin_job_postings.csv
            force_rebuild: If True, rebuild even if cached version exists
        """
        cache_path = os.path.join(self.data_dir, 'insights_cache.pkl')
        
        # Try loading from cache (faster)
        if not force_rebuild and os.path.exists(cache_path):
            print("\n" + "="*80)
            print("LOADING CACHED DATA-DRIVEN INSIGHTS")
            print("="*80)
            print(f"\nLoading precomputed mappings from: {cache_path}")
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.role_skill_map = cached_data['role_skill_map']
            self.role_company_map = cached_data['role_company_map']
            self.role_salary_map = cached_data['role_salary_map']
            self.role_descriptions = cached_data['role_descriptions']
            
            print(f"✓ Loaded {len(self.role_skill_map)} role mappings")
            print(f"✓ Skills mapping: {len(self.role_skill_map)} roles")
            print(f"✓ Company mapping: {len(self.role_company_map)} roles")
            print(f"✓ Salary mapping: {len(self.role_salary_map)} roles")
            return
        
        # Build from scratch (takes time but only once)
        print("\n" + "="*80)
        print("BUILDING DATA-DRIVEN INSIGHTS FROM REAL JOB DATA")
        print("="*80)
        
        # Step 1: Load datasets
        print("\n[1/5] Loading job skills dataset...")
        df_skills = pd.read_csv(job_skills_path)
        print(f"✓ Loaded: {len(df_skills)} rows")
        
        print("\n[2/5] Loading LinkedIn job postings...")
        df_postings = pd.read_csv(job_postings_path)
        print(f"✓ Loaded: {len(df_postings)} rows")
        
        # Step 2: Merge datasets
        print("\n[3/5] Merging datasets...")
        # Find common column (usually 'job_link' or similar)
        common_cols = set(df_skills.columns) & set(df_postings.columns)
        print(f"   Common columns: {common_cols}")
        
        # Use job_link if available, otherwise use index
        if 'job_link' in common_cols:
            df_merged = pd.merge(df_postings, df_skills, on='job_link', how='inner')
        else:
            # Fallback: merge on index if no common column
            df_merged = pd.merge(
                df_postings.reset_index(), 
                df_skills.reset_index(),
                on='index',
                how='inner'
            )
        
        print(f"✓ Merged: {len(df_merged)} rows")
        self.df_merged = df_merged
        
        # Step 3: Categorize jobs
        print("\n[4/5] Categorizing job roles...")
        df_merged = self._categorize_jobs(df_merged)
        
        # Step 4: Build mappings
        print("\n[5/5] Building role-skill and role-company mappings...")
        self._build_skill_mappings(df_merged)
        self._build_company_mappings(df_merged)
        self._build_salary_mappings(df_merged)
        self._generate_role_descriptions()
        
        # Step 5: Save cache
        print(f"\n💾 Caching insights for faster future loads...")
        cached_data = {
            'role_skill_map': self.role_skill_map,
            'role_company_map': self.role_company_map,
            'role_salary_map': self.role_salary_map,
            'role_descriptions': self.role_descriptions
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"✓ Cache saved to: {cache_path}")
        
        # Print summary
        print(f"\n" + "="*80)
        print("DATA-DRIVEN INSIGHTS SUMMARY")
        print("="*80)
        for role in self.role_skill_map.keys():
            num_skills = len(self.role_skill_map[role])
            num_companies = len(self.role_company_map.get(role, {}))
            print(f"   {role}: {num_skills} unique skills, {num_companies} companies")
    
    def _categorize_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize job titles into standard roles.
        
        Why categorization?
        -------------------
        Job titles vary wildly: "Sr. Frontend Dev", "UI Engineer", "React Developer"
        We normalize them into standard categories for consistent analysis.
        
        Args:
            df: Merged dataset
            
        Returns:
            DataFrame with 'job_category' column
        """
        TECH_CATEGORIES = {
            'AI Engineer': ['ai engineer', 'machine learning engineer', 'ml engineer', 'ai researcher'],
            'Data Scientist': ['data scientist', 'data science', 'ml scientist'],
            'Data Analyst': ['data analyst', 'sql analyst', 'business analyst', 'analytics', 'bi analyst'],
            'Data Engineer': ['data engineer', 'big data', 'data architect', 'etl developer'],
            'Web Developer': ['web developer', 'frontend developer', 'full stack', 'ui developer', 'react developer'],
            'Backend Developer': ['backend developer', 'software engineer', 'api developer', 'java developer'],
            'DevOps Engineer': ['devops', 'site reliability', 'sre', 'cloud engineer', 'infrastructure'],
        }
        
        def categorize_job(title):
            title_lower = str(title).lower()
            for category, keywords in TECH_CATEGORIES.items():
                for keyword in keywords:
                    if keyword in title_lower:
                        return category
            return 'Other'
        
        # Apply role normalization
        def normalize_role(role_name):
            role_lower = str(role_name).lower()
            if 'analyst' in role_lower or 'sql analyst' in role_lower:
                return 'Data Analyst'
            if 'data scientist' in role_lower:
                return 'Data Scientist'
            return role_name
        
        df['job_category'] = df['job_title'].apply(categorize_job)
        
        # Apply role normalization
        df['job_category'] = df['job_category'].apply(normalize_role)
        
        df_tech = df[df['job_category'] != 'Other'].copy()
        
        print(f"   Categorized {len(df_tech)} tech jobs:")
        print(df_tech['job_category'].value_counts())
        
        return df_tech
    
    def _build_skill_mappings(self, df: pd.DataFrame) -> None:
        """
        Build role → skill frequency mapping.
        
        How it works:
        -------------
        1. Group all job postings by role (e.g., all "Web Developer" jobs)
        2. Extract all skills from those postings
        3. Count frequency of each skill
        4. Store as: {'Web Developer': {'javascript': 847, 'react': 623, ...}}
        
        This tells us: "Out of 1000 Web Developer jobs, 847 required JavaScript"
        
        Args:
            df: Categorized job dataset
        """
        print("\n   Building skill-frequency mappings...")
        
        # Parse skills column
        if 'job_skills' in df.columns:
            df['skills_list'] = df['job_skills'].apply(
                lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()]
            )
        elif 'skills_list' in df.columns:
            # Already parsed
            pass
        else:
            print("   ⚠️  No skills column found")
            return
        
        # Group by role and count skills
        for role in df['job_category'].unique():
            if role == 'Other':
                continue
            
            role_jobs = df[df['job_category'] == role]
            all_skills = []
            
            for skills in role_jobs['skills_list']:
                if isinstance(skills, list):
                    all_skills.extend(skills)
            
            # Count frequencies
            skill_counts = Counter(all_skills)
            self.role_skill_map[role] = dict(skill_counts)
        
        print(f"   ✓ Built skill mappings for {len(self.role_skill_map)} roles")
    
    def _build_company_mappings(self, df: pd.DataFrame) -> None:
        """
        Build role → company frequency mapping.
        
        How it works:
        -------------
        1. Group all job postings by role
        2. Extract company names
        3. Count which companies hire most for each role
        4. Store as: {'Web Developer': {'Google': 45, 'Meta': 32, ...}}
        
        This tells us: "Google posted 45 Web Developer jobs in our dataset"
        
        Args:
            df: Categorized job dataset
        """
        print("\n   Building company-frequency mappings...")
        
        # Find company column
        company_col = None
        for col in ['company_name', 'company', 'employer', 'organization']:
            if col in df.columns:
                company_col = col
                break
        
        if not company_col:
            print("   ⚠️  No company column found")
            return
        
        # Group by role and count companies
        for role in df['job_category'].unique():
            if role == 'Other':
                continue
            
            role_jobs = df[df['job_category'] == role]
            companies = role_jobs[company_col].dropna().tolist()
            
            # Count frequencies
            company_counts = Counter(companies)
            self.role_company_map[role] = dict(company_counts)
        
        print(f"   ✓ Built company mappings for {len(self.role_company_map)} roles")
    
    def _build_salary_mappings(self, df: pd.DataFrame) -> None:
        """
        Build role → salary range mapping.
        
        How it works:
        -------------
        1. Group jobs by role
        2. Extract salary information (if available)
        3. Compute min, max, average per role
        4. Store as: {'Web Developer': {'min': 75000, 'max': 130000, 'avg': 95000}}
        
        Note: Salary data is often missing in job postings.
        If unavailable, we'll use estimates based on market research.
        
        Args:
            df: Categorized job dataset
        """
        print("\n   Building salary mappings...")
        
        # Find salary column(s)
        salary_cols = []
        for col in ['salary', 'min_salary', 'max_salary', 'base_salary', 'compensation']:
            if col in df.columns:
                salary_cols.append(col)
        
        if not salary_cols:
            print("   ⚠️  No salary column found - using market estimates")
            # Use fallback estimates
            self.role_salary_map = self._get_salary_estimates()
            return
        
        # Extract salary data
        for role in df['job_category'].unique():
            if role == 'Other':
                continue
            
            role_jobs = df[df['job_category'] == role]
            
            # Try to extract numeric salary values
            salaries = []
            for col in salary_cols:
                if col in role_jobs.columns:
                    sal_values = pd.to_numeric(role_jobs[col], errors='coerce').dropna()
                    salaries.extend(sal_values.tolist())
            
            if salaries:
                self.role_salary_map[role] = {
                    'min': int(min(salaries)),
                    'max': int(max(salaries)),
                    'avg': int(np.mean(salaries))
                }
            else:
                # Fallback to estimates
                self.role_salary_map[role] = self._get_salary_estimates().get(role, {})
        
        print(f"   ✓ Built salary mappings for {len(self.role_salary_map)} roles")
    
    def _get_salary_estimates(self) -> Dict:
        """
        Provide salary estimates based on US market research (2024).
        
        Why estimates?
        --------------
        Job postings often don't include salary data. We use industry averages
        as fallback to ensure the system always provides useful information.
        
        Returns:
            Dictionary of salary estimates per role
        """
        return {
            'Web Developer': {'min': 75000, 'max': 130000, 'avg': 95000},
            'Data Scientist': {'min': 95000, 'max': 160000, 'avg': 120000},
            'Data Engineer': {'min': 90000, 'max': 150000, 'avg': 115000},
            'Backend Developer': {'min': 80000, 'max': 140000, 'avg': 105000},
            'DevOps Engineer': {'min': 85000, 'max': 145000, 'avg': 110000},
            'AI Engineer': {'min': 100000, 'max': 180000, 'avg': 135000}
        }
    
    def _generate_role_descriptions(self) -> None:
        """
        Generate dynamic role descriptions based on dataset patterns.
        
        Why dynamic descriptions?
        -------------------------
        Instead of hardcoding what each role does, we infer it from:
        - Most common skills (indicates core responsibilities)
        - Top companies (indicates industry focus)
        
        Returns:
            Dictionary of role descriptions
        """
        for role in self.role_skill_map.keys():
            top_skills = self.get_top_skills(role, top_n=5)
            top_companies = self.get_top_companies(role, top_n=3)
            
            if top_skills:
                skills_str = ', '.join([s.capitalize() for s in top_skills])
                companies_str = ', '.join([c for c, _ in top_companies])
                
                description = (
                    f"Specializes in {skills_str}. "
                    f"Top hiring companies include {companies_str}."
                )
                self.role_descriptions[role] = description
            else:
                self.role_descriptions[role] = f"Tech role focused on {role} responsibilities."
    
    # ============================================================
    # PUBLIC API - GET INSIGHTS
    # ============================================================
    
    def get_top_skills(self, role: str, top_n: int = 10) -> List[str]:
        """
        Get most frequently required skills for a role.
        
        Why frequency-based?
        --------------------
        Skills that appear in more job postings are MORE IMPORTANT.
        This reflects actual employer demand, not personal opinions.
        
        Example:
            If 90% of Web Developer jobs require JavaScript, but only 30% require Sass,
            JavaScript is clearly more essential.
        
        Args:
            role: Job category (e.g., 'Web Developer')
            top_n: Number of top skills to return
            
        Returns:
            List of top N skills sorted by frequency
        """
        if role not in self.role_skill_map:
            return []
        
        skill_counts = self.role_skill_map[role]
        sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, count in sorted_skills[:top_n]]
    
    def get_skill_frequencies(self, role: str) -> Dict[str, int]:
        """
        Get all skills with their frequencies for a role.
        
        Args:
            role: Job category
            
        Returns:
            Dictionary: {skill: frequency}
        """
        return self.role_skill_map.get(role, {})
    
    def get_top_companies(self, role: str, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Get companies that hire most frequently for a role.
        
        Why this approach?
        ------------------
        Companies that post more jobs for a role are:
        1. More likely to hire for that role
        2. Investing more in that skillset
        3. Better targets for job seekers
        
        Args:
            role: Job category
            top_n: Number of top companies to return
            
        Returns:
            List of tuples: [(company, job_count), ...]
        """
        if role not in self.role_company_map:
            return []
        
        company_counts = self.role_company_map[role]
        sorted_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_companies[:top_n]
    
    def get_salary_info(self, role: str) -> Dict:
        """
        Get salary information for a role.
        
        Args:
            role: Job category
            
        Returns:
            Dictionary with min, max, avg salary
        """
        return self.role_salary_map.get(role, {})
    
    def compute_skill_gap(self, role: str, extracted_skills: List[str]) -> Dict:
        """
        Compute skill gap between resume and job market requirements.
        
        How it works:
        -------------
        1. Get top 15 required skills from dataset (for this role)
        2. Compare with skills extracted from resume
        3. Identify missing skills
        4. Calculate match percentage
        5. Prioritize missing skills by frequency (most important first)
        
        Why top 15 skills?
        -------------------
        Too few (5): Misses important secondary skills
        Too many (50): Overwhelming, includes rarely-needed skills
        15 is the sweet spot: Covers essentials + important preferred skills
        
        Args:
            role: Predicted job category
            extracted_skills: Skills found in resume
            
        Returns:
            Dictionary with skill gap analysis
        """
        if role not in self.role_skill_map:
            return None
        
        # Get top 15 required skills (with frequencies)
        required_skills_dict = self.get_skill_frequencies(role)
        top_required = sorted(required_skills_dict.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Convert extracted skills to lowercase for comparison
        extracted_lower = [s.lower() for s in extracted_skills]
        
        # Find missing skills
        missing_skills = []
        matched_count = 0
        
        for skill, frequency in top_required:
            if skill not in extracted_lower:
                missing_skills.append({
                    'skill': skill,
                    'frequency': frequency,
                    'importance': 'essential' if frequency > len(required_skills_dict) * 0.5 else 'preferred'
                })
            else:
                matched_count += 1
        
        # Calculate match percentage
        total_required = len(top_required)
        match_percentage = (matched_count / total_required * 100) if total_required > 0 else 0
        
        # Sort missing skills by frequency (most important first)
        missing_skills.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Separate essential vs preferred
        missing_essential = [m['skill'] for m in missing_skills if m['importance'] == 'essential']
        missing_preferred = [m['skill'] for m in missing_skills if m['importance'] == 'preferred']
        
        # Learning priority (top 2 most frequent missing skills)
        learning_priority = [m['skill'] for m in missing_skills[:2]]
        
        return {
            'match_percentage': match_percentage,
            'total_required': total_required,
            'matched': matched_count,
            'missing_essential': missing_essential,
            'missing_preferred': missing_preferred,
            'learning_priority': learning_priority,
            'all_missing': missing_skills
        }
    
    def get_role_description(self, role: str) -> str:
        """
        Get dynamically generated role description.
        
        Args:
            role: Job category
            
        Returns:
            Description string
        """
        return self.role_descriptions.get(role, f"Tech role: {role}")
    
    def format_salary_range(self, role: str) -> str:
        """
        Format salary information as readable string.
        
        Args:
            role: Job category
            
        Returns:
            Formatted salary string (e.g., "$75K - $130K (avg: $95K)")
        """
        salary = self.get_salary_info(role)
        if not salary:
            return "Salary data not available"
        
        if 'avg' in salary:
            return f"${salary['min']//1000}K - ${salary['max']//1000}K (avg: ${salary['avg']//1000}K)"
        else:
            return f"${salary.get('min', 0)//1000}K - ${salary.get('max', 0)//1000}K"


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    """
    Example: Build and test data-driven insights
    """
    print("\n" + "="*80)
    print("DATA-DRIVEN INSIGHTS - DEMONSTRATION")
    print("="*80)
    
    # Initialize
    insights = DataDrivenInsights(data_dir='data_insights')
    
    # Load data (update paths to your actual dataset locations)
    job_skills_path = r"C:\Users\shaik\Downloads\job_skills.csv\job_skills.csv"
    job_postings_path = r"C:\Users\shaik\Downloads\linkedin_job_postings.csv\linkedin_job_postings.csv"
    
    # Build mappings
    insights.load_and_prepare_data(job_skills_path, job_postings_path)
    
    # Test with sample role
    test_role = "Web Developer"
    
    print(f"\n" + "="*80)
    print(f"TESTING: {test_role}")
    print("="*80)
    
    # Top skills
    print(f"\n📚 Top 10 Required Skills:")
    for i, skill in enumerate(insights.get_top_skills(test_role, 10), 1):
        freq = insights.role_skill_map[test_role][skill]
        print(f"   {i}. {skill} (appears in {freq} job postings)")
    
    # Top companies
    print(f"\n🏢 Top 5 Hiring Companies:")
    for i, (company, count) in enumerate(insights.get_top_companies(test_role, 5), 1):
        print(f"   {i}. {company} ({count} job postings)")
    
    # Salary
    print(f"\n💰 Salary Range:")
    print(f"   {insights.format_salary_range(test_role)}")
    
    # Skill gap example
    print(f"\n🔍 Skill Gap Analysis (sample resume with: react, html, css):")
    sample_skills = ['react', 'html', 'css', 'git']
    gap = insights.compute_skill_gap(test_role, sample_skills)
    
    print(f"   Match: {gap['match_percentage']:.1f}%")
    print(f"   Missing Essential: {gap['missing_essential']}")
    print(f"   Missing Preferred: {gap['missing_preferred']}")
    print(f"   Learning Priority: {gap['learning_priority']}")
