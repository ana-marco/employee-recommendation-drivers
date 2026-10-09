"""
Employee recommendation drivers - shared utilities
==================================================

Shared constants, plotting configuration, and helper functions
used across all project notebooks.

Usage:
    import sys
    sys.path.insert(0, "../src")
    from utils import (
        save_fig, set_plot_style, backup_notebook,
        SEED, FIG_DIR, TARGET_COLORS, TARGET_ORDER
    )
    
    set_plot_style()
"""

import os
import glob as glob_module
import matplotlib.pyplot as plt
import seaborn as sns
import re


# ============================================================
# CONSTANTS
# ============================================================

SEED = 42
"""Random seed for reproducibility across all notebooks."""

FIG_DIR = "../figures"
"""Default directory for saving figures."""

DATA_DIR = "../data/processed"
"""Directory containing processed train/val/test splits."""

# Semantic colours for target variable
TARGET_COLORS = {
    "Positive": "#2ecc71",      # Green
    "No Opinion": "#95a5a6",    # Grey
    "Negative": "#e74c3c"       # Red
}
"""Colour mapping for 3-class target visualisations."""

TARGET_ORDER = ["Positive", "No Opinion", "Negative"]
"""Consistent ordering for target class displays."""

# Sequential colourmap for non-target visuals
SEQUENTIAL_CMAP = "Blues"
"""Default colourmap for heatmaps and sequential data."""


# ============================================================
# PLOT STYLING
# ============================================================

def set_plot_style():
    """
    Apply minimal, data-focused plot styling.
    Follows data-to-viz.com / Tufte principles:
    - High data-ink ratio
    - No unnecessary gridlines
    - Clean spines (left + bottom only)
    """
    sns.set_style("white")  # No grid by default
    plt.rcParams.update({
        # Figure and saving
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        
        # Minimal spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Typography
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        
        # No grid by default
        "axes.grid": False,
    })

RATING_LABELS = {
    "overall_rating": "Overall Rating",
    "work_life_balance": "Work-Life Balance",
    "culture_values": "Culture & Values",
    "diversity_inclusion": "Diversity & Inclusion",
    "career_opp": "Career Opportunities",
    "comp_benefits": "Compensation & Benefits",
    "senior_mgmt": "Senior Management"
}
"""Human-readable labels for sub-rating columns."""

# ============================================================
# FIGURE SAVING
# ============================================================

def save_fig(name, fig_dir=None):
    """
    Save current figure with auto-incrementing version number.
    
    Saves two copies:
    - {name}_v{N}.png — versioned backup
    - {name}.png — latest version (overwrites)
    
    Parameters
    ----------
    name : str
        Base name for the figure (without extension).
    fig_dir : str, optional
        Directory to save figures. Defaults to FIG_DIR ("../figures").
    
    Examples
    --------
    >>> plt.plot([1, 2, 3])
    >>> save_fig("my_plot")
    Saved: my_plot_v1.png
    
    >>> # Running again increments version
    >>> save_fig("my_plot")
    Saved: my_plot_v2.png
    """
    if fig_dir is None:
        fig_dir = FIG_DIR
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Find existing versions
    existing = glob_module.glob(os.path.join(fig_dir, f"{name}_v*.png"))
    versions = []
    for f in existing:
        try:
            version_str = os.path.basename(f).replace(f"{name}_v", "").replace(".png", "")
            versions.append(int(version_str))
        except ValueError:
            continue
    
    # Increment version
    v = max(versions) + 1 if versions else 1
    
    # Save versioned and latest
    plt.savefig(os.path.join(fig_dir, f"{name}_v{v}.png"), bbox_inches="tight")
    # plt.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
    
    print(f"Saved: {name}_v{v}.png")


# ============================================================
# NOTEBOOK BACKUP
# ============================================================

def backup_notebook(notebook_path, backup_dir="../backups"):
    """
    Save timestamped backup of a Jupyter notebook.
    
    Creates a copy with format: {notebook_name}_{YYYYMMDD}.ipynb
    
    Parameters
    ----------
    notebook_path : str
        Path to the notebook file to backup.
    backup_dir : str, optional
        Directory to store backups. Defaults to "../backups".
    
    Examples
    --------
    >>> backup_notebook("05_tuning_evaluation.ipynb")
    Backup saved: 05_tuning_evaluation_20260228.ipynb
    """
    import shutil
    from datetime import datetime
    
    os.makedirs(backup_dir, exist_ok=True)
    
    name = os.path.basename(notebook_path).replace(".ipynb", "")
    timestamp = datetime.now().strftime("%Y%m%d")
    backup_name = f"{name}_{timestamp}.ipynb"
    
    shutil.copy(notebook_path, os.path.join(backup_dir, backup_name))
    print(f"Backup saved: {backup_name}")


# ============================================================
# DATA LOADING (optional convenience function)
# ============================================================

def load_processed_data(data_dir=None, include_test=False):
    """
    Load processed train/val(/test) splits and topic features.
    
    Parameters
    ----------
    data_dir : str, optional
        Directory containing processed CSVs and .npy files.
        Defaults to DATA_DIR ("../data/processed").
    include_test : bool, optional
        If True, also load test set. Default False to prevent
        accidental test set access during development.
    
    Returns
    -------
    dict
        Dictionary with keys: 'train', 'val', ('test'),
        each containing {'df': DataFrame, 'topics': ndarray}.
    
    Examples
    --------
    >>> data = load_processed_data()
    >>> X_train = data['train']['df']
    >>> topic_train = data['train']['topics']
    
    >>> # For final evaluation only:
    >>> data = load_processed_data(include_test=True)
    """
    import pandas as pd
    import numpy as np
    
    if data_dir is None:
        data_dir = DATA_DIR
    
    result = {}
    
    # Load train
    result['train'] = {
        'df': pd.read_csv(os.path.join(data_dir, 'train.csv')),
        'topics': np.load(os.path.join(data_dir, 'train_topic_features.npy'))
    }
    
    # Load val
    result['val'] = {
        'df': pd.read_csv(os.path.join(data_dir, 'val.csv')),
        'topics': np.load(os.path.join(data_dir, 'val_topic_features.npy'))
    }
    
    # Optionally load test
    if include_test:
        result['test'] = {
            'df': pd.read_csv(os.path.join(data_dir, 'test.csv')),
            'topics': np.load(os.path.join(data_dir, 'test_topic_features.npy'))
        }
    
    # Print summary
    for split_name, split_data in result.items():
        df_shape = split_data['df'].shape
        topic_shape = split_data['topics'].shape
        print(f"{split_name.capitalize():6s}: {df_shape[0]:,} rows, "
              f"{df_shape[1]} cols + {topic_shape[1]} topics")
    
    return result

# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_text(series):
    """Clean a text Series: newlines → space, lowercase, strip, collapse spaces, fill NaN."""
    return (
        series
        .fillna("")
        .str.replace(r"\r\n|\n", " ", regex=True)
        .str.lower()
        .str.strip()
        .str.replace(r" {2,}", " ", regex=True)
    )


def get_company_patterns(firms_raw, abbreviations=None):
    """
    Build regex pattern to match company names in text.
    
    Parameters
    ----------
    firms_raw : array-like
        Unique firm names from dataset
    abbreviations : list, optional
        Additional company abbreviations to remove
        
    Returns
    -------
    str
        Compiled regex pattern for company name matching
    """
    import re
    
    if abbreviations is None:
        abbreviations = [
            'hsbc', 'bcg', 'gsk', 'pwc', 'ey', 'kpmg', 'ibm', 'jp', 'ms', 'gt', 'gs',
            'amex', 'bain', 'mckinsey', 'deloitte', 'accenture', 'capgemini',
            'cisco', 'oracle', 'sap', 'salesforce', 'microsoft', 'google', 'apple',
            'amazon', 'aws', 'azure', 'meta', 'facebook', 'netflix', 'uber',
            'nhs', 'bbc', 'ikea', 'aldi', 'lidl', 'tesco', 'asda', 'morrisons'
        ]
    
    firm_names = []
    for firm in firms_raw:
        name = str(firm).lower().strip()
        firm_names.append(name)
        name_spaces = name.replace('-', ' ')
        if name_spaces != name:
            firm_names.append(name_spaces)
    
    # Sort longest-first so "jp morgan" matches before "jp"
    all_to_remove = sorted(set(list(firm_names) + abbreviations), key=len, reverse=True)
    pattern = r'\b(' + '|'.join(re.escape(name) for name in all_to_remove) + r")('?s?)?\b"
    
    return pattern


def remove_companies(series, company_pattern):
    """Remove company names from text Series, then collapse whitespace."""
    import re
    return (
        series
        .str.replace(company_pattern, '', regex=True, flags=re.IGNORECASE)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

# ============================================================
# FEATURE GROUPS
# ============================================================

OVERALL_RATING_COL = "overall_rating"
"""Holistic 1-5 rating — keep separate for ablation testing."""

SUB_RATING_COLS = [
    "work_life_balance",
    "culture_values",
    "career_opp",
    "comp_benefits",
    "senior_mgmt",
]
"""Actionable sub-ratings (excludes diversity_inclusion — 84% missing)."""

ALL_RATING_COLS = [OVERALL_RATING_COL] + SUB_RATING_COLS
"""All 6 rating columns combined."""

# ============================================================
# MODEL EVALUATION
# ============================================================

MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
               'XGBoost', 'MLP']
"""Standard model family names for consistent labeling."""
