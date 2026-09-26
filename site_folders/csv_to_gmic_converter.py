#!/usr/bin/env python3
"""
Convert CSV format to GMIC pickle format for breast cancer screening
Based on the data preparation notebook and GMIC documentation
"""

import pandas as pd
import pickle
import os
import argparse
from typing import Dict, List, Any
import numpy as np


def create_gmic_cancer_labels(exam_df: pd.DataFrame, use_view_labels: bool = True) -> Dict[str, int]:
    """
    Create GMIC cancer labels for an exam
    
    Args:
        exam_df: DataFrame containing all views for one exam
        use_view_labels: Whether to use view-level labels (True) or exam-level labels (False)
        
    Returns:
        Dictionary with GMIC format cancer labels
    """
    if use_view_labels and 'view_level_label' in exam_df.columns:
        # Use view-level labels to determine breast-side labels
        left_images = exam_df[exam_df['laterality'] == 'L']
        right_images = exam_df[exam_df['laterality'] == 'R']
        
        # Each breast side is malignant if any of its views are positive
        left_malignant = 1 if len(left_images) > 0 and left_images['view_level_label'].max() == 1 else 0
        right_malignant = 1 if len(right_images) > 0 and right_images['view_level_label'].max() == 1 else 0
    else:
        # Fallback to exam-level labels
        exam_label = exam_df['exam_level_label'].max() if 'exam_level_label' in exam_df.columns else exam_df.get('label', 0).max()
        
        # Simple logic: if exam is positive, both sides are positive
        if exam_label == 1:
            left_malignant = 1
            right_malignant = 1
        else:
            left_malignant = 0
            right_malignant = 0
    
    # GMIC format: benign = opposite of malignant
    left_benign = 1 - left_malignant
    right_benign = 1 - right_malignant
    
    # Overall exam labels
    overall_malignant = max(left_malignant, right_malignant)
    overall_benign = 1 - overall_malignant
    
    return {
        'benign': overall_benign,
        'malignant': overall_malignant,
        'left_benign': left_benign,
        'left_malignant': left_malignant,
        'right_benign': right_benign,
        'right_malignant': right_malignant,
        'unknown': 0
    }


def create_enhanced_gmic_cancer_labels(exam_df: pd.DataFrame, tumor_laterality_col: str = None) -> Dict[str, int]:
    """
    Enhanced version that uses view-level labels and tumor laterality information if available
    
    Args:
        exam_df: DataFrame containing all views for one exam
        tumor_laterality_col: Column name containing tumor laterality ('L', 'R', or NaN)
        
    Returns:
        Dictionary with GMIC format cancer labels
    """
    # Check if we have view-level labels
    if 'view_level_label' in exam_df.columns:
        # Use view-level labels directly (most accurate)
        left_images = exam_df[exam_df['laterality'] == 'L']
        right_images = exam_df[exam_df['laterality'] == 'R']
        
        left_malignant = 1 if len(left_images) > 0 and left_images['view_level_label'].max() == 1 else 0
        right_malignant = 1 if len(right_images) > 0 and right_images['view_level_label'].max() == 1 else 0
        
    else:
        # Fallback to exam-level + tumor laterality logic
        exam_label = exam_df['exam_level_label'].max() if 'exam_level_label' in exam_df.columns else exam_df.get('label', 0).max()
        
        # If exam is negative, all sides are negative
        if exam_label == 0:
            left_malignant, right_malignant = 0, 0
        else:
            # If exam is positive, check tumor laterality
            if tumor_laterality_col and tumor_laterality_col in exam_df.columns:
                tumor_side = exam_df[tumor_laterality_col].iloc[0]
                
                if pd.notna(tumor_side):
                    # We know which side has the tumor
                    if tumor_side == 'L':
                        left_malignant, right_malignant = 1, 0
                    elif tumor_side == 'R':
                        left_malignant, right_malignant = 0, 1
                    else:
                        # Unknown tumor side, mark both as positive (conservative)
                        left_malignant, right_malignant = 1, 1
                else:
                    # Tumor laterality unknown, mark both as positive (conservative)
                    left_malignant, right_malignant = 1, 1
            else:
                # No tumor laterality info, mark both as positive (conservative)
                left_malignant, right_malignant = 1, 1
    
    left_benign = 1 - left_malignant
    right_benign = 1 - right_malignant
    overall_malignant = max(left_malignant, right_malignant)
    overall_benign = 1 - overall_malignant
    
    return {
        'benign': overall_benign,
        'malignant': overall_malignant,
        'left_benign': left_benign,
        'left_malignant': left_malignant,
        'right_benign': right_benign,
        'right_malignant': right_malignant,
        'unknown': 0
    }


def extract_filename_from_path(file_path: str) -> str:
    """
    Extract filename without extension from full path
    
    Args:
        file_path: Full path to image file
        
    Returns:
        Filename without extension and directory
    """
    filename = os.path.basename(file_path)
    # Remove common extensions
    for ext in ['.png', '.dcm', '.jpg', '.jpeg', '.tiff', '.tif']:
        if filename.lower().endswith(ext.lower()):
            filename = filename[:-len(ext)]
            break
    return filename


def convert_csv_to_gmic_format(
    df: pd.DataFrame, 
    split_filter: str = None,
    tumor_laterality_col: str = None,
    use_enhanced_labels: bool = False,
    prefer_view_labels: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert dataframe to GMIC exam list format
    
    Args:
        df: Input dataframe
        split_filter: Filter by split (train/dev/test)
        tumor_laterality_col: Column name for tumor laterality
        use_enhanced_labels: Whether to use enhanced labeling logic
        prefer_view_labels: Whether to prefer view-level labels over exam-level labels
        
    Returns:
        List of exam dictionaries in GMIC format
    """
    if split_filter:
        df = df[df['split_group'] == split_filter].copy()
        print(f"Processing {split_filter} split: {len(df)} rows")
    
    exam_list = []
    
    # Create exam key (patient_id + exam_id)
    df['exam_key'] = df['patient_id'].astype(str) + '_' + df['exam_id'].astype(str)
    
    for exam_key, exam_df in df.groupby('exam_key'):
        exam = {}
        
        # Set horizontal flip (from the data or default)
        if 'horizontal_flip' in exam_df.columns:
            exam['horizontal_flip'] = exam_df['horizontal_flip'].iloc[0]
        else:
            exam['horizontal_flip'] = 'NO'
        
        # Add each view
        for _, row in exam_df.iterrows():
            # Create GMIC view name (e.g., "L-CC", "R-MLO")
            gmic_view = f"{row['laterality'].upper()}-{row['view'].upper()}"
            filename = extract_filename_from_path(row['file_path'])
            
            if gmic_view not in exam:
                exam[gmic_view] = []
            exam[gmic_view].append(filename)
        
        # Add cancer labels
        if use_enhanced_labels:
            exam['cancer_label'] = create_enhanced_gmic_cancer_labels(exam_df, tumor_laterality_col)
        else:
            exam['cancer_label'] = create_gmic_cancer_labels(exam_df, use_view_labels=prefer_view_labels)
        
        exam_list.append(exam)
    
    return exam_list


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to GMIC pickle format')
    parser.add_argument('--input-csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output-dir', required=True, help='Directory to save pickle files')
    parser.add_argument('--tumor-laterality-col', help='Column name for tumor laterality (optional)')
    parser.add_argument('--enhanced-labels', action='store_true', 
                       help='Use enhanced labeling logic with tumor laterality')
    parser.add_argument('--prefer-view-labels', action='store_true', default=True,
                       help='Prefer view-level labels over exam-level labels (default: True)')
    parser.add_argument('--split-column', default='split_group', 
                       help='Column name for data splits (default: split_group)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CSV
    print(f"Loading CSV from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Print basic info about the dataset
    print(f"Unique patients: {df['patient_id'].nunique()}")
    if 'exam_level_label' in df.columns:
        print(f"Exam-level label distribution: {df.groupby('patient_id')['exam_level_label'].max().value_counts().to_dict()}")
    if 'view_level_label' in df.columns:
        print(f"View-level label distribution: {df['view_level_label'].value_counts().to_dict()}")
    if 'label' in df.columns:  # Fallback for old format
        print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    if args.split_column in df.columns:
        print(f"Split distribution: {df[args.split_column].value_counts().to_dict()}")
    
    # Convert each split
    splits = df[args.split_column].unique() if args.split_column in df.columns else [None]
    
    for split in splits:
        if split is None:
            # No split column, process all data
            exam_list = convert_csv_to_gmic_format(
                df, 
                tumor_laterality_col=args.tumor_laterality_col,
                use_enhanced_labels=args.enhanced_labels,
                prefer_view_labels=args.prefer_view_labels
            )
            output_path = os.path.join(args.output_dir, "exam_list_all.pkl")
        else:
            exam_list = convert_csv_to_gmic_format(
                df, 
                split_filter=split,
                tumor_laterality_col=args.tumor_laterality_col,
                use_enhanced_labels=args.enhanced_labels,
                prefer_view_labels=args.prefer_view_labels
            )
            output_path = os.path.join(args.output_dir, f"exam_list_{split}.pkl")
        
        # Save to pickle
        with open(output_path, 'wb') as f:
            pickle.dump(exam_list, f)
        
        print(f"Saved {len(exam_list)} exams to {output_path}")
        
        # Print sample exam for verification
        if exam_list:
            print(f"\nSample exam from {split if split else 'all'} split:")
            sample = exam_list[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
    
    print("\nConversion completed!")


def verify_pickle_file(pickle_path: str):
    """
    Utility function to verify and inspect a pickle file
    
    Args:
        pickle_path: Path to pickle file to verify
    """
    print(f"Verifying {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        exam_list = pickle.load(f)
    
    print(f"Number of exams: {len(exam_list)}")
    
    if exam_list:
        print("\nFirst exam structure:")
        sample = exam_list[0]
        for key, value in sample.items():
            print(f"  {key}: {value}")
        
        # Check cancer label distribution
        cancer_labels = [exam['cancer_label'] for exam in exam_list]
        malignant_count = sum(1 for label in cancer_labels if label['malignant'] == 1)
        benign_count = len(cancer_labels) - malignant_count
        
        print(f"\nLabel distribution:")
        print(f"  Malignant: {malignant_count}")
        print(f"  Benign: {benign_count}")


if __name__ == "__main__":
    main()