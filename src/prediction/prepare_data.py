import pandas as pd
import numpy as np
import os

def prepare_and_save_data(input_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Drop Leaked Data
    leaked_cols = ['predicted_score', 'overall_avg', 'science_math_diff', 'eng_hist_ratio']
    df = df.drop(columns=leaked_cols)
    print(f"Dropped leaked columns: {leaked_cols}")
    
    # 2. Reshape (Melt) the dataset
    # We want one row per subject. 
    target_cols = ['math_score', 'science_score', 'english_score', 'history_score']
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    print("Reshaping dataset (1 row per student -> 1 row per subject)...")
    melted_df = pd.melt(
        df, 
        id_vars=feature_cols, 
        value_vars=target_cols,
        var_name='subject', 
        value_name='target_score'
    )
    
    # Clean up subject names (e.g., 'math_score' -> 'Maths')
    subject_map = {
        'math_score': 'Maths',
        'science_score': 'Science',
        'english_score': 'English',
        'history_score': 'History'
    }
    melted_df['subject'] = melted_df['subject'].map(subject_map)
    
    # 3. Create the ordinal difficulty indicator from knowledge_gap
    # The handbook recommends mapping a difficulty indicator to 1 (low), 2 (medium), 3 (high)
    melted_df['difficulty_indicator'] = pd.qcut(melted_df['knowledge_gap'], q=3, labels=[1, 2, 3]).astype(int)
    melted_df = melted_df.drop(columns=['knowledge_gap'])
    
    # 4. Generate Synthetic Confidence Column (Seed = 42 for reproducibility)
    print("Generating synthetic confidence column...")
    np.random.seed(42)
    
    # Create a systematic bias per student to ensure persistent overconfidence/anxiety
    unique_students = melted_df['student_id'].unique()
    # Bias is standard normal scaled by 2 (creates shifts of up to +/- 4-6 confidence points)
    student_bias = {sid: np.random.normal(loc=0, scale=2.0) for sid in unique_students}
    
    # Apply generation rule: (true score / 10) + student bias + random noise
    def generate_confidence(row):
        base_conf = row['target_score'] / 10.0
        bias = student_bias[row['student_id']]
        noise = np.random.normal(loc=0, scale=0.5)
        raw_conf = base_conf + bias + noise
        return np.clip(np.round(raw_conf), 1, 10)
    
    melted_df['confidence'] = melted_df.apply(generate_confidence, axis=1)
    
    # Calculate miscalibration fraction for the report
    gaps = (melted_df['confidence'] * 10) - melted_df['target_score']
    overconfident = (gaps > 20).sum() / len(melted_df)
    anxious = (gaps < -20).sum() / len(melted_df)
    print(f"Synthetic Column Stats - Overconfident: {overconfident*100:.1f}%, Anxious: {anxious*100:.1f}%")
    
    # Save the cleaned dataset
    melted_df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to: {output_path}")

if __name__ == "__main__":
    # Assuming script is run from the root directory
    input_file = "Education_numerical.csv"
    output_file = os.path.join("data", "cleaned_features.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in the root directory.")
    else:
        prepare_and_save_data(input_file, output_file)