import pandas as pd
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils_chem import standardize_and_identify

# Define paths
data_raw_path = Path("data_raw/cornelissen_table.csv")
data_processed_dir = Path("data_processed")
results_path = Path("results/cornelissen_ingest_report.json")

# Ensure output directories exist
data_processed_dir.mkdir(parents=True, exist_ok=True)
results_path.parent.mkdir(parents=True, exist_ok=True)

# Define tasks
tasks = {
    'bbb': 'Status BBB',
    'efflux': 'Status Efflux',
    'influx': 'Status Influx',
    'pampa': 'Status PAMPA'
}

# Read raw data
print("Reading cornelissen_table.csv...")
df_raw = pd.read_csv(data_raw_path)

raw_rows = len(df_raw)
print(f"Raw rows: {raw_rows}")

# Process each task
task_reports = {}

for task_name, label_col in tasks.items():
    print(f"\n{'='*50}")
    print(f"Processing task: {task_name.upper()}")
    print(f"{'='*50}")
    
    # Filter rows where label is not null
    df_task = df_raw[df_raw[label_col].notna()].copy()
    rows_with_label = len(df_task)
    print(f"Rows with label: {rows_with_label}")
    
    # Standardize molecules
    print("Standardizing molecules...")
    results = []
    invalid_dropped = 0
    
    for idx, row in df_task.iterrows():
        parent_smiles = row.get('ParentSmiles')
        label = row.get(label_col)
        
        if pd.isna(parent_smiles):
            invalid_dropped += 1
            continue
        
        mid = standardize_and_identify(parent_smiles, None)
        
        if mid is None:
            invalid_dropped += 1
            continue
        
        # Convert label to int
        try:
            y_value = int(label)
        except (ValueError, TypeError):
            invalid_dropped += 1
            continue
        
        results.append({
            'canonical_smiles': mid.canonical_smiles,
            'inchikey': mid.inchikey,
            'y': y_value
        })
    
    standardized_rows = len(results)
    print(f"Standardized rows: {standardized_rows}")
    print(f"Invalid dropped: {invalid_dropped}")
    
    # Build dataframe
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print(f"Warning: No valid data for task {task_name}")
        task_reports[task_name] = {
            'rows_with_label': rows_with_label,
            'invalid_dropped': invalid_dropped,
            'standardized_rows': standardized_rows,
            'conflict_inchikeys_removed': 0,
            'final_rows': 0,
            'class_balance': {}
        }
        continue
    
    # Remove label conflicts: if an inchikey appears with both y=0 and y=1, drop that inchikey entirely
    print("Removing label conflicts...")
    inchikey_counts = df.groupby('inchikey')['y'].nunique()
    conflict_inchikeys = inchikey_counts[inchikey_counts > 1].index.tolist()
    conflict_inchikeys_removed = len(conflict_inchikeys)
    
    if conflict_inchikeys:
        print(f"Found {conflict_inchikeys_removed} inchikeys with conflicting labels")
        df = df[~df['inchikey'].isin(conflict_inchikeys)]
    
    # Deduplicate by inchikey (keep first)
    print("Deduplicating by inchikey...")
    df = df.drop_duplicates(subset=['inchikey'], keep='first')
    
    # Add source_dataset column
    df['source_dataset'] = 'Cornelissen'
    
    # Reorder columns
    df = df[['canonical_smiles', 'inchikey', 'y', 'source_dataset']]
    
    final_rows = len(df)
    print(f"Final rows: {final_rows}")
    
    # Calculate class balance
    class_balance = df['y'].value_counts().to_dict()
    class_balance = {str(k): int(v) for k, v in class_balance.items()}
    
    # Save processed data
    output_path = data_processed_dir / f"cornelissen_{task_name}.csv"
    print(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    
    # Store report for this task
    task_reports[task_name] = {
        'rows_with_label': rows_with_label,
        'invalid_dropped': invalid_dropped,
        'standardized_rows': standardized_rows,
        'conflict_inchikeys_removed': conflict_inchikeys_removed,
        'final_rows': final_rows,
        'class_balance': class_balance
    }

# Create overall report
report = {
    'raw_rows': raw_rows,
    'tasks': task_reports
}

# Save report
print(f"\n{'='*50}")
print(f"Saving report to {results_path}...")
with open(results_path, 'w') as f:
    json.dump(report, f, indent=2)

# Print report to stdout
print("\n" + "="*50)
print("INGESTION REPORT")
print("="*50)
print(json.dumps(report, indent=2))
print("="*50)
