import json
import pandas as pd

def extract_indicator_maj_N_from_jsonl(file_path, N):
    """
    Reads a JSONL file and extracts the "indicator_maj_N" field from each line,
    along with the line number as "sb_idx".
    
    Args:
        file_path (str): Path to the JSONL file
        N (int): The N value to extract (e.g., 2 for "indicator_maj_2", 4 for "indicator_maj_4")
        
    Returns:
        pd.DataFrame: DataFrame with columns "sb_idx" and "indicator_maj_N"
    """
    data = []
    field_name = f"indicator_maj_{N}"
    
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            try:
                entry = json.loads(line.strip())
                # Extract indicator_maj_N field, default to None if not present
                indicator_maj_N = entry.get(field_name, None)
                
                data.append({
                    "sb_idx": line_num,
                    field_name: indicator_maj_N
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error on line {line_num}: {e}")
                continue
    
    return pd.DataFrame(data)

# Example usage:
# df = extract_indicator_maj_N_from_jsonl("/path/to/your/file.jsonl", 4)
# print(df.head())
# print(f"Total lines processed: {len(df)}") 