"""
Script to import any Dataset and convert it to JSONL format for RAG indexing.
"""

import argparse
import csv
import json
import hashlib
import time
from typing import Dict, Any, List
from pathlib import Path


def make_id(text: str) -> str:
    """Generate a unique ID from text"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def format_field_value(value: Any) -> str:
    """
    Format a field value for display.
    Handles lists, dicts, and strings appropriately.
    """
    if value is None:
        return ""
    
    if isinstance(value, list):
        # Format list items
        formatted_items = []
        for i, item in enumerate(value, 1):
            if isinstance(item, str):
                formatted_items.append(f"  {i}. {item}")
            elif isinstance(item, dict):
                formatted_items.append(f"  {i}. {json.dumps(item, ensure_ascii=False)}")
            else:
                formatted_items.append(f"  {i}. {str(item)}")
        return "\n".join(formatted_items)
    
    if isinstance(value, dict):
        # Format dict as key-value pairs
        formatted_items = []
        for key, val in value.items():
            formatted_items.append(f"  {key}: {val}")
        return "\n".join(formatted_items)
    
    return str(value)


def format_field_name(field_name: str) -> str:
    """
    Format field name for display (capitalize, replace underscores).
    """
    # Replace underscores with spaces
    formatted = field_name.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter of each word
    words = formatted.split()
    formatted = ' '.join(word.capitalize() for word in words)
    return formatted


def format_command_entry(entry: Dict[str, Any]) -> str:
    """
    Format a command entry into a readable text representation.
    Dynamically uses all fields from the entry, regardless of their names.
    """
    parts = []
    
    # Fields to skip (metadata/internal fields)
    skip_fields = {'id', 'source_url', 'meta', 'source', 'import_ts'}
    
    # Special handling for common fields (prioritize them)
    priority_fields = ['command', 'name', 'title', 'question', 'category', 'type']
    other_fields = []
    
    # Separate priority and other fields
    for key in entry.keys():
        if key not in skip_fields and entry[key] is not None:
            if key.lower() in priority_fields:
                continue  # Will handle separately
            other_fields.append(key)
    
    # Handle priority fields first
    handled_fields = set()
    
    # Command/Name/Title/Question (main identifier)
    for key in ['command', 'name', 'title', 'question']:
        if key in entry and entry[key]:
            value = entry[key]
            if isinstance(value, (str, int, float)):
                parts.append(f"{format_field_name(key)}: {value}")
                handled_fields.add(key)
                break
    
    # Category/Type
    for key in ['category', 'type', 'group']:
        if key in entry and entry[key]:
            value = entry[key]
            if isinstance(value, (str, int, float)):
                parts.append(f"{format_field_name(key)}: {value}")
                handled_fields.add(key)
                break
    
    if parts:  # Add separator if we added priority fields
        parts.append("")
    
    # Handle all other fields dynamically
    for key in sorted(entry.keys()):
        if key in skip_fields or key in handled_fields:
            continue
        
        value = entry[key]
        if value is None or value == "":
            continue
        
        # Format field name
        field_label = format_field_name(key)
        
        # Format field value
        formatted_value = format_field_value(value)
        
        if not formatted_value:
            continue
        
        # Add field
        parts.append(f"{field_label}:")
        
        # If value is a list or dict, it's already formatted with indentation
        if isinstance(value, (list, dict)):
            parts.append(formatted_value)
        else:
            # For simple values, add them directly
            parts.append(formatted_value)
        
        parts.append("")
    
    return "\n".join(parts)


def convert_entry_to_rag_format(entry: Dict[str, Any], source: str = "dataset") -> Dict[str, Any]:
    """
    Convert a dataset entry to RAG system format.
    
    Args:
        entry: Dictionary with entry data (can be any structure)
        source: Source identifier for the dataset
    
    Returns:
        Dictionary in RAG format (id, source_url, title, text, meta)
    """
    # Try to find a main identifier field (for title and ID generation)
    main_id_field = None
    main_id_value = None
    
    # Priority order for finding main identifier
    id_fields = ['command', 'name', 'title', 'question', 'id', 'key']
    for field in id_fields:
        if field in entry and entry[field]:
            main_id_field = field
            main_id_value = str(entry[field])
            break
    
    # If no main identifier found, use first non-empty field or generate from content
    if not main_id_value:
        for key, value in entry.items():
            if value and isinstance(value, (str, int, float)) and key not in ['source', 'meta']:
                main_id_field = key
                main_id_value = str(value)[:50]  # Limit length
                break
    
    # Generate unique ID
    if main_id_value:
        doc_id = make_id(f"{source}_{main_id_value}")
    else:
        # Fallback: hash the entire entry
        entry_str = json.dumps(entry, sort_keys=True)
        doc_id = make_id(f"{source}_{entry_str}")
    
    # Create title from main identifier or first field
    if main_id_value:
        if main_id_field == 'question':
            # For Q&A datasets, use question as title (truncated if too long)
            title = main_id_value[:100] + ("..." if len(main_id_value) > 100 else "")
        elif main_id_field in ['command', 'name', 'title']:
            title = f"{main_id_field.capitalize()}: {main_id_value}"
        else:
            title = f"{format_field_name(main_id_field)}: {main_id_value}"
    else:
        title = f"Entry from {source}"
    
    # Format the full text using all fields
    text = format_command_entry(entry)
    
    # Create metadata - include all relevant fields
    meta = {
        "source": source,
        "import_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "original_len_words": len(text.split())
    }
    
    # Add main identifier to metadata
    if main_id_field:
        meta['main_field'] = main_id_field
    
    # Add category/type if available
    for cat_field in ['category', 'type', 'group', 'topic']:
        if cat_field in entry and entry[cat_field]:
            meta[cat_field] = entry[cat_field]
            break
    
    # Add source reference if available
    if entry.get('source'):
        meta['dataset_source'] = entry.get('source')
    
    # For Q&A datasets, add question/solution flags
    if 'question' in entry:
        meta['has_question'] = True
    if 'solution' in entry:
        meta['has_solution'] = True
    
    # Generate virtual URL
    if main_id_value:
        # Sanitize for URL
        url_safe_id = main_id_value.replace(' ', '_').replace('/', '_')[:50]
        source_url = f"{source}://entry/{url_safe_id}"
    else:
        source_url = f"{source}://entry/{doc_id}"
    
    return {
        "id": doc_id,
        "source_url": source_url,
        "title": title,
        "text": text,
        "meta": meta
    }

def load_parquet_dataset(file_path: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(file_path)
    df = table.to_pandas()

    entries = df.to_dict(orient="records")
    print(f"Loaded {len(entries)} entries from Parquet file")
    return entries


def parse_csv_value(value: str) -> Any:
    """
    Parse a CSV cell value, handling JSON arrays/objects and simple strings.
    
    Args:
        value: Raw string value from CSV
    
    Returns:
        Parsed value (list, dict, or string)
    """
    if not value or not value.strip():
        return None
    
    value = value.strip()
    
    # Try to parse as JSON (for arrays like examples)
    if value.startswith('[') or value.startswith('{'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    # If it looks like a comma-separated list (but not JSON), split it
    if ',' in value and not value.startswith('['):
        # Check if it's a simple list
        parts = [p.strip() for p in value.split(',')]
        if len(parts) > 1:
            return parts
    
    return value


def load_csv_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from CSV file.
    
    Automatically detects column names and converts rows to dictionaries.
    Handles examples field as JSON array or comma-separated values.
    """
    entries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        sample = f.read(1024)
        f.seek(0)
        
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
        except:
            delimiter = ','  # Default to comma
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        for row_num, row in enumerate(reader, 2):  # Start from 2 (header is row 1)
            # Convert all values, handling JSON and lists
            entry = {}
            for key, value in row.items():
                if key:  # Skip empty column names
                    # Normalize key name (remove spaces, lowercase)
                    normalized_key = key.strip().lower().replace(' ', '_')
                    parsed_value = parse_csv_value(value)
                    if parsed_value is not None:
                        entry[normalized_key] = parsed_value
            
            if entry:  # Only add non-empty entries
                entries.append(entry)
    
    print(f"Loaded {len(entries)} entries from CSV file")
    return entries


def load_dataset_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a file (JSON, JSONL, or CSV format).
    
    Automatically detects file format based on extension and content.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    entries = []
    file_ext = file_path.suffix.lower()
    
    # Check if it's CSV
    if file_ext == '.csv':
        return load_csv_dataset(file_path)
    
    # Try JSON/JSONL formats
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to parse as JSON first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    entries = data
                elif isinstance(data, dict) and 'entries' in data:
                    entries = data['entries']
                elif isinstance(data, dict) and 'commands' in data:
                    entries = data['commands']
                else:
                    raise ValueError("Unsupported JSON structure")
                print(f"Loaded {len(entries)} entries from JSON file")
            except json.JSONDecodeError:
                # Not valid JSON, try JSONL
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                print(f"Loaded {len(entries)} entries from JSONL file")
    except UnicodeDecodeError:
        # If it's not text, might be CSV with different encoding
        if file_ext == '.csv' or not file_ext:
            # Try CSV anyway
            return load_csv_dataset(file_path)
        if file_ext == '.parquet':
            return load_parquet_dataset(file_path)
        else:
            raise ValueError(f"Could not decode file {file_path}. Is it a valid text file?")
    
    return entries


def download_dataset_from_url(url: str, output_path: str) -> str:
    """
    Download dataset from URL and save to file.
    
    Returns:
        Path to downloaded file
    """
    import requests
    
    print(f"Downloading dataset from {url}...")
    response = requests.get(url, timeout=30, headers={"User-Agent": "RAG-dataset-importer/1.0"})
    response.raise_for_status()
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Dataset saved to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description='Import Dataset and convert to RAG JSONL format'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input dataset file (JSON, JSONL, Parquet, or CSV format). If not provided, will try to download from URL or search default locations.'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='URL to download dataset from (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/dataset.jsonl',
        help='Output JSONL file path (default: data/dataset.jsonl)'
    )
    parser.add_argument(
        '--source-name',
        type=str,
        default='dataset',
        help='Source identifier for metadata (default: dataset)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate entries before converting (check required fields)'
    )
    
    args = parser.parse_args()
    
    # Load dataset
    dataset_file = args.input
    
    if args.url and not args.input:
        # Download from URL
        dataset_file = download_dataset_from_url(
            args.url,
            'data/dataset_raw.json'
        )
    elif not args.input:
        print("Error: No input file specified and no default file found.")
        print("Please provide --input or --url argument.")
        print("Supported formats: JSON, JSONL, CSV")
        return
    
    # Load entries
    try:
        entries = load_dataset_from_file(dataset_file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if not entries:
        print("Warning: No entries found in dataset")
        return
    
    print(f"Processing {len(entries)} entries...")
    
    # Convert entries
    converted = []
    skipped = 0
    
    for i, entry in enumerate(entries, 1):
        if args.validate:
            # Check required fields
            if not entry.get('command'):
                print(f"Warning: Entry {i} missing 'command' field, skipping")
                skipped += 1
                continue
        
        try:
            converted_entry = convert_entry_to_rag_format(entry, args.source_name)
            converted.append(converted_entry)
        except Exception as e:
            print(f"Warning: Error converting entry {i}: {e}")
            skipped += 1
            continue
    
    # Save to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in converted:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\n✓ Successfully converted {len(converted)} entries")
    if skipped > 0:
        print(f"  Skipped {skipped} entries due to errors")
    print(f"✓ Output saved to: {output_path}")
    print(f"\nTo build index from this dataset, run:")
    print(f"  python main.py --mode build --data {output_path} --index zygai_commands_index")

if __name__ == "__main__":
    main()

