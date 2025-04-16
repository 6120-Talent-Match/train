import json
import os
import glob


def combine_json_files(input_dir, output_file):
  """
  Combine multiple JSON files into a single JSON file.

  Args:
      input_dir (str): Directory containing JSON files to combine
      output_file (str): Path for the output combined JSON file
  """
  # Get all JSON files in the directory
  json_files = glob.glob(os.path.join(input_dir, "*.json"))
  print(f"Found {len(json_files)} JSON files in {input_dir}")

  # Initialize an empty list to store all data
  combined_data = []
  filtered_count = 0

  # Process each JSON file
  for file_path in json_files:
    print(f"Processing {file_path}...")
    try:
      with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # If data is a list, filter out entries with "error" field then extend combined_data
        if isinstance(data, list):
          filtered_data = [item for item in data if "error" not in item]
          filtered_count += len(data) - len(filtered_data)
          combined_data.extend(filtered_data)
        # If data is a dictionary and doesn't have "error" field, append it to combined_data
        elif isinstance(data, dict) and "error" not in data:
          combined_data.append(data)
        elif isinstance(data, dict) and "error" in data:
          filtered_count += 1
          print(f"Filtered out an entry with error field in {file_path}")
        else:
          print(f"Unsupported data type in {file_path}. Skipping.")
    except Exception as e:
      print(f"Error processing {file_path}: {e}")

  print(f"Total entries filtered out: {filtered_count}")
  print(f"Total records after combining and filtering: {len(combined_data)}")

  # Save the combined data to the output file
  with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2)

  print(f"Combined data saved to {output_file}")


if __name__ == "__main__":
  input_directory = "resume_data"
  output_file = "combined_resume_data.json"

  combine_json_files(input_directory, output_file)
  print("Done!")
