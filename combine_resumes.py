import json
import os


def process_batch_files():
  # Output file
  output_file = "combined_resume_data.json"

  # Fields to extract
  fields_to_extract = ["id", "category", "skills", "education", "experience"]

  # Combined data
  combined_data = []

  # Process each batch file
  for i in range(1, 5):  # batch_1 to batch_4
    batch_file = f"resume_results/batch_{i}_results.json"

    if not os.path.exists(batch_file):
      print(f"Warning: {batch_file} does not exist. Skipping.")
      continue

    print(f"Processing {batch_file}...")

    try:
      with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)

      # Extract only the specified fields
      for item in batch_data:
        filtered_item = {}
        for field in fields_to_extract:
          if field in item:
            filtered_item[field] = item[field]

        combined_data.append(filtered_item)

      print(
          f"Successfully processed {len(batch_data)} records from {batch_file}")

    except Exception as e:
      print(f"Error processing {batch_file}: {str(e)}")

  # Save combined data
  try:
    with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(combined_data, f, indent=2)

    print(f"Successfully combined data saved to {output_file}")
    print(f"Total records: {len(combined_data)}")

  except Exception as e:
    print(f"Error saving combined data: {str(e)}")


if __name__ == "__main__":
  process_batch_files()
