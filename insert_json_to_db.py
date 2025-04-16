import json
from db import TalentMatchDB
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the JSON file
file_path = "resume_data/batch_5_results.json"


def main():
  # Initialize database connection
  try:
    db = TalentMatchDB(settings.db_params)
    logger.info("Database connection initialized successfully")
  except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}")
    return

  # Load the JSON file
  try:
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as file:
      resumes = json.load(file)

    if not resumes:
      logger.warning("The file is empty or has no data")
      return

    logger.info(f"Successfully loaded {len(resumes)} resume records")

    # Process and insert each resume
    success_count = 0
    error_count = 0

    for i, resume in enumerate(resumes):
      try:
        logger.info(
            f"Processing resume {i+1}/{len(resumes)}: ID {resume.get('id', 'unknown')}")
        db.insert_resume(resume)
        success_count += 1
      except Exception as e:
        logger.error(
            f"Failed to insert resume {resume.get('id', 'unknown')}: {str(e)}")
        error_count += 1

    logger.info(
        f"Insertion complete. Success: {success_count}, Errors: {error_count}")

  except Exception as e:
    logger.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
  main()
