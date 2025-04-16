import gc
from llama_cpp import Llama
import pandas as pd
import json
import time
from tqdm import tqdm

# Load the model
llm = Llama.from_pretrained(
    repo_id="C0ldSmi1e/resume-reader",
    filename="unsloth.Q8_0.gguf",
    n_ctx=4096,  # Increase the context window
)

prompt_template = """
# Role and Task

You are an experienced HR and now you will review a resume then extract key information from it.

# Input Format

The input is the resume text, and you will review it.

# Output Format

Your response should be ONLY in the following JSON format:
{{
  "skills": array[string],
  "education": array[string],
  "experience": array[string]
}}

Example:
{{
  "skills": [
      "Accounting",
      "ads",
      "advertising",
      "analytical skills",
      "benefits",
      "billing",
      "budgeting",
      "clients",
      "Customer Service",
      "data analysis",
      "delivery",
      "documentation",
      "employee relations",
      "financial management",
      "government relations",
      "Human Resources",
      "insurance",
      "labor relations",
      "layout",
      "Marketing",
      "marketing collateral",
      "medical billing",
      "medical terminology",
      "office",
      "organizational",
      "payroll",
      "performance reviews",
      "personnel",
      "policies",
      "posters",
      "presentations",
      "public relations",
      "purchasing",
      "reporting",
      "statistics",
      "website"
    ],
    "education": [
      "Jefferson College, Business Administration",
      "Sainte Genevieve Senior High, High School Diploma"
    ],
    "experience": [
      "HR Administrator/Marketing Associate",
      "Advanced Medical Claims Analyst",
      "Assistant General Manager",
      "Executive Support / Marketing Assistant",
      "Reservation & Front Office Manager",
      "Owner/ Partner",
      "Price Integrity Coordinator"
    ]
}}

 - For education, only return the school, major and degree, if any field is not availible, ignore.
 - For experience, only return the roles.
 - Your response should be only the JSON object, without other text or explaination
 - If there's any field that you can't recognize, set it as an empty array.

# Input
{}

### Response
<think>
{}
</think>
{}
"""


def process_resume(resume_text):
  try:
    # Format the prompt with the resume text
    formatted_prompt = prompt_template.format(resume_text, "", "")

    # Run the model with the formatted prompt
    output = llm(formatted_prompt, max_tokens=8096,
                 stop=["</s>", "<|im_end|>"])

    # Get the response text
    response_text = output["choices"][0]["text"]

    # Try to parse as JSON (in case the response is not valid JSON)
    try:
      json_response = json.loads(response_text)
      return response_text, True
    except json.JSONDecodeError:
      return response_text, False
  except Exception as e:
    return str(e), False


# Load the CSV file
print("Loading evaluation data...")
eval_data = pd.read_csv("./data/eval.csv")
print(f"Loaded {len(eval_data)} resumes from CSV")

"""
# Create a new DataFrame to store the results
results = []

# Process each resume
print("Processing resumes...")
for i, row in tqdm(eval_data.iterrows(), total=len(eval_data)):
  resume_text = row["text"]
  resume_id = row["id"]

  # Skip empty resumes
  if pd.isna(resume_text) or resume_text.strip() == "":
    continue

  # Process the resume
  response, success = process_resume(resume_text)

  # Add to results
  result = {
      "id": resume_id,
      "response": response,
      "success": success
  }
  results.append(result)

  # Sleep briefly to avoid overloading the system
  time.sleep(0.1)

  # Save incremental results every 10 items
  if (i + 1) % 10 == 0:
    with open(f"./resume_results_partial_{i+1}.json", "w") as f:
      json.dump(results, f, indent=2)
    print(f"Saved partial results for {i+1} resumes")

# Save the final results
with open("./resume_results.json", "w") as f:
  json.dump(results, f, indent=2)

print(
    f"Processed {len(results)} resumes. Results saved to resume_results.json")
"""

# Free model resources
del llm
gc.collect()

# Example of running a single resume for testing
if __name__ == "__main__" and False:  # Set to True to run this test
  resume = """
             MECHANICAL ENGINEER       Summary     5 years and 9 months experience as Mechanical Engineer in the operation and maintenance of boilers, swimming pools, deep well pumping stations and incinerators. 10 years experience as Mechanical Engineer/ Section Head in the operation and maintenance of water treatment plants, sewage treatment plants, sewage lifting stations, deep well pumping stations, swimming pools, raw water pumping and distribution stations. 7 years experience as lead man in the operation and maintenance of Gas Turbine Power Plant. 2 years experience as sewage treatment plant operator. 1.5 years experience as diesel generator set operator. 2 years 9 months experience as merchant ship electrician. 1 year experience as assistant electrician/wiper in merchant ship. 1 year experience as textile weaving supervisor.        Highlights          Pump and piping systems  Motor Control Panel  Operation and maintenance of sewage treatment plants and sewage lifting stations.  Operation and maintenance of Reverse Osmosis plant, with PLC controls  Operation and maintenance of Swimming Pools.  Operation and maintenance of boilers, incinerator, and Gas Turbine Power plant.  Operation and maintenance of Deep well pumping station.      Operation and maintenance of diesel engine driven generator sets power plant.            Accomplishments              Our ship was in trouble when the right terminal shaft of the woodward governor that connects the governor to the injection pumps was broken while we were Somewhat near the Aleutian island in Alaska. The emergency speed of the ship was activated but that is too slow. I suggested to modified the linkage connection by using the left side terminal shaft of the governor. Then we were able to reach safely the port of Ketchikan, Alaska.    When the power turbine blades of one of our Gas Turbine Engine  were all broken, our chief Engineer asked me if we can replace it? This procedure were never done before me. I studied the video and bought a hydraulic jack and fabricate special tools so that we can separate the compressor from the combustion chamber. We recorded the hydraulic pressure when we had loosen the nut of the long stud bolt that hold the compressor and the combustion chamber, we used the same pressure when we put it back.   There was a power outage, after the resumption of the power supply we lost the program of the PLC that controls our reverse osmosis plant.Since we don't have the program, we temporary            convert the control by installing relays, magnetic contactors and           timers and rewire it so that all sensors will function to protect the           equipment and resume our much needed operation.       Experience      Mechanical engineer   05/2006   to   03/2012     Company Name   City  ,   State       Supervise in the Operation and Maintenance of 8 units of boilers, 7 swimming pools, 12 deep wells and 2 incinerators.  Changed the two sand filters and installed new chlorine dosing system on the New infantry  swimming pool ( size of  pool 82 feet by 82 feet) .  Installed new deep well pumps and conducted the testing and commissioning.  Monitor the work of the waste water treatment plant contractor in the installation of pumps and machines and also in the testing and commissioning.  Supervise in the installation of swimming pool pumps, heaters, surface skimmers and in changing the inlet diffusers.  Supervise in the installation of boilers and water softener for a  small laundry in one of our satellite camp.           Mechanical Engineer / Section Head   11/1995   to   01/2006     Company Name   City  ,   State       Over all in-charge in the operation and maintenance of 6 water treatment plants, 2 waste water treatment plants, 8 sewage lifting stations, 15 deep well pumping stations and 2 swimming pools.  Supervise in the excavation and installation of sewer line from the workers accommodation up to the waste water treatment plant with two lifting stations.  Monitor and supervise in the excavation and installation of uPVC pipes for potable water and irrigation water in the officers housing Villas. Replacing the old corroded and weak pipes.  Review all the proposals of the different contractor for the new 2 water treatment plants and submit recommendations to the officer in-charge.  Check our daily, weekly and monthly reports that includes also the water analysis of the raw and product waters of the waste water treatment plants and the water treatment plants.   Inspect the work of the contractor that digs and bore new wells and also in the installation, testing and commissioning of deep well pumps.           Education      Graduate  :   Mechanical Engineering course   1973       FEATI UNIVERSITY   City  ,     Philippines     Mechanical Engineering course        Affiliations     Former member of Philippine Society of Mechanical Engineers       Skills     Electro/Mechanical Skill, Trouble shooting, installation of pipes and pumps, problem solver, design of controls for pumps and motors. installation of swimming pool heaters.    
    """

  # Format the prompt with the resume text
  formatted_prompt = prompt_template.format(resume, "", "")

  # Run the model with the formatted prompt
  output = llm(formatted_prompt, max_tokens=8096, stop=["</s>", "<|im_end|>"])

  # Print the response
  print("--------------------------------")
  print(output["choices"][0]["text"])

  # Free resources in test example too
  del llm
  gc.collect()
