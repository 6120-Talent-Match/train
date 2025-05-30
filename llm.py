import os
from prompt import system_prompt
from config import settings
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def get_response(user_prompt: str):
  response = client.responses.create(
      model="gpt-4o",
      input=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": user_prompt}
      ],
      text={
          "format": {
              "type": "json_schema",
              "name": "resume_info",
              "schema": {
                  "type": "object",
                  "properties": {
                      "skills": {
                          "type": "array",
                          "items": {
                              "type": "string"
                          }
                      },
                      "education": {
                          "type": "array",
                          "items": {
                              "type": "string"
                          }
                      },
                      "experience": {
                          "type": "array",
                          "items": {
                              "type": "string"
                          }
                      },
                  },
                  "required": ["skills", "education", "experience"],
                  "additionalProperties": False
              },
              "strict": True
          }
      }
  )
  return response.output_text


if __name__ == "__main__":
  print(get_response("         HR SPECIALIST, US HR OPERATIONS       Summary     Versatile  media professional with background in Communications, Marketing, Human Resources and Technology.\u00a0        Experience     09/2015   to   Current     HR Specialist, US HR Operations    Company Name   \uff0d   City  ,   State       Managed communication regarding launch of Operations group, policy changes and system outages      Designed standard work and job aids to create comprehensive training program for new employees and contractors         Audited job postings for old, pending, on-hold and draft positions.           Audited union hourly, non-union hourly and salary background checks and drug screens             Conducted monthly new hire benefits briefing to new employees across all business units               Served as a link between HR Managers and vendors by handling questions and resolving system-related issues         Provide real-time process improvement feedback on key metrics and initiatives  Successfully re-branded US HR Operations SharePoint site  Business Unit project manager for RFI/RFP on Background Check and Drug Screen vendor         01/2014   to   05/2015     IT, Marketing and Communications Co-op    Company Name   \uff0d   City  ,   State      Posted new articles, changes and updates to corporate SharePoint site including graphics and visual communications.  Researched and drafted articles and feature stories to promote company activities and programs.  Co-edited and developed content for quarterly published newsletter.  Provided communication support for internal and external events.  Collaborated with Communication team, media professionals and vendors to determine program needs for print materials, web design and digital communications.  Entrusted to lead product, service and software launches for Digital Asset Management tool, Marketing Toolkit website and Executive Tradeshows Calendar.  Created presentations for management and executive approval to ensure alignment with corporate guidelines and branding.  Maintained the MySikorsky SharePoint site and provided timely solutions to mitigate issues.\u00a0\u00a0\u00a0\u00a0  Created story board and produced video for annual IT All Hands meeting.         10/2012   to   01/2014     Relationship Coordinator/Marketing Specialist    Company Name   \uff0d   City  ,   State       Partnered with vendor to manage the in-house advertising program consisting of print and media collateral pieces.     Coordinated pre-show and post-show activities at trade shows.     Managed marketing campaigns to generate new business and to support partner and sales teams.     Ordered marketing collateral for meetings, trade shows and advisors.    Improved, administered and modified marketing programs to increase product awareness.  Assisted in preparing internal promotional publications, managed marketing material inventory and supervised distribution of publications to ensure high quality product output.  Coordinated marketing materials including brochures, promotional materials and products.  Partnered with graphic designers to develop appropriate materials and branding for brochures.  Used tracking and reporting systems for sales leads and appointments.         09/2009   to   10/2012     Assistant Head Teller    Company Name   \uff0d   City  ,   State       Received an internal audit score of  100 %.     Performed daily and monthly audits of ATM machines and tellers.     Educated customers on a variety of retail products and available credit options.       Consistently met or exceeded quarterly sales goals     Promoted products and services to\ncustomers while maintaining company brand identity\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\n  Implemented programs to achieve\nand exceed customer and company participation goals\u00a0\n\n\u00a0  Organized company sponsored events on campus resulting in increased\nbrand awareness\n\n\u00b7\u00a0\u00a0\u00a0\u00a0\n  Coached peers on\nthe proper use of programs to improve work flow efficiency  Utilized product knowledge to successfully sell\nto and refer clients based on individual needs  Promoted marketing the grand opening\nof new branch locations to strengthen company brand affinity\n\n\u00b7\u00a0\u00a0\u00a0\u00a0   Organized company sponsored events\nresulting in increased brand awareness and improved sales\n\n\u00b7\u00a0\u00a0\u00a0\u00a0   Coached peers on the proper use of\nprograms to increase work flow efficiency\n\n          Senior Producer - 2014 SHU Media Exchange    Company Name   \uff0d   City  ,   State      Planned and executed event\u00a0focusing on Connecticut's creative corridor, growth of industry and opportunities that come with development. A\u00a0 panel of industry professionals addressed topics related to media and hosted a question and answer session for approximately 110 attendees. Following the forum, guests were invited to engage in networking and conversation at a post-event reception.         Education     2014     Master of Arts  :   Corporate Communication & Public Relations    Sacred Heart University   \uff0d   City  ,   State             2013     Bachelor of Arts  :   Relational Communication    Western Connecticut State University   \uff0d   City  ,   State              Skills    Adobe Photoshop, ADP, Asset Management, branding, brochures, content, Customer Care, Final Cut Pro, graphics, graphic, HR, Illustrator, InDesign, Innovation, inventory, Lotus Notes, marketing, marketing materials, marketing material, materials, Microsoft Office, SharePoint, newsletter, presentations, process improvement, Project Management, promotional materials, publications, Quality, real-time, Recruitment, reporting, RFP, sales, stories, Employee Development, video, web design, website, articles   "))
