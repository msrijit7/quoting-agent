import logging as std_logging
import os
import json
import os
import uuid
# import asyncio  # <-- ADD THIS LINE
import json
from enum import Enum
from typing import AsyncGenerator, List, Optional, Dict, Any

from pydantic import BaseModel, Field
from typing_extensions import override

from google.adk.agents import LlmAgent, BaseAgent, SequentialAgent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.agent_tool import AgentTool 
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset, 
    # StdioServerParameters, 
    # SseConnectionParams,
    StreamableHTTPConnectionParams
)
from google.adk.tools.mcp_tool.mcp_tool import MCPTool
from google.cloud import storage
from google.adk.models.lite_llm import LiteLlm
from google import genai
from google.genai.types import GenerateContentConfig, Part


from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.units import inch

import pprint
import warnings

from app.config import PROJECT_ID, LOCATION, GCS_BUCKET

# Ignore all warnings
warnings.filterwarnings("ignore")


# add your gcp projec and location
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_CLOUD_LOCATION"] = LOCATION
os.environ['ANTHROPIC_API_KEY'] = ''


# --- Constants ---
APP_NAME = "quoting_app"
USER_ID = "user_test_001"
SESSION_ID = "session_test_001"
GEMINI_MODEL = "gemini-2.5-flash"
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"

# storage
LOCAL_ROOT_DIR = "./app"
LOCAL_OUTPUT_DIR = LOCAL_ROOT_DIR + "/data/output_quotes"
GCS_EMAIL_FILE_PATH = "data/input/emails.json"
GCS_OUTPUT_DIR = "data/output/"


MAPS_MCP_SERVER_URL = "https://mcp-google-maps-server-993356934697.us-central1.run.app/mcp"

# --- Configure Logging ---
log_level = std_logging.DEBUG
logger_names = []
for name, logger in std_logging.root.manager.loggerDict.items():
    if name.startswith('google_adk.') and isinstance(
        logger, std_logging.Logger
    ):
        logger_names.append(name)
        logger.setLevel(log_level)

std_logging.info('Forced logging level %s for the following modules: %s', log_level, ', '.join(logger_names))
# Set the same for future loggers
std_logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = std_logging.getLogger(__name__)

# Create output directory if it doesn't exist
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# Global variables
currency_map = {} 
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
       
# --- Pydantic Models for Data Structures ---
class TransportationMode(str, Enum):
    """Enum for transportation modes."""
    AIR = "Air"
    SEA = "Sea"
    ROAD = "Road"


# --- Pydantic Models for Data Structures ---
class ExtractedEntities(BaseModel):
    """Entities extracted from an email for quoting purposes."""
    customer_name: str = Field(None, description="Name of the customer or company requesting the quote.")
    product_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of products, e.g., [{'name': 'widget', 'quantity': 100}, {'name': 'gizmo', 'quantity': 10}]"
    )
    contact_person: Optional[str] = Field(None, description="Name of the contact person if mentioned in the email.")
    origin_address: str = Field(None, description="origin address if mentioned.")
    destination_address: str = Field(None, description="destination address if mentioned.")
    transportation_mode: Optional[TransportationMode] = Field(None, description="transportation mode if mentioned.")
    additional_notes: Optional[str] = Field(None, description="Any other relevant details extracted for quoting.")
    model_config = {"arbitrary_types_allowed": True}

class QuoteItem(BaseModel):
    """Single quote item information for a generated quote."""
    product_name: str
    quantity: int
    unit_price: float
    item_total_price: float

class AllQuoteItems(BaseModel):
    """all quote item information for a generated quote."""
    items: List[QuoteItem]
    

class QuoteDetails(BaseModel):
    """Detailed information for a generated quote."""
    quote_id: str = Field(default_factory=lambda: f"Q-{uuid.uuid4().hex[:8].upper()}")
    customer_name: str
    contact_email: Optional[str] = None
    transporation_mode: TransportationMode = TransportationMode.ROAD
    origin_address: Optional[str] = None
    destination_address: Optional[str] = None
    driving_distance: Optional[float] = None
    multiplier: float = 1.0
    all_items: AllQuoteItems
    grand_total: float
    currency: str = "USD"
    notes: Optional[str] = "Thank you for your business!"
    model_config = {"arbitrary_types_allowed": True}



# --- Custom collect_addtional_data Agent
class CustomAdditionalDataCollectionAgent(BaseAgent):
    """
    Custom agent for collecting additional data for quoting.
    """

    # --- Field Declarations for Pydantic ---
    # Declare the agents passed during initialization as class attributes with type hints
    map_assistant_agent: LlmAgent

    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, 
        name: str,
        map_assistant_agent: LlmAgent
    ):
        """
        Initialize the agent.
        """
        super().__init__(name=name, map_assistant_agent=map_assistant_agent, sub_agents=[map_assistant_agent])

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the collect addional data.
        Uses the instance attributes assigned by Pydantic (e.g., self.map_assistant_agent).
        """
        global currency_map
        logger.info(f"[{self.name}] - Starting data collection agent.")

        # clean the state variables of previous quoting request if exists
        if 'road_distance_value' in ctx.session.state:
            del ctx.session.state['road_distance_value']
        if 'multiplier' in ctx.session.state:
            del ctx.session.state['multiplier']
        if 'current_quote_details' in ctx.session.state:
            del ctx.session.state["current_quote_details"]
        if 'currency_conversion_rate' in ctx.session.state:
            del ctx.session.state["currency_conversion_rate"]
        if 'target_currency' in ctx.session.state:
            del ctx.session.state["target_currency"]


        extracted_entities_tbd = ctx.session.state.get("extracted_entities_for_quote", None)
        try:
            if isinstance(extracted_entities_tbd, dict):
                extracted_entities = extracted_entities_tbd
            elif isinstance(extracted_entities_tbd, str):
                extracted_entities_text = extracted_entities_tbd
                # 1. Convert the string to a Python dictionary (JSON object)
                if "```json" in extracted_entities_text:
                    extracted_entities_text = extracted_entities_text.removeprefix("```json\n").removesuffix("\n```")
                extracted_entities = json.loads(extracted_entities_text)

            ctx.session.state["extracted_entities_for_quote"] = extracted_entities
            logger.info(f"[{self.name}] - Extracted entities: {extracted_entities}")
        except json.JSONDecodeError as e:
            print(f"[{self.name}] -: Error decoding JSON: {e}")
            return
        except (KeyError, IndexError) as e:
            print(f"[{self.name}] -: Error accessing key: {e}. The JSON structure might be different than expected.")
            return

        ctx.session.state['multiplier'] = 1.0 # default value
        if 'transportation_mode' in extracted_entities:
            if extracted_entities['transportation_mode'] == TransportationMode.ROAD:
                if 'origin_address' in extracted_entities and 'destination_address' in extracted_entities:
                    # Use the sequential_agent instance attribute assigned during init
                    async for event in self.map_assistant_agent.run_async(ctx):
                        logger.info(f"[{self.name}] Event from map_assistant_agent: {event.model_dump_json(indent=2, exclude_none=True)}")
                        yield event
            elif extracted_entities['transportation_mode'] == TransportationMode.SEA:
                ctx.session.state['multiplier'] = 0.8
            elif extracted_entities['transportation_mode'] == TransportationMode.AIR:
                ctx.session.state['multiplier'] = 2.0
            else:
                logger.error(f"[{self.name}] non-supported transportation mode: {extracted_entities['transportation_mode']}. Aborting workflow.")
                pass
        else:
            logger.info(f"[{self.name}] No transportation mode is found in extracted entities, and set default mulitplier to 1.")
        
        # collect the currency converssion rate
        if 'origin_address' in extracted_entities:
            if not currency_map:
                with open(f"{LOCAL_ROOT_DIR}/data/currency_to_city_map.json", 'r') as f:
                    currency_map = json.load(f)
            origin_city = extracted_entities['origin_address'].lower()
            for currency, cities in currency_map.items():
                if currency == 'USD':
                    continue
                if origin_city in cities:
                    get_currency_conversion_rate_func(
                        base_currency="USD",
                        target_currency=currency,
                        tool_context=ToolContext(invocation_context=ctx)
                    )
                    break
        
        logger.info(f"[{self.name}] task is finished.")



# --- Function Tool Implementations ---
def get_email_from_user_func(customer_name: str, tool_context: ToolContext) -> str: #InvocationContext
    """
    Retrieves the latest email content and contact email for a specified customer name
    from the email repository (emails.json).
    Stores customer_name, contact_email, and email_body in the session.
    """
    logger.info(f"[FunctionTool] get_email_from_user_func called for: {customer_name}")
    try:
        # with open(EMAIL_REPO_PATH, 'r') as f:
        #     data = json.load(f)
        storage_client = storage.Client()

        # Get the bucket and blob (file)
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_EMAIL_FILE_PATH)

        # Download the contents of the blob as a string and parse it
        data_string = blob.download_as_text()
        data = json.loads(data_string)
   
        customer_data = next((c for c in data.get("customers", []) if c.get("name", "").lower() == customer_name.lower()), None)
        
        if customer_data and customer_data.get("emails"):
            latest_email = customer_data["emails"][-1] # Get the last email
            email_body = latest_email.get("body", "Email body not found.")
            contact_email = customer_data.get("contact_email", "Contact email not found.")
            
            tool_context.state["active_customer_name"] = customer_data["name"]
            tool_context.state["active_customer_contact_email"] = contact_email
            tool_context.state["active_email_body"] = email_body
            
            logger.info(f"[FunctionTool] Email found for {customer_name}. Body: {email_body[:100]}...")
            return f"Email for {customer_name}:\nSubject: {latest_email.get('subject', 'N/A')}\nBody:\n{email_body}"
        else:
            logger.warning(f"[FunctionTool] No email data found for customer: {customer_name}")
            return f"Sorry, I could not find any email records for customer '{customer_name}'."
    except Exception as e:
        logger.error(f"[FunctionTool] Error in get_email_from_user_func: {e}")
        return f"An error occurred while trying to retrieve the email: {str(e)}"


def get_currency_conversion_rate_func(
    base_currency: str, target_currency: str, tool_context: ToolContext
) -> dict:
    """
    Gets the conversion rate from a base currency to a target currency.
    This is a simplified version; a real implementation would use an API.
    """
    logger.info(
        f"[FunctionTool] get_currency_conversion_rate_func called for: "
        f"{base_currency} to {target_currency}"
    )
    # In a real scenario, this would call an MCP tool to get the conversion rate.
    # For this example, we'll use a fixed rate.
    conversion_rates = {
        "USD_to_RMB": 7.25,
        "RMB_to_USD": 1/7.25,
        "USD_to_EUR": 0.92,
        "EUR_to_USD": 1/0.92,
    }
    key = f"{base_currency}_to_{target_currency}"
    if key not in conversion_rates:
        raise ValueError(f"Conversion rate for {key} not found.")
    
    tool_context.state["currency_conversion_rate"] = conversion_rates[key]
    tool_context.state["target_currency"] = target_currency
    
    return {"conversion_rate": conversion_rates[key], "target_currency": target_currency}


def generate_quote_func(tool_context: ToolContext) -> dict:
    """
    Generates a formal quote based on extracted entities.
    This is a simplified version; a real implementation would involve pricing logic.
    """
    logger.info(f"[FunctionTool] generate_quote_func is called")
    if "extracted_entities_for_quote" not in tool_context.state:
        logger.error(f"[FunctionTool] generate_quote_func: the extracted_entities_for_quote is not in session state.")
        return "Error - no extracted entities is found in session state. "
    
    entities = tool_context.state.get("extracted_entities_for_quote")
    pprint.pprint(entities)
    
    items = []
    grand_total = 0.0
    
    # Dummy pricing logic
    if 'product_details' not in entities:
        return {"Error:" f"No product details found in the entities {entities}"}

    # get and adjust the multiplier 
    multiplier = 1.0
    if 'road_distance_value' in tool_context.state:
        distance_value = tool_context.state['road_distance_value']
        if distance_value > 2000:
            tool_context.state['multiplier'] = 1.2  
        
    if 'multiplier' in tool_context.state:
        multiplier = tool_context.state['multiplier']
   
    for product_detail in entities['product_details']:
        name = product_detail.get("name", "Unknown Product")
        quantity = product_detail.get("quantity", 1)
        
        unit_price = 10.0  # Default price
        if "widget" in name.lower():
            unit_price = 15.75
        elif "supergizmo" in name.lower():
            unit_price = 50.0
        elif "gizmo" in name.lower():
            unit_price = 25.50

        item_total = quantity * unit_price
        items.append(QuoteItem(product_name=name, quantity=quantity, unit_price=unit_price, item_total_price=item_total))
        grand_total += item_total
    grand_total *= multiplier

    target_currency = "USD"
    if "currency_conversion_rate" in tool_context.state:
        conversion_rate = tool_context.state["currency_conversion_rate"]
        target_currency = tool_context.state["target_currency"]

        for item in items:
            item.unit_price *= conversion_rate
            item.item_total_price *= conversion_rate
        grand_total *= conversion_rate

    if "road_distance_value" in tool_context.state:
        quote = QuoteDetails(
            customer_name=tool_context.state.get("active_customer_name", "N/A"),
            contact_email=tool_context.state.get("active_customer_contact_email", "N/A"),
            transporation_mode=entities.get("transportation_mode", TransportationMode.ROAD),
            origin_address=entities.get("origin_address", "N/A"),
            destination_address=entities.get("destination_address", "N/A"),
            driving_distance=tool_context.state.get("road_distance_value", 0.0),
            multiplier=multiplier,
            all_items=AllQuoteItems(items=items),
            grand_total=grand_total,
            currency=target_currency,
            notes=f"Quote based on request: details from email."
        )
    else:
        quote = QuoteDetails(
            customer_name=tool_context.state.get("active_customer_name", "N/A"),
            contact_email=tool_context.state.get("active_customer_contact_email", "N/A"),
            transporation_mode=entities.get("transportation_mode", TransportationMode.ROAD),
            origin_address=entities.get("origin_address", "N/A"),
            destination_address=entities.get("destination_address", "N/A"),
            multiplier=multiplier,
            all_items=AllQuoteItems(items=items),
            grand_total=grand_total,
            currency=target_currency,
            notes=f"Quote based on request: Details from email."
        )

    logger.info(f"[FunctionTool] Generated quote: {quote.model_dump_json(indent=2)}")
    tool_context.state['current_quote_details'] = quote.model_dump()
    return quote


def change_quote_func(modifications: str, tool_context: ToolContext) -> dict:
    """
    Modifies an existing quote based on user instructions.
    The current quote should be in session state: current_quote_details.
    This function uses an LLM to interpret modifications and apply them.
    For simplicity in this example, we'll just append modification note.
    A real version would re-calculate.
    """
    logger.info(f"[FunctionTool] change_quote_func called with modifications: {modifications}")
    current_quote_dict = tool_context.state.get("current_quote_details")
    if not current_quote_dict:
        raise ValueError("No current quote found in session to modify.")
    
    try:
        quote = QuoteDetails.model_validate(current_quote_dict)
        # using LLM to parse the modification request
        prompt = f"""
            Given a list of quote items and a request to modify the quote items, e.g., change the quantity of a quote item,
            or change the unit price all of a quote item, your task is to modify the content of the quote items 
            according to the request, and output the modified quote items. 

            A quote item contains the following fields:
            -    product_name: str
            -    quantity: int
            -    unit_price: float
            -    item_total_price: float

            When you modify the item, you need follow these rules:
            # Note only 'quantity' and 'unit_price' are allowed to modified.
            # item_total_price = quantity * unit_price, so after you modify the quanity or unit_price, you need re-calculate the item_total_price.
            
            <Input_Items>
            {quote.all_items.model_dump_json(indent=2)}
            </Input_Items>

            <Modification_Request>
                {modifications}
            </Modification_Request>
        """
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AllQuoteItems,
            ),
        )

        all_quote_items: AllQuoteItems = response.parsed
        logger.info(f"Updated quote items: {all_quote_items.model_dump_json(indent=2)}")
        new_grand_total = 0.0
        multiplier = quote.multiplier #tool_context.state['multiplier'] if 'multiplier' in tool_context.state['multiplier'] else 1.0
        for item in all_quote_items.items:
            new_grand_total += item.item_total_price
        new_grand_total *= multiplier
        quote.grand_total = new_grand_total
        quote.all_items = all_quote_items
        quote.notes = str(quote.notes) + f"\nSYSTEM: {modifications}."
        logger.info(f"Updated quote items: {all_quote_items.model_dump_json(indent=2)}, New total: {quote.grand_total}")
                    
        tool_context.state['current_quote_details'] = quote.model_dump()
        return quote
    except Exception as e:
        logger.error(f"Error modifying quote: {e}")
        # Re-raise or return error message
        raise ValueError(f"Could not modify quote: {e}")


def create_pdf_from_dictionary(data_dict: Dict[str, Any], filename: str = "output.pdf"):
    """
    Creates a PDF file from the contents of a Python dictionary.

    Args:
        data_dict (Dict[str, Any]): The dictionary containing the data to write to PDF.
        filename (str): The name of the PDF file to create.
    """
    c = pdf_canvas.Canvas(filename, pagesize=letter)
    width, height = letter # Get page dimensions

    # Set initial text position
    x_start = inch
    y_start = height - inch
    line_height = 0.25 * inch

    c.setFont("Helvetica", 12)
    c.drawString(x_start, y_start, "--- Generated Quote---")
    y_start -= line_height # Move down for the next line

    # Iterate through the dictionary and write key-value pairs to the PDF
    for key, value in data_dict.items():
        # Handle nested dictionaries gracefully for display
        if isinstance(value, dict):
            c.drawString(x_start, y_start, f"{key}:")
            y_start -= line_height
            # Indent nested dictionary items
            for sub_key, sub_value in value.items():
                c.drawString(x_start + 0.5 * inch, y_start, f"  {sub_key}: {sub_value}")
                y_start -= line_height
        else:
            c.drawString(x_start, y_start, f"{key}: {value}")
            y_start -= line_height

        # Add a new page if content goes beyond the current page
        if y_start < inch:
            c.showPage()
            y_start = height - inch # Reset Y position for new page
            c.setFont("Helvetica", 12) # Re-set font in case it was changed
            c.drawString(x_start, y_start, "--- Dictionary Contents (Continued)---")
            y_start -= line_height

    c.save()
    print(f"PDF '{filename}' created successfully!")


def send_quote_func(tool_context: ToolContext) -> str:
    """
    Generates a PDF of the current quote and sends it.
    The quote should be in session state: current_quote_details.
    The customer email should be in session state: active_customer_contact_email.
    Returns the path to the generated PDF.
    """
    logger.info("[FunctionTool] send_quote_func called.")
    quote_dict = tool_context.state.get("current_quote_details")
    customer_email = tool_context.state.get("active_customer_contact_email")

    if not quote_dict:
        return "Error: No quote found in session to send."
    if not customer_email:
        return "Error: No customer email found in session to send the quote to."

    try:
        quote = QuoteDetails.model_validate(quote_dict)
    except Exception as e:
        return f"Error: Invalid quote data in session: {e}"

    pdf_file_name = f"Quote_{quote.quote_id}_{quote.customer_name.replace(' ', '_')}.pdf"
    local_pdf_path = os.path.join(LOCAL_OUTPUT_DIR, pdf_file_name)
    gcs_pdf_path = os.path.join(GCS_OUTPUT_DIR, pdf_file_name)
    full_gcs_pdf_path = f"gs://{GCS_BUCKET}/{gcs_pdf_path}"

    # Simulate PDF generation (creating a text file with .pdf extension)
    try:
        create_pdf_from_dictionary(quote_dict, local_pdf_path)

        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_pdf_path)
        blob.upload_from_filename(local_pdf_path, timeout=120)
        os.remove(local_pdf_path)
        # print("PDF file upload successful.")
        # Simulate sending email
        logger.info(f"[FunctionTool] Simulating sending email with quote PDF to {customer_email}")

        tool_context.state["sent_quote_pdf_path"] = full_gcs_pdf_path
        return f"Quote PDF generated: {full_gcs_pdf_path}. It has been (simulated) sent to {customer_email}."
    except Exception as e:
        logger.error(f"[FunctionTool] Error generating or sending PDF: {e}")
        return f"Error during PDF generation/sending: {str(e)}"
    

def get_current_session_state_func(tool_context: ToolContext) -> dict:
    """
    Get the current session state.
    """
    logger.info(f"[FunctionTool] get_current_state_func called.")
    logger.info(f"[FunctionTool] current state: {tool_context.state}")
    return tool_context.state.to_dict()


# --- Callback functions Implementations ---
def after_mcptools_callback_func(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Dict
) -> Optional[Dict]:
    extracted_entities = tool_context.state.get("extracted_entities_for_quote", None)
    if extracted_entities is None or not isinstance(extracted_entities, dict):
        logger.error(f"after_mcptools_callback_func(): extracted_entities {extracted_entities} is not a dictionary.")
        return None

    if tool.name != 'maps_distance_matrix':
        logger.error(f"after_mcptools_callback_func(): tool name {tool.name} is not 'maps_distance_matrix'.")
        return None

    try:
        distance_matrix_text = tool_response.content[0].text
    except Exception as e:
        logger.error(f"after_mcptools_callback_func(): no expectected response from calling maps_distance_matrix(); error: {e}; tool_response: {tool_response}")
        return None

    if not isinstance(distance_matrix_text, str):
        logger.error(f"after_mcptools_callback_func(): the distance_matrix is not a string. Its value: {distance_matrix_text} meters")

    try:
        # 1. Convert the string to a Python dictionary (JSON object)
        data = json.loads(distance_matrix_text)

        # # 2a. for stdio mcp google map server: Check if 'results' key exists and is not empty, and if the status is "OK"
        # if 'results' in data and data['results'] and data['results'][0]['elements'][0]['status'] == 'OK':
            
        #     # 3. Access the nested 'distance' object
        #     # This path is now safe to access because of the checks above
        #     distance_object = data['results'][0]['elements'][0]['distance']

        # 2b. for streamableHttp mcp google map server: 
        if 'distances' in data and  isinstance(data['distances'], list) and len(data['distances']) == 1 and isinstance(data['distances'][0], list) and len(data['distances'][0]) == 1:
            
            # 3. Access the nested 'distance' object
            distance_object = data['distances'][0][0]
            
            # 4. Get the value of the 'value' key from the distance object
            distance_value = distance_object['value']
            
            # 5. store the final value to state
            tool_context.state['road_distance_value'] = int(distance_value / 1000)
            print(f"after_mcptools_callback_func(): the extracted distance is: {distance_value}")

            # 6. instruct the map_assistant_agent to skip the summary of the response
            tool_context.actions.skip_summarization = True
        else:
            # Handle cases where results are missing or the status is not "OK"
            print(f"after_mcptools_callback_func(): Could not retrieve distance. The 'results' field was missing, empty, or the status was not 'OK'.")
            print(f"after_mcptools_callback_func(): json-data {data}")
    except json.JSONDecodeError as e:
        print(f"after_mcptools_callback_func(): Error decoding JSON: {e}")
    except (KeyError, IndexError) as e:
        print(f"after_mcptools_callback_func(): Error accessing key: {e}. The JSON structure might be different than expected.")

    return None


email_parsing_agent = LlmAgent(
    name="email_parsing_agent",
    model=GEMINI_MODEL,
    instruction=f"""You are an expert email parser. Your task is to analyze an email provided in session state with key 'active_email_body'. 
    Extract the following information and structure it according to the {ExtractedEntities.__name__} schema:
    - customer_name: The name of the company or individual.
    - product_details: A list of products mentioned with their quantities (e.g., [{{"name": "widget", "quantity": 100}}]). Try to find all product mentions.
    - contact_person: The name of the person who sent the email, if identifiable.
    - origin_address:  The origin address of the shipping if mentioned.
    - destination_address: destination address of the shipping mentioned.
    - transportation_mode: transportation mode if mentioned, a value from ["Sea", "Road", "Air"]
    - additional_notes: Any other relevant information for quoting.

    If the email does not seem to be a quote request, output that information clearly.
    """,
    output_key="extracted_entities_for_quote",
    input_schema=None,
    output_schema=ExtractedEntities, # Ensures LLM output conforms to this Pydantic model
)


map_assistant_agent = LlmAgent(
    name="map_assistant_agent",
    model=GEMINI_MODEL,
    instruction=f"""
    You are a google map assistant, your task is to Help with estimate the driving distance of two places: orgin_address and destination_address, 
    which can be found in session state with key 'extracted_entities_for_quote'.
    if you are asked for driving distances of two places, call tool function 'maps_distance_matrix', and parse its result and reply the user.
    If not, reply the request in a proper way.
    """,
    tools=[
        MCPToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=MAPS_MCP_SERVER_URL #"http://localhost:3000/mcp"
            ),
            errlog=None,
            tool_filter=['maps_distance_matrix']
        )
    ],
    after_tool_callback=after_mcptools_callback_func,
)


addtional_data_collection_agent = CustomAdditionalDataCollectionAgent(
    name="additional_data_collection_agent",
    map_assistant_agent=map_assistant_agent,
)

quote_generation_agent = LlmAgent(
    name="quote_generation_agent",
    model=GEMINI_MODEL,
    instruction=f"""You are a quote specialist.
    You have received extracted entities from an email, included in session state with key 'extracted_entities_for_quote'.
    Your task is to use the 'generate_quote_func' tool to generate a formal quote based on these entities. 
    Note the 'driving_distance' unit in the quote is km, not meters; make sure to include 'multiplier' in the quote.

    If the extracted entities don't exist in session state, output that information clearly.

    """,
    tools=[generate_quote_func], # Tool this agent can use
)


# --- Sequential Agent for the Quoting Pipeline ---
quoting_pipeline_agent = SequentialAgent(
    name="quoting_pipeline_agent",
    sub_agents=[email_parsing_agent, addtional_data_collection_agent, quote_generation_agent],
    description="A pipeline that first parses an email for quote details, then collect addtional data, and finally generates a formal quote. Trigger this after an email is retrieved and user has confirmed to be a quote request."
)


# --- Main Conversational Agent ---
root_agent = LlmAgent(
    name="root_agent",
    # model=LiteLlm(model=MODEL_CLAUDE_SONNET),
    model=GEMINI_MODEL,
    instruction="""You are a polite and efficient quoting assistant. Your goal is to help the user manage customer quotes.
You have a set of tools to perform tasks. Based on the user's request, decide which tool to use.
Always inform the user about the actions you are taking and the results.
If you need more information from the user (e.g., a customer name), ask for it clearly.

Available Tools:
- get_email_from_user_func: Use to fetch the latest email for a customer.
- quoting_pipeline_agent: Use to process a retrieved email (that is a quote request) to extract details and generate a quote.
- change_quote_func: Use to make changes to an already generated quote.
- send_quote_func: Use to create a PDF of the current quote and simulate sending it.
- get_current_session_state_func: Use to display current session state varaibles and values of the quoting assistant.

Workflow hints:
1. User might ask to get an email. Use 'get_email_from_user_func' to get the latest email and show its content to user. 
2. If the email is a quote request and user ask to generate a quote, get the email content from session state with key 'active_email_body', and send it to the 'quoting_pipeline_agent' tool to generate the quote.
3. User might ask to change parts of the generated quote. Use 'change_quote_func'.
4. Finally, user might ask to send the quote. Use 'send_quote_func'.

when calling a tool, If the tool returns an error, inform the user politely; If the tool is successful, present the result clearly.
Keep track of important information in the session (e.g., current customer, email content, quote details).
The tools will help manage this session data. Your responses should reflect the outcomes of these tool uses.
If a tool reports an error, inform the user clearly.
""",
    tools=[
        get_email_from_user_func,
        AgentTool(agent=quoting_pipeline_agent),
        change_quote_func,
        send_quote_func,
        get_current_session_state_func
    ],
    # enable_tool_use=True
)

