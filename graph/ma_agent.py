import os
import getpass
from typing import TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set up environment variables
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "investment-bank-company-comparison"

# Define state structures
# Define the output structures using Pydantic models
class FinancialMetrics(BaseModel):
    revenue: Optional[float]
    net_income: Optional[float]
    EBITDA: Optional[float]
    gross_profit_margin: Optional[float]
    net_profit_margin: Optional[float]
    ROE: Optional[float]
    ROA: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    PE_ratio: Optional[float]
    market_cap: Optional[float]

class BusinessModel(BaseModel):
    core_products: List[str]
    revenue_streams: List[str]
    target_markets: List[str]

class CompetitiveLandscape(BaseModel):
    market_share: Optional[float]
    key_competitors: List[str]
    competitive_advantages: List[str]

class GrowthInnovation(BaseModel):
    revenue_growth_rate: Optional[float]
    RD_expenditure: Optional[float]
    recent_launches_acquisitions: List[str]


class CompanyProfile(BaseModel):
    name: str
    industry: str
    founded_year: Optional[int]
    headquarters: str
    employees: Optional[int]
    financial_metrics: FinancialMetrics
    business_model: BusinessModel
    competitive_landscape: CompetitiveLandscape
    growth_innovation: GrowthInnovation
    risk_factors: List[str]



class ComparisonState(TypedDict):
    company_a: CompanyProfile
    company_b: CompanyProfile
    comparison_report: Optional[Dict[str, Any]]
    current_step: str
    search_results: Dict[str, List[Dict[str, Any]]]
    reflection_notes: List[str]
    token_usage: Dict[str, int]
    cost: float
    iteration_count: int
    search_attempts: Dict[str, int]
    investment_summary: str


# Initialize tools and models
# Initialize tools and models
tavily_tool = TavilySearchResults(max_results=10, k=10)

# Initialize LLMs (one for analysis, one for structuring)
llm_analyzer = ChatOpenAI(model="gpt-4o-mini")  # Or a suitable model
llm_structurer = ChatOpenAI(model="gpt-3.5-turbo") # A potentially less expensive model

# Token counting function
def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    return len(encoding.encode(string))


def initialize_company_profile(name: str) -> CompanyProfile:
    """Initializes a company profile with default values."""
    return {
        "name": name,
        "industry": "N/A",
        "founded_year": None,
        "headquarters": "N/A",
        "employees": None,
        "financial_metrics": {
            "revenue": None,
            "net_income": None,
            "EBITDA": None,
            "gross_profit_margin": None,
            "net_profit_margin": None,
            "ROE": None,
            "ROA": None,
            "debt_to_equity": None,
            "current_ratio": None,
            "PE_ratio": None,
            "market_cap": None
        },
        "business_model": {
            "core_products": [],
            "revenue_streams": [],
            "target_markets": []
        },
        "competitive_landscape": {
            "market_share": None,
            "key_competitors": [],
            "competitive_advantages": []
        },
        "growth_innovation": {
            "revenue_growth_rate": None,
            "RD_expenditure": None,
            "recent_launches_acquisitions": []
        },
        "risk_factors": []
    }

# Node definitions
def gather_data(state: ComparisonState) -> ComparisonState:
    logger.info("Gathering data...")
    for company in ['company_a', 'company_b']:
        if state['search_attempts'][company] < 2:  # Limit to 2 attempts per company
            queries = [
                f"{state[company]['name']} annual report financial statements business overview",
                f"{state[company]['name']} financial metrics revenue net income market cap",
                f"{state[company]['name']} core products revenue streams",
                f"{state[company]['name']} growth rate R&D expenditure recent acquisitions",
                f"{state[company]['name']} risk factors competitive landscape"
            ]
            for query in queries:
                try:
                    search_results = tavily_tool.invoke(query)
                    state['search_results'][company].extend(search_results)
                    state['search_attempts'][company] += 1
                    logger.info(f"Search results for {state[company]['name']} query '{query}':")
                    logger.info(json.dumps(search_results, indent=2))
                except Exception as e:
                    state['reflection_notes'].append(f"Error searching for {state[company]['name']}: {str(e)}")
        else:
            state['reflection_notes'].append(f"Max search attempts reached for {state[company]['name']}. Proceeding with available data.")
    state['current_step'] = 'gather_data'
    return state

def analyze_data(state: ComparisonState) -> ComparisonState:
    logger.info("Analyzing data...")
    for company in ['company_a', 'company_b']:
        analysis_prompt = f"""
        Analyze the following search results for {state[company]['name']}:
        {json.dumps(state['search_results'][company], indent=2)}

        Extract relevant information to fill in the CompanyProfile structure for the company.
        Use the most recent and reliable information available, prioritizing official sources and financial reports.
        If information is not available, use 'N/A' for text fields and null for numeric fields.
        Return the information in a valid JSON format with the following structure:
        {{
            "name": "{state[company]['name']}",
            "industry": "",
            "founded_year": null,
            "headquarters": "",
            "employees": null,
            "financial_metrics": {{
                "revenue": null,
                "net_income": null,
                "EBITDA": null,
                "gross_profit_margin": null,
                "net_profit_margin": null,
                "ROE": null,
                "ROA": null,
                "debt_to_equity": null,
                "current_ratio": null,
                "PE_ratio": null,
                "market_cap": null
            }},
            "business_model": {{
                "core_products": [],
                "revenue_streams": [],
                "target_markets": []
            }},
            "competitive_landscape": {{
                "market_share": null,
                "key_competitors": [],
                "competitive_advantages": []
            }},
            "growth_innovation": {{
                "revenue_growth_rate": null,
                "RD_expenditure": null,
                "recent_launches_acquisitions": []
            }},
            "risk_factors": []
        }}
        Ensure all fields are filled with the most accurate and recent data available.
        """
        tokens = num_tokens_from_string(analysis_prompt)
        logger.info(f"Tokens in analysis prompt for {company}: {tokens}")
        response = llm.invoke(analysis_prompt)
        state['token_usage'][f'analyze_data_{company}'] = tokens + num_tokens_from_string(response.content)
        state['cost'] += (state['token_usage'][f'analyze_data_{company}'] / 1000) * 0.002

        try:
            parsed_response = json.loads(response.content)
            # Iterate through the parsed response and update individual fields
            for key, value in parsed_response.items():
                state[company][key] = value
        except json.JSONDecodeError as e:
            state['reflection_notes'].append(f"Error parsing analysis response for {company}: {str(e)}")
            state['reflection_notes'].append(f"Raw response: {response.content}")  # Include raw response for debugging
            # If JSON parsing fails, initialize the company profile with default values
            state[company] = initialize_company_profile(state[company]['name'])
        except Exception as e:  # Catch any other exceptions during data processing
            state['reflection_notes'].append(f"An unexpected error occurred during analysis for {company}: {str(e)}")
            state[company] = initialize_company_profile(state[company]['name'])  # Initialize with default values

    state['current_step'] = 'analyze_data'
    return state

def validate_critical_info(state: ComparisonState) -> ComparisonState:
    logger.info("Validating critical information...")
    critical_fields = [
        'industry', 'founded_year', 'headquarters', 'employees',
        'financial_metrics.revenue', 'financial_metrics.net_income', 'financial_metrics.market_cap',
        'business_model.core_products', 'growth_innovation.revenue_growth_rate', 'risk_factors'
    ]
    
    for company in ['company_a', 'company_b']:
        for field in critical_fields:
            if '.' in field:
                category, subfield = field.split('.')
                value = state[company].get(category, {}).get(subfield)
            else:
                value = state[company].get(field)
            
            if value in [None, 'N/A', []]:
                state['reflection_notes'].append(f"Missing critical information for {company}: {field}")
                logger.warning(f"Missing critical information for {company}: {field}")
    
    state['current_step'] = 'validate_critical_info'
    return state
    
    for company in ['company_a', 'company_b']:
        for field in critical_fields:
            if '.' in field:
                category, subfield = field.split('.')
                if state[company][category].get(subfield) in [None, 'N/A', []]:
                    state['reflection_notes'].append(f"Missing critical information for {company}: {field}")
            elif state[company].get(field) in [None, 'N/A', []]:
                state['reflection_notes'].append(f"Missing critical information for {company}: {field}")
    
    state['current_step'] = 'validate_critical_info'
    return state

def additional_targeted_search(state: ComparisonState) -> ComparisonState:
    logger.info("Performing additional targeted search...")
    for company in ['company_a', 'company_b']:
        missing_fields = [note.split(': ')[1] for note in state['reflection_notes'] if note.startswith(f"Missing critical information for {company}")]
        if missing_fields and state['search_attempts'][company] < 3:
            for field in missing_fields:
                search_query = f"{state[company]['name']} {field.replace('_', ' ')} latest"
                try:
                    search_results = tavily_tool.invoke(search_query)
                    state['search_results'][company].extend(search_results)
                    state['search_attempts'][company] += 1
                    logger.info(f"Additional search results for {state[company]['name']} field '{field}':")
                    logger.info(json.dumps(search_results, indent=2))
                except Exception as e:
                    state['reflection_notes'].append(f"Error in additional search for {state[company]['name']} {field}: {str(e)}")
    
    state['current_step'] = 'additional_targeted_search'
    return state

def aggregate_data(state: ComparisonState) -> ComparisonState:
    logger.info("Aggregating data...")
    for company in ['company_a', 'company_b']:
        aggregation_prompt = f"""
        Aggregate and reconcile the following company information:
        {json.dumps(state[company], indent=2)}
        
        Use the most recent and reliable information. If there are conflicts, prefer more recent data from official sources.
        Return the aggregated information in the same JSON format as the input.
        Ensure all fields are filled with the most accurate and recent data available.
        If information is truly not available, use 'N/A' for text fields and null for numeric fields.
        """
        tokens = num_tokens_from_string(aggregation_prompt)
        logger.info(f"Tokens in aggregation prompt for {company}: {tokens}")
        response = llm.invoke(aggregation_prompt)
        state['token_usage'][f'aggregate_data_{company}'] = tokens + num_tokens_from_string(response.content)
        state['cost'] += (state['token_usage'][f'aggregate_data_{company}'] / 1000) * 0.002
        
        try:
            parsed_response = json.loads(response.content)
            state[company] = parsed_response
        except json.JSONDecodeError as e:
            state['reflection_notes'].append(f"Error parsing aggregation response for {company}: {str(e)}")
    
    state['current_step'] = 'aggregate_data'
    return state

def compare_companies(state: ComparisonState) -> ComparisonState:
    logger.info("Comparing companies...")
    comparison_prompt = f"""
    Compare {state['company_a']['name']} and {state['company_b']['name']} based on their profiles:
    {json.dumps(state['company_a'], indent=2)}
    {json.dumps(state['company_b'], indent=2)}
    
    Generate a structured comparison report following this outline:
    1. Executive Summary
    2. Industry Context
    3. Financial Performance Comparison
    4. Business Model and Strategy
    5. Growth and Innovation
    6. Risk Analysis
    7. Valuation Comparison
    8. Competitive Positioning
    9. Future Outlook
    10. Conclusion and Recommendation
    
    If information is missing for certain aspects, acknowledge the lack of data in your comparison.
    Ensure to use relative metrics, analyze trends, consider industry benchmarks, and provide critical insights beyond simple comparisons.
    
    Return the report as a valid JSON object with these sections as keys.
    """
    tokens = num_tokens_from_string(comparison_prompt)
    logger.info(f"Tokens in comparison prompt: {tokens}")
    response = llm.invoke(comparison_prompt)
    state['token_usage']['compare_companies'] = tokens + num_tokens_from_string(response.content)
    state['cost'] += (state['token_usage']['compare_companies'] / 1000) * 0.002
    
    try:
        state['comparison_report'] = json.loads(response.content)
    except json.JSONDecodeError as e:
        state['reflection_notes'].append(f"Error parsing comparison response: {str(e)}")
        state['reflection_notes'].append("Raw response: " + response.content)
    
    state['current_step'] = 'compare_companies'
    return state

def generate_summary(state: ComparisonState) -> ComparisonState:
    logger.info("Generating summary...")
    summary_prompt = f"""
    Based on the comparison report and company profiles:
    {json.dumps(state['comparison_report'], indent=2)}
    {json.dumps(state['company_a'], indent=2)}
    {json.dumps(state['company_b'], indent=2)}

    Generate a concise, actionable summary for an investment banker. The summary should:
    1. Highlight key differences and similarities between the companies
    2. Identify potential investment opportunities or risks
    3. Provide a clear recommendation with supporting rationale

    Limit the summary to 500 words or less.
    """
    tokens = num_tokens_from_string(summary_prompt)
    logger.info(f"Tokens in summary prompt: {tokens}")
    response = llm.invoke(summary_prompt)
    state['token_usage']['generate_summary'] = tokens + num_tokens_from_string(response.content)
    state['cost'] += (state['token_usage']['generate_summary'] / 1000) * 0.002
    
    state['investment_summary'] = response.content
    state['current_step'] = 'generate_summary'
    return state

def reflect_on_analysis(state: ComparisonState) -> ComparisonState:
    logger.info("Reflecting on analysis...")
    reflection_prompt = f"""
    Review the current state of the analysis:
    Company A: {json.dumps(state['company_a'], indent=2)}
    Company B: {json.dumps(state['company_b'], indent=2)}
    Comparison Report: {json.dumps(state['comparison_report'], indent=2)}
    Investment Summary: {state['investment_summary']}
    
    Identify any gaps in information or areas that need further research.
    Consider that we have limited search attempts and some information might not be available.
    If the analysis is complete given the available data, state that no further action is needed.
    
    Return your reflection as a valid JSON object with keys for 'gaps', 'next_steps', and 'status' (either 'complete' or 'needs_more_data').
    """
    tokens = num_tokens_from_string(reflection_prompt)
    logger.info(f"Tokens in reflection prompt: {tokens}")
    reflection = llm.invoke(reflection_prompt)
    state['token_usage']['reflect_on_analysis'] = tokens + num_tokens_from_string(reflection.content)
    state['cost'] += (state['token_usage']['reflect_on_analysis'] / 1000) * 0.002
    
    try:
        parsed_reflection = json.loads(reflection.content)
        state['reflection_notes'].append(json.dumps(parsed_reflection, indent=2))
        state['current_step'] = 'reflect_on_analysis'
        return state
    except json.JSONDecodeError as e:
        state['reflection_notes'].append(f"Error parsing reflection response: {str(e)}")
        state['reflection_notes'].append("Raw response: " + reflection.content)
        state['current_step'] = 'reflect_on_analysis'
        return state

# Graph structure
def create_comparison_graph():
    graph = StateGraph(ComparisonState)
    
    graph.add_node("gather_data", gather_data)
    graph.add_node("analyze_data", analyze_data)
    graph.add_node("validate_critical_info", validate_critical_info)
    graph.add_node("additional_targeted_search", additional_targeted_search)
    graph.add_node("aggregate_data", aggregate_data)
    graph.add_node("compare_companies", compare_companies)
    graph.add_node("generate_summary", generate_summary)
    graph.add_node("reflect_on_analysis", reflect_on_analysis)
    
    graph.add_edge(START, "gather_data")
    graph.add_edge("gather_data", "analyze_data")
    graph.add_edge("analyze_data", "validate_critical_info")
    graph.add_edge("validate_critical_info", "additional_targeted_search")
    graph.add_edge("additional_targeted_search", "aggregate_data")
    graph.add_edge("aggregate_data", "compare_companies")
    graph.add_edge("compare_companies", "generate_summary")
    graph.add_edge("generate_summary", "reflect_on_analysis")
    graph.add_edge("reflect_on_analysis", END)
    
    def should_continue(state: ComparisonState):
        state['iteration_count'] += 1
        logger.info(f"Iteration count: {state['iteration_count']}")
        if state['iteration_count'] >= 3 or all(attempts >= 3 for attempts in state['search_attempts'].values()):
            logger.info("Reached maximum iterations or search attempts. Ending process.")
            return END
        try:
            last_reflection = json.loads(state['reflection_notes'][-1])
            if last_reflection['status'] == 'needs_more_data':
                logger.info("More data needed. Continuing to gather data.")
                return "gather_data"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        logger.info("Analysis complete. Ending process.")
        return END
    
    graph.add_conditional_edges("reflect_on_analysis", should_continue)
    
    return graph.compile()

# Execution
memory = MemorySaver()
comparison_graph = create_comparison_graph()

initial_state = ComparisonState(
    company_a=initialize_company_profile("Apple"),
    company_b=initialize_company_profile("Microsoft"),
    comparison_report=None,
    current_step="start",
    search_results={"company_a": [], "company_b": []},
    reflection_notes=[],
    token_usage={},
    cost=0.0,
    iteration_count=0,
    search_attempts={"company_a": 0, "company_b": 0},
    investment_summary=""
)


config = {"configurable": {"thread_id": "company_comparison", "checkpointer": memory}}
final_state = comparison_graph.invoke(initial_state, config)

# Output results
logger.info("\nCompany A Profile:")
logger.info(json.dumps(final_state['company_a'], indent=2))

logger.info("\nCompany B Profile:")
logger.info(json.dumps(final_state['company_b'], indent=2))

logger.info("\nComparison Report:")
logger.info(json.dumps(final_state['comparison_report'], indent=2))

logger.info("\nInvestment Summary:")
logger.info(final_state['investment_summary'])

logger.info("\nReflection Notes:")
for note in final_state['reflection_notes']:
    logger.info(note)

logger.info("\nToken Usage:")
logger.info(json.dumps(final_state['token_usage'], indent=2))
logger.info(f"\nTotal Cost: ${final_state['cost']:.4f}")

# Optionally, you can analyze the execution path
logger.info("\nExecution Path:")
try:
    for step in memory.get_all_checkpoints("company_comparison"):
        logger.info(f"Step: {step['action']}")
        logger.info(f"Input: {step['input']}")
        logger.info(f"Output: {step['output']}")
        logger.info("---")
except AttributeError:
    logger.info("Unable to retrieve execution path. The MemorySaver object doesn't have the expected method.")

if __name__ == "__main__":
    # You can add any additional execution code here if needed
    pass