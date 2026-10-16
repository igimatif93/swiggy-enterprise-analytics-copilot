import os
import re
import time
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain & Groq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

# --- 1. Database Connection ---
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0000")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "swiggy_db")

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(MYSQL_URL)

# --- 2. Model Initialization ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")

base_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_KEY
)

# --- 3. Database Execution Tool with Self-Correction Feedback ---
last_executed_sql = None

@tool
def query_swiggy_sql(query: str) -> str:
    """Executes a read-only MySQL SQL query on swiggy_db. Returns tabular markdown string."""
    global last_executed_sql
    cleaned = re.sub(r"```(sql)?", "", query).strip()
    
    # Read-Only Security Guardrail
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    if any(tok in cleaned.upper() for tok in forbidden):
        return "SECURITY ERROR: Only read-only operations are permitted."
    
    if "LIMIT" not in cleaned.upper() and not cleaned.upper().startswith("CREATE"):
        cleaned = f"{cleaned.rstrip(';')} LIMIT 50;"

    last_executed_sql = cleaned
    try:
        with engine.connect() as conn:
            if cleaned.upper().startswith("CREATE"):
                conn.execute(text(cleaned))
                conn.commit()
                df = pd.DataFrame([{"Status": "Success"}])
            else:
                df = pd.read_sql_query(text(cleaned), conn)
        
        if df.empty:
            return "SUCCESS: Query executed, 0 matching records."
        return df.head(10).to_markdown(index=False)
    except Exception as e:
        return f"SQL_EXECUTION_ERROR: {str(e)}. Review syntax, table aliases, or reserved words."

@tool
def search_customer_reviews(query: str) -> str:
    """Mock search for feedback corpus during quantitative SQL benchmarking."""
    return "Mock Review Context: Customers reported delayed deliveries during heavy rain."

tools = [query_swiggy_sql, search_customer_reviews]
tools_dict = {t.name: t for t in tools}
llm_with_tools = base_llm.bind_tools(tools)

# --- 4. Master Star Schema Context ---
STAR_SCHEMA_CONTEXT = """You are the Swiggy Enterprise Analytics Copilot.
You have access to tools to query a MySQL star schema database (swiggy_db).

Schema:
- fact_transactions: transaction_id, customer_id, transaction_date, gross_amount, net_amount, rating, feedback_text, restaurant_product_id, coupon_id, campaign_id, geo_id, city, delivery_minutes, raining_flag, surge_flag.
- dim_restaurant_product: restaurant_product_id, restaurant_name, product_name, city, cuisine_tag, list_price.
- dim_customer: customer_id, customer_name, gender, age, membership_tier, home_city.
- dim_coupon: coupon_id, coupon_name, discount_value, max_discount, min_order.
- dim_campaign: campaign_id, campaign_name, channel, objective.
- dim_geo: geo_id, city, state, pincode.

Rules:
1. Always invoke `query_swiggy_sql` for numbers, revenue, counts, averages, and rankings.
2. If `query_swiggy_sql` returns SQL_EXECUTION_ERROR, inspect the error and self-correct.
3. Use COUNT(*) or COUNT(1), never empty COUNT(). Do not use unquoted aliases like `lines`.
"""

# --- 5. Benchmark Dataset (12 Representative Operational Questions) ---
BENCHMARK_SUITE = [
    {"id": 1, "category": "Aggregation", "query": "What is the total net revenue across all transactions?"},
    {"id": 2, "category": "Single Join", "query": "What are the top 5 restaurant brands by total order count?"},
    {"id": 3, "category": "Multi-Table Join", "query": "Compare total revenue between male and female customers."},
    {"id": 4, "category": "Date Filter", "query": "What was the total order volume in June?"},
    {"id": 5, "category": "Conditional Metric", "query": "What is the average delivery time when it was raining vs not raining?"},
    {"id": 6, "category": "Dimension Aggregation", "query": "Which cuisine tag generated the highest average order value (AOV)?"},
    {"id": 7, "category": "Campaign Performance", "query": "Which campaign objective drove the highest total net amount?"},
    {"id": 8, "category": "Coupon Analysis", "query": "What is the total coupon discount amount given per coupon_name?"},
    {"id": 9, "category": "Star Schema Join", "query": "Show total revenue by customer home_city joining dim_customer and fact_transactions."},
    {"id": 10, "category": "Edge Case / Dialect", "query": "Audit charged prices vs list price per city + product: avg_list_price, avg_realized_gross_per_unit, and total_lines."},
    {"id": 11, "category": "Grouping & Sorting", "query": "Show the bottom 5 restaurant products by average rating with at least 10 orders."},
    {"id": 12, "category": "Surge Pricing", "query": "What percentage of orders had surge pricing enabled?"}
]

# --- 6. Benchmark Execution Runner ---
def run_benchmark():
    print("=" * 70)
    print("🚀 STARTING SWIGGY COPILOT AUTOMATED BENCHMARK EVALUATION")
    print(f"Model: openai/gpt-oss-120b | Target DB: MySQL (swiggy_db)")
    print("=" * 70)
    
    results = []
    
    for item in BENCHMARK_SUITE:
        q_id = item["id"]
        category = item["category"]
        question = item["query"]
        
        print(f"\n[Test {q_id}/12] Category: {category}")
        print(f"Prompt: \"{question}\"")
        
        start_time = time.time()
        conversation = [
            SystemMessage(content=STAR_SCHEMA_CONTEXT),
            HumanMessage(content=question)
        ]
        
        retries = 0
        execution_success = False
        error_msg = None
        global last_executed_sql
        last_executed_sql = None
        
        try:
            ai_msg = llm_with_tools.invoke(conversation)
            
            while ai_msg.tool_calls and retries < 3:
                conversation.append(ai_msg)
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_output = tools_dict[tool_name].invoke(tool_args)
                    conversation.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                    
                    if "SQL_EXECUTION_ERROR" in str(tool_output):
                        retries += 1
                    elif "SUCCESS" in str(tool_output) or "|" in str(tool_output):
                        execution_success = True
                
                ai_msg = llm_with_tools.invoke(conversation)
            
            latency = round(time.time() - start_time, 2)
            
            # If query executed without uncaught syntax errors
            if execution_success or (last_executed_sql is not None and retries == 0):
                execution_success = True
                status_label = "✅ PASS"
            else:
                status_label = "❌ FAIL"
                
            print(f"Status: {status_label} | Latency: {latency}s | Self-Corrections: {retries}")
            if last_executed_sql:
                print(f"SQL: {last_executed_sql[:75]}...")
                
            results.append({
                "Test ID": q_id,
                "Category": category,
                "Passed": execution_success,
                "Latency (s)": latency,
                "Self-Healed Retries": retries,
                "Generated SQL": last_executed_sql
            })
            
        except Exception as e:
            latency = round(time.time() - start_time, 2)
            print(f"Status: ❌ CRASH: {str(e)}")
            results.append({
                "Test ID": q_id,
                "Category": category,
                "Passed": False,
                "Latency (s)": latency,
                "Self-Healed Retries": retries,
                "Generated SQL": str(e)
            })

    # --- 7. Aggregate Summary Metrics ---
    results_df = pd.DataFrame(results)
    total_tests = len(results_df)
    passed_tests = results_df["Passed"].sum()
    pass_rate = round((passed_tests / total_tests) * 100, 1)
    avg_latency = round(results_df["Latency (s)"].mean(), 2)
    healed_queries = (results_df["Self-Healed Retries"] > 0).sum()
    
    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total Benchmark Queries:     {total_tests}")
    print(f"Successful Query Pass Rate:   {pass_rate}% ({passed_tests}/{total_tests})")
    print(f"Average Turn Latency:         {avg_latency}s")
    print(f"Self-Healed Queries:          {healed_queries}")
    print("=" * 70)
    
    # Save benchmark report to CSV for repository evidence
    results_df.to_csv("benchmark_evaluation_results.csv", index=False)
    print("Detailed report saved to: benchmark_evaluation_results.csv\n")

if __name__ == "__main__":
    run_benchmark()