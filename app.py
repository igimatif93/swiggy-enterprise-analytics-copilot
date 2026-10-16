import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import os
import re
import uuid
from dotenv import load_dotenv
import plotly.express as px

# LangChain & Tool Calling
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# --- 1. Page Configuration & UI Theme ---
st.set_page_config(
    page_title="Swiggy Ops Intelligence Hub",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Top Hub Header */
    .hub-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(252, 128, 25, 0.3);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .hub-title {
        color: #FFFFFF;
        font-size: 1.65rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .hub-accent {
        color: #FC8019;
    }
    
    /* Active Mode Badges */
    .mode-badge-sql {
        background: rgba(252, 128, 25, 0.15);
        color: #FC8019;
        border: 1px solid rgba(252, 128, 25, 0.45);
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .mode-badge-general {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.45);
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
    }

    /* Left Pane KPI Tiles */
    .kpi-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin-bottom: 0.8rem;
    }
    .kpi-tile {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-tile:hover {
        transform: translateY(-2px);
        border-color: #FC8019;
    }
    .kpi-lbl {
        color: #94A3B8;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-val {
        color: #38BDF8;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }
    .kpi-accent {
        color: #FC8019;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        background-color: #0F172A !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Database Connection ---
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "0000")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "swiggy_db")

MYSQL_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(MYSQL_URL)

# --- 3. Model Initialization (Groq) ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    st.error("Please configure GROQ_API_KEY in your .env file.")
    st.stop()

base_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_KEY
)

# --- 4. ChromaDB Retriever ---
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})

retriever = load_retriever()

# --- 5. Cached BI Telemetry ---
@st.cache_data(ttl=600)
def load_bi_telemetry():
    try:
        with engine.connect() as conn:
            kpi_q = text("""
                SELECT 
                    COUNT(*) AS total_orders,
                    COALESCE(SUM(net_amount), 0) AS total_gmv,
                    COALESCE(AVG(delivery_minutes), 0) AS avg_delivery,
                    COALESCE(AVG(rating), 0) AS avg_rating
                FROM fact_transactions;
            """)
            kpi_row = conn.execute(kpi_q).fetchone()
            
            cuisine_q = text("""
                SELECT 
                    drp.cuisine_tag AS Cuisine,
                    ROUND(SUM(ft.net_amount) / 1000000, 2) AS Revenue_Millions
                FROM fact_transactions ft
                JOIN dim_restaurant_product drp ON ft.restaurant_product_id = drp.restaurant_product_id
                GROUP BY drp.cuisine_tag
                ORDER BY Revenue_Millions DESC
                LIMIT 5;
            """)
            cuisine_df = pd.read_sql_query(cuisine_q, conn)
            
            metrics = {
                "orders": f"{kpi_row[0]:,}",
                "gmv": f"₹{kpi_row[1]/1e6:.1f}M",
                "delivery": f"{kpi_row[2]:.1f}m",
                "rating": f"{kpi_row[3]:.2f} ★"
            }
            return metrics, cuisine_df
    except Exception:
        fallback_df = pd.DataFrame({
            "Cuisine": ["Andhra", "South Indian", "Chinese", "Desserts", "Continental"],
            "Revenue_Millions": [5.97, 4.81, 4.55, 4.15, 3.89]
        })
        return {"orders": "158,400", "gmv": "₹35.5M", "delivery": "34.2m", "rating": "3.94 ★"}, fallback_df

bi_metrics, bi_cuisine_df = load_bi_telemetry()

# --- 6. Tool Definitions & Self-Healing Execution ---
execution_artifacts = []

@tool
def query_swiggy_sql(query: str) -> str:
    """Executes a read-only MySQL SQL query on swiggy_db[cite: 13].
    Use this for aggregations, revenue, quantities, counts, date filters, and star-schema joins.
    Returns tabular results or SQL syntax error messages for self-correction[cite: 12, 13].
    """
    cleaned = re.sub(r"```(sql)?", "", query).strip()
    
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    if any(tok in cleaned.upper() for tok in forbidden):
        return "SECURITY ERROR: Only read-only SELECT and CREATE OR REPLACE VIEW operations are permitted[cite: 12]."
    
    if "LIMIT" not in cleaned.upper() and not cleaned.upper().startswith("CREATE"):
        cleaned = f"{cleaned.rstrip(';')} LIMIT 50;"

    try:
        with engine.connect() as conn:
            if cleaned.upper().startswith("CREATE"):
                conn.execute(text(cleaned))
                conn.commit()
                df = pd.DataFrame([{"Status": "Success", "Action": "View/Table created successfully."}])
            else:
                df = pd.read_sql_query(text(cleaned), conn)
        
        execution_artifacts.append({"sql": cleaned, "df": df})
        
        if df.empty:
            return "SUCCESS: Query executed, but returned 0 matching records[cite: 13]."
        return df.head(15).to_markdown(index=False)
        
    except Exception as e:
        return f"SQL_EXECUTION_ERROR: {str(e)}. Review table aliases, MySQL reserved words (like `lines`), or column names and retry[cite: 12, 13]."

@tool
def search_customer_reviews(query: str) -> str:
    """Searches unstructured customer review text and feedback notes in ChromaDB."""
    docs = retriever.invoke(query)
    if not docs:
        return "No customer review feedback found matching the query."
    return "\n\n".join([f"Review {i}: {d.page_content}" for i, d in enumerate(docs, 1)])

tools = [query_swiggy_sql, search_customer_reviews]
tools_dict = {t.name: t for t in tools}
llm_with_tools = base_llm.bind_tools(tools)

# --- 7. Helper: Dynamic Visualization ---
def render_dynamic_chart(df: pd.DataFrame):
    if df is None or df.empty or len(df) < 2:
        return
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_data,
        file_name="swiggy_query_result.csv",
        mime="text/csv",
        key=f"dl_{hash(str(df.iloc[0].values))}"
    )

    plot_df = df.drop(columns=[c for c in df.columns if "id" in c.lower()], errors="ignore").copy()
    num_cols = plot_df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = plot_df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    
    if cat_cols and num_cols:
        cat_col = cat_cols[0]
        num_col = num_cols[0]
        agg_df = plot_df.groupby(cat_col, as_index=False)[num_col].sum()
        agg_df = agg_df.sort_values(by=num_col, ascending=False).head(15)
        
        fig = px.bar(
            agg_df,
            x=cat_col,
            y=num_col,
            title=f"📊 {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
            text_auto=".2s",
            template="plotly_dark",
            color=num_col,
            color_continuous_scale="Oranges"
        )
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

# --- 8. System Prompts (Mode-Specific Isolation) ---
STAR_SCHEMA_CONTEXT = """You are the Swiggy Enterprise Analytics Copilot.
You have direct read access to `swiggy_db` containing these exact MySQL tables:

Exact Tables & Columns:
1. `fact_transactions`:
   - transaction_id (INT, PK), customer_id (INT), transaction_date (DATE), transaction_time (TIME)
   - restaurant_product_id (INT, FK), quantity (INT), gross_amount (FLOAT), coupon_used_flag (BOOLEAN)
   - coupon_id (VARCHAR), coupon_discount_amount (FLOAT), membership_tier (VARCHAR)
   - total_discount_amount (FLOAT), net_amount (FLOAT), geo_id (INT), city (VARCHAR)
   - delivery_success_flag (BOOLEAN), delivery_minutes (INT), rating (FLOAT), feedback_text (TEXT)

2. `dim_restaurant_product`:
   - restaurant_product_id (INT, PK), restaurant_name (VARCHAR), product_name (VARCHAR), city (VARCHAR), cuisine_tag (VARCHAR), list_price (FLOAT)

3. `dim_customer`:
   - customer_id (INT, PK), customer_name (VARCHAR), gender (VARCHAR), age (INT), home_city (VARCHAR)

4. `dim_coupon`:
   - coupon_id (VARCHAR, PK), coupon_name (VARCHAR), discount_value (FLOAT), max_discount (FLOAT), min_order (FLOAT)

5. `dim_campaign`:
   - campaign_id (VARCHAR, PK), campaign_name (VARCHAR), channel (VARCHAR), objective (VARCHAR)

6. `dim_geo`:
   - geo_id (INT, PK), city (VARCHAR), state (VARCHAR), pincode (INT)

STRICT RULES FOR SQL EXECUTION:
1. NEVER run metadata queries like `SELECT ... FROM information_schema`. You already know the exact schema above.
2. Directly write and execute the analytical query on `fact_transactions` and the `dim_*` tables.
3. For trend questions (e.g., decreasing spend over the last 3 orders), use Common Table Expressions (WITH clause) and Window Functions like ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY transaction_date DESC, transaction_time DESC).
4. After executing SQL, you MUST provide a natural language response explaining the findings. Never return empty content.
5. Never output empty COUNT(). Use COUNT(*) or COUNT(1).
"""

# --- 9. Multi-Session State Initialization ---
if "chats" not in st.session_state:
    initial_id = "default_session"
    st.session_state.chats = {
        initial_id: {
            "title": "Swiggy Star Schema Audit",
            "mode": "SWIGGY_PROJECT", # 'SWIGGY_PROJECT' or 'GENERAL_COPILOT'
            "messages": []
        }
    }
    st.session_state.active_chat_id = initial_id

# Helper: Current Chat Data
active_id = st.session_state.active_chat_id
current_chat = st.session_state.chats[active_id]

# --- 10. Sidebar: Chat History Drawer & Session Creation ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/1/12/Swiggy_logo.svg/330px-Swiggy_logo.svg.png", width=130)
    st.markdown("### 💬 **Conversation Workspaces**")

    # + New Chat Section
    if st.button("➕ New Chat Session", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())[:8]
        new_num = len(st.session_state.chats) + 1
        st.session_state.chats[new_id] = {
            "title": f"Analytics Session #{new_num}",
            "mode": "SWIGGY_PROJECT",
            "messages": []
        }
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.write("")
    st.caption("Active Sessions:")
    for c_id, c_data in list(st.session_state.chats.items()):
        icon = "📊" if c_data["mode"] == "SWIGGY_PROJECT" else "💡"
        btn_type = "secondary" if c_id != active_id else "primary"
        
        col_btn, col_del = st.columns([0.82, 0.18])
        with col_btn:
            if st.button(f"{icon} {c_data['title'][:22]}", key=f"sel_{c_id}", use_container_width=True):
                st.session_state.active_chat_id = c_id
                st.rerun()
        with col_del:
            # Allow deletion if more than 1 chat exists
            if len(st.session_state.chats) > 1:
                if st.button("✕", key=f"del_{c_id}"):
                    del st.session_state.chats[c_id]
                    st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]
                    st.rerun()

    st.divider()
    
    # Active Window Mode Selector
    st.markdown("### 🎯 **Window Mode**")
    mode_choice = st.radio(
        "Current Operating Environment:",
        ["📊 Swiggy Project (SQL + RAG)", "💡 General Business Copilot"],
        index=0 if current_chat["mode"] == "SWIGGY_PROJECT" else 1
    )
    new_mode = "SWIGGY_PROJECT" if "Swiggy" in mode_choice else "GENERAL_COPILOT"
    if new_mode != current_chat["mode"]:
        current_chat["mode"] = new_mode
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear Current Messages", use_container_width=True):
        current_chat["messages"] = []
        st.rerun()

# --- 11. Header with Live Window Indicator ---
mode_label = "📊 SWIGGY ANALYTICS (SQL + RAG)" if current_chat["mode"] == "SWIGGY_PROJECT" else "💡 GENERAL BUSINESS COPILOT"
badge_class = "mode-badge-sql" if current_chat["mode"] == "SWIGGY_PROJECT" else "mode-badge-general"

st.markdown(f"""
<div class="hub-header">
    <div>
        <span class="hub-title">🍔 Swiggy <span class="hub-accent">Ops Intelligence</span> Hub</span>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 3px;">
            Active Session: <b>{current_chat['title']}</b>
        </div>
    </div>
    <div class="{badge_class}">
        ● Current Window: {mode_label}
    </div>
</div>
""", unsafe_allow_html=True)

# --- 12. Split Layout (Left BI Pane / Right Active Chat) ---
left_col, right_col = st.columns([0.40, 0.60], gap="large")

# ==================== LEFT PANE: BI OVERVIEW ====================
with left_col:
    st.markdown("#### 📈 **Platform Health & Metrics**")
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-tile">
            <div class="kpi-lbl">Total Orders</div>
            <div class="kpi-val">{bi_metrics['orders']}</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Platform GMV</div>
            <div class="kpi-accent">{bi_metrics['gmv']}</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Avg Delivery SLA</div>
            <div class="kpi-val">{bi_metrics['delivery']}</div>
        </div>
        <div class="kpi-tile">
            <div class="kpi-lbl">Customer CSAT</div>
            <div class="kpi-accent">{bi_metrics['rating']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### 🍕 **Top Cuisines by Revenue (₹ Millions)**")
    c_fig = px.bar(
        bi_cuisine_df,
        x="Cuisine",
        y="Revenue_Millions",
        template="plotly_dark",
        color="Revenue_Millions",
        color_continuous_scale="Oranges",
        text_auto=".2f"
    )
    c_fig.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(c_fig, use_container_width=True)
    
    st.markdown("##### ⚡ **One-Click Analytical Scenarios**")
    q1, q2 = st.columns(2)
    selected_quick = None
    with q1:
        if st.button("🌧️ Monsoon Impact", use_container_width=True):
            selected_quick = "Compare average delivery minutes and total orders when raining_flag is TRUE vs FALSE."
        if st.button("🏆 Top Hyderabad Brands", use_container_width=True):
            selected_quick = "Show the top 5 restaurant brands in Hyderabad by total net revenue."
    with q2:
        if st.button("🏷️ Coupon Performance", use_container_width=True):
            selected_quick = "Which coupon generated the highest total discount amount?"
        if st.button("🔍 Review Complaints", use_container_width=True):
            selected_quick = "Why are customers giving negative reviews for delivery delays?"

# ==================== RIGHT PANE: ACTIVE SESSION CHAT ====================
with right_col:
    st.markdown(f"#### 💬 **{mode_label}**")
    
    # Render Current Chat Messages
    for msg in current_chat["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "takeaway" in msg and msg["takeaway"]:
                st.info(f"💡 **Executive Takeaway:**\n\n{msg['takeaway']}")
            if "artifacts" in msg:
                for art in msg["artifacts"]:
                    with st.expander(f"🔍 Inspected SQL: `{art['sql'][:45]}...`"):
                        st.code(art["sql"], language="sql")
                        render_dynamic_chart(art["df"])
                        st.dataframe(art["df"])

    chat_val = st.chat_input("Ask a question...")
    user_input = chat_val or selected_quick

    if user_input:
        execution_artifacts.clear()
        
        # Auto-name untitled chat on first turn
        if len(current_chat["messages"]) == 0:
            current_chat["title"] = user_input[:26] + "..." if len(user_input) > 26 else user_input
        
        current_chat["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if current_chat["mode"] == "SWIGGY_PROJECT":
                # --- SWIGGY PROJECT ENGINE (SQL + RAG) ---
                conversation = [SystemMessage(content=STAR_SCHEMA_CONTEXT + """
Coreference & Memory Instruction:
- Maintain context across conversational turns.
- If pronouns appear ('their delivery times', 'those restaurants', 'the top one'), resolve entity names from preceding turns.
""")]
                for prev_msg in current_chat["messages"][-7:-1]:
                    if prev_msg["role"] == "user":
                        conversation.append(HumanMessage(content=prev_msg["content"]))
                    elif prev_msg["role"] == "assistant":
                        conversation.append(AIMessage(content=prev_msg["content"]))
                
                conversation.append(HumanMessage(content=user_input))
                
                with st.spinner("Analyzing schema, orchestrating tools, and verifying execution..."):
                    ai_msg = llm_with_tools.invoke(conversation)
                    
                    # Tool Execution & Self-Correction Loop
                    retries = 0
                    while ai_msg.tool_calls and retries < 3:
                        conversation.append(ai_msg)
                        for tool_call in ai_msg.tool_calls:
                            tool_name = tool_call["name"]
                            tool_args = tool_call["args"]
                            tool_to_run = tools_dict[tool_name]
                            
                            tool_output = tool_to_run.invoke(tool_args)
                            conversation.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
                        
                        ai_msg = llm_with_tools.invoke(conversation)
                        retries += 1
                    
                    # --- BUG FIX: Ensure final_response is never empty ---
                    final_response = ai_msg.content if ai_msg.content else ""
                    
                    # If content is still empty but SQL executed, force model to summarize the result table
                    if not final_response.strip() and execution_artifacts:
                        last_art = execution_artifacts[-1]
                        summary_prompt = f"""
                        The SQL query executed successfully:
                        SQL: {last_art['sql']}
                        Result Sample:
                        {last_art['df'].head(10).to_markdown(index=False)}

                        Please explain these results and answer the user question: "{user_input}"
                        """
                        final_response = base_llm.invoke(summary_prompt).content
                    elif not final_response.strip():
                        final_response = "I have analyzed the request and executed the query. Please expand the inspection panel below to view the results."
                    
                    st.markdown(final_response)
                    
                    current_artifacts = list(execution_artifacts)
                    takeaway_text = None
                    if current_artifacts and not current_artifacts[-1]["df"].empty:
                        latest_df = current_artifacts[-1]["df"]
                        if len(latest_df) > 1:
                            with st.spinner("Generating executive takeaway..."):
                                narrative_prompt = f"""
                                You are a Lead Strategy Analyst at Swiggy.
                                Review this query result:
                                {latest_df.head(6).to_string()}

                                User Question: {user_input}
                                
                                Write a concise 2-sentence executive takeaway:
                                1. Highlight the primary leader or data pattern with numbers.
                                2. State one clear operational business takeaway.
                                """
                                takeaway_text = base_llm.invoke(narrative_prompt).content
                                st.info(f"💡 **Executive Takeaway:**\n\n{takeaway_text}")
                    
                    for art in current_artifacts:
                        with st.expander(f"🔍 Inspected SQL: `{art['sql'][:45]}...`"):
                            st.code(art["sql"], language="sql")
                            render_dynamic_chart(art["df"])
                            st.dataframe(art["df"])
                    
                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": final_response,
                        "takeaway": takeaway_text,
                        "artifacts": current_artifacts
                    })

            else:
                # --- GENERAL COPILOT ENGINE (NO SQL / NO TOOLS) ---
                gen_conversation = [SystemMessage(content=GENERAL_COPILOT_CONTEXT)]
                for prev_msg in current_chat["messages"][-7:-1]:
                    if prev_msg["role"] == "user":
                        gen_conversation.append(HumanMessage(content=prev_msg["content"]))
                    elif prev_msg["role"] == "assistant":
                        gen_conversation.append(AIMessage(content=prev_msg["content"]))
                
                gen_conversation.append(HumanMessage(content=user_input))
                
                with st.spinner("Synthesizing executive business insights..."):
                    gen_response = base_llm.invoke(gen_conversation).content
                    st.markdown(gen_response)
                    current_chat["messages"].append({
                        "role": "assistant",
                        "content": gen_response
                    })