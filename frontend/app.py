"""
frontend/app.py

FinTrust Compass — Streamlit Employee Assistant UI

Single-page interface with chat as the default view.
Additional tools are opened from a plus-menu action sheet and rendered
inline on the same page.
"""

import json
import uuid
import requests
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "http://localhost:8000"

DOMAIN_COLORS = {
    "loans":           "#2196F3",   # blue
    "deposits":        "#4CAF50",   # green
    "accounts":        "#9C27B0",   # purple
    "cards":           "#FF9800",   # orange
    "compliance":      "#F44336",   # red
    "digital_banking": "#00BCD4",   # teal
}

DOMAIN_ICONS = {
    "loans":           "🏦",
    "deposits":        "💰",
    "accounts":        "🏧",
    "cards":           "💳",
    "compliance":      "⚖️",
    "digital_banking": "📱",
}

SAMPLE_QUESTIONS = {
    "🏦 Loans": [
        "What are the eligibility criteria for a home loan?",
        "What is the maximum LTV ratio for a vehicle loan?",
        "What documents are required for a personal loan?",
        "Can a self-employed person apply for a home loan?",
    ],
    "💰 Deposits": [
        "What is the interest rate for a 1-year fixed deposit?",
        "What happens if I break a recurring deposit early?",
        "What is the minimum amount to open a fixed deposit?",
        "Can NRIs open fixed deposits at FinTrust?",
    ],
    "🏧 Accounts": [
        "What is the minimum balance for a savings account?",
        "What KYC documents are needed for a current account?",
        "What are the charges for non-maintenance of minimum balance?",
        "What are the benefits of a FinTrust salary account?",
    ],
    "💳 Cards": [
        "What are the annual fees for FinTrust credit cards?",
        "How does the credit card billing cycle work?",
        "What is the daily ATM withdrawal limit on debit cards?",
        "How do I dispute a credit card transaction?",
    ],
    "⚖️ Compliance": [
        "What are the AML reporting requirements for suspicious transactions?",
        "What is the threshold for CTR reporting?",
        "What are the KYC refresh intervals for different customer categories?",
        "What actions must be taken for PEP customers?",
    ],
    "📱 Digital Banking": [
        "What 2FA methods does FinTrust support for internet banking?",
        "What are the transaction limits for NEFT and RTGS?",
        "How does FinTrust handle failed UPI transactions?",
        "What is the daily limit for mobile banking transfers?",
    ],
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinTrust Compass",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stChatMessage { border-radius: 12px; margin-bottom: 0.5rem; }
    .domain-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        color: white;
        margin-bottom: 6px;
    }
    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #dee2e6;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 13px;
    }
    .confidence-high   { color: #28a745; font-weight: 600; }
    .confidence-medium { color: #fd7e14; font-weight: 600; }
    .confidence-low    { color: #dc3545; font-weight: 600; }
    div[data-testid="stSidebarContent"] h2 { color: #1a73e8; }
    .elig-pass   { background:#d4edda; border-left:4px solid #28a745; padding:8px 12px; border-radius:4px; margin:3px 0; }
    .elig-cond   { background:#fff3cd; border-left:4px solid #fd7e14; padding:8px 12px; border-radius:4px; margin:3px 0; }
    .elig-fail   { background:#f8d7da; border-left:4px solid #dc3545; padding:8px 12px; border-radius:4px; margin:3px 0; }
    .elig-ns     { background:#e2e3e5; border-left:4px solid #6c757d; padding:8px 12px; border-radius:4px; margin:3px 0; }
    .decision-PASS       { color:#155724; background:#d4edda; padding:12px 20px; border-radius:8px; font-size:18px; font-weight:700; }
    .decision-CONDITIONAL{ color:#856404; background:#fff3cd; padding:12px 20px; border-radius:8px; font-size:18px; font-weight:700; }
    .decision-FAIL       { color:#721c24; background:#f8d7da; padding:12px 20px; border-radius:8px; font-size:18px; font-weight:700; }
    .decision-ERROR      { color:#383d41; background:#e2e3e5; padding:12px 20px; border-radius:8px; font-size:18px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "show_tool_sheet" not in st.session_state:
    st.session_state.show_tool_sheet = False
if "dialog_to_open" not in st.session_state:
    st.session_state.dialog_to_open = None


# ---------------------------------------------------------------------------
# Cached API fetchers (module-level so dialogs share the cache)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _fetch_products():
    try:
        r = requests.get(f"{API_BASE}/eligibility/products", timeout=5)
        return r.json().get("products", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def _fetch_applicant_categories():
    try:
        r = requests.get(f"{API_BASE}/checklist/categories", timeout=5)
        return r.json().get("categories", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def _fetch_calc_products():
    try:
        r = requests.get(f"{API_BASE}/calculator/products", timeout=5)
        return r.json().get("products", [])
    except Exception:
        return []


@st.cache_data(ttl=300)
def _fetch_compare_products():
    try:
        r = requests.get(f"{API_BASE}/compare/products", timeout=5)
        d = r.json()
        return d.get("products", []), d.get("quick_pairs", [])
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://placehold.co/240x60/1a73e8/white?text=FinTrust+Compass", width=240)
    st.markdown("### 🧭 Employee Assistant")
    #st.caption("Powered by Gemini 2.5 Flash · RAG · Multi-Agent")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE}/health", timeout=3).json()
        st.success(f"API online · {health['vector_store_docs']:,} policy chunks loaded")
    except Exception:
        st.error("API offline — start the server:\n`uvicorn api.main:app --reload`")

    st.divider()

    # Sample questions
    st.markdown("### 💬 Sample Questions")
    for category, questions in SAMPLE_QUESTIONS.items():
        with st.expander(category):
            for q in questions:
                if st.button(q, key=f"btn_{q[:30]}", use_container_width=True):
                    st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        # Full reset: clear chat plus tool/form/result state.
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        # Re-initialize defaults expected by this app.
        st.session_state.messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.pending_query = None
        st.session_state.show_tool_sheet = False
        st.session_state.dialog_to_open = None
        st.rerun()

    st.caption(f"Session: `{st.session_state.conversation_id[:8]}...`")


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Eligibility Checker
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("✅ Product Eligibility Checker", width="large")
def show_eligibility():
    st.caption(
        "Fill in the customer's details and the AI agent will evaluate eligibility "
        "against FinTrust's official policy documents."
    )

    products = _fetch_products()
    if not products:
        st.error("Cannot load product list. Make sure the API server is running.")
        return

    product_map = {p["label"]: p["product_id"] for p in products}

    col_prod, col_domain = st.columns([3, 1])
    with col_prod:
        selected_label = st.selectbox("Select product", list(product_map.keys()))
    selected_product = product_map[selected_label]
    selected_domain  = next((p["domain"] for p in products if p["product_id"] == selected_product), "")
    with col_domain:
        domain_color = DOMAIN_COLORS.get(selected_domain, "#607d8b")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="domain-badge" style="background:{domain_color}">'
            f'{DOMAIN_ICONS.get(selected_domain,"")}&nbsp;{selected_domain.replace("_"," ").title()}'
            f'</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📋 Applicant Profile")
    st.caption("Fill in all available fields. Leave blank if not known — the agent will flag missing information.")

    LOAN_PRODUCTS    = {"home_loan", "personal_loan", "vehicle_loan"}
    DEPOSIT_PRODUCTS = {"fixed_deposit", "recurring_deposit"}
    ACCOUNT_PRODUCTS = {"savings_account", "current_account", "salary_account"}
    CARD_PRODUCTS    = {"credit_card", "debit_card"}

    profile = {}
    c1, c2, c3 = st.columns(3)

    if selected_product in LOAN_PRODUCTS:
        with c1:
            profile["age"]             = st.number_input("Age (years)", 18, 80, 35, key="e_age")
            profile["employment_type"] = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed Professional", "Self-Employed Business", "Pensioner"],
                key="e_emp",
            )
        with c2:
            profile["monthly_income"]    = st.number_input("Monthly Income (₹)", 0, 10_000_000, 75_000, step=5_000, key="e_inc")
            profile["cibil_score"]       = st.number_input("CIBIL Score", 300, 900, 720, key="e_cibil")
        with c3:
            profile["loan_amount"]       = st.number_input("Loan Amount Required (₹)", 0, 100_000_000, 2_500_000, step=100_000, key="e_lamt")
            profile["loan_tenure_years"] = st.number_input("Desired Tenure (years)", 1, 30, 15, key="e_ten")

        if selected_product == "home_loan":
            c4, c5 = st.columns(2)
            with c4:
                profile["property_value"] = st.number_input("Property Value (₹)", 0, 100_000_000, 4_000_000, step=100_000, key="e_pval")
                profile["property_type"]  = st.selectbox("Property Type", ["Under Construction", "Ready to Move", "Resale", "Plot + Construction"], key="e_ptyp")
            with c5:
                profile["co_applicant"]   = st.selectbox("Co-Applicant?", ["No", "Yes — Spouse", "Yes — Parent", "Yes — Other"], key="e_coa")
                profile["existing_loans"] = st.selectbox("Existing Loans?", ["None", "1 Loan", "2+ Loans"], key="e_exl")
        elif selected_product == "vehicle_loan":
            c4, c5 = st.columns(2)
            with c4:
                profile["vehicle_type"]  = st.selectbox("Vehicle Type", ["New Car", "Used Car (< 3 yrs)", "Used Car (3-7 yrs)", "Two Wheeler", "Commercial Vehicle"], key="e_vtyp")
                profile["vehicle_value"] = st.number_input("Vehicle On-Road Price (₹)", 0, 10_000_000, 800_000, step=50_000, key="e_vval")
            with c5:
                profile["down_payment"]    = st.number_input("Down Payment (₹)", 0, 10_000_000, 160_000, step=10_000, key="e_dp")
                profile["driving_license"] = st.selectbox("Valid Driving License?", ["Yes", "No"], key="e_dl")
        elif selected_product == "personal_loan":
            c4, c5 = st.columns(2)
            with c4:
                profile["company_category"]      = st.selectbox("Employer Category", ["Govt / PSU", "Listed Company", "Private Ltd", "Partnership / Proprietorship"], key="e_comp")
                profile["work_experience_years"] = st.number_input("Total Work Experience (years)", 0, 40, 5, key="e_wexp")
            with c5:
                profile["existing_emi"] = st.number_input("Existing Monthly EMI obligations (₹)", 0, 500_000, 0, step=1_000, key="e_emi")
                profile["purpose"]      = st.selectbox("Loan Purpose", ["Medical Emergency", "Education", "Travel", "Home Renovation", "Wedding", "Other"], key="e_pur")

    elif selected_product in DEPOSIT_PRODUCTS:
        with c1:
            profile["customer_type"]  = st.selectbox("Customer Type", ["Resident Individual", "NRI", "HUF", "Company / Trust"], key="e_cust_d")
            profile["age"]            = st.number_input("Age (years)", 0, 120, 35, key="e_age_d")
        with c2:
            profile["deposit_amount"] = st.number_input("Deposit Amount (₹)", 0, 100_000_000, 50_000, step=1_000, key="e_damt")
            profile["tenure_months"]  = st.number_input("Desired Tenure (months)", 1, 120, 12, key="e_tmon")
        with c3:
            profile["kyc_complete"]     = st.selectbox("KYC Complete?", ["Yes", "No", "Partial"], key="e_kyc")
            profile["existing_account"] = st.selectbox("Existing FinTrust Account?", ["Yes", "No"], key="e_exacc")
        if selected_product == "recurring_deposit":
            profile["monthly_installment"] = st.number_input("Monthly Installment (₹)", 100, 1_000_000, 5_000, step=100, key="e_rd_inst")

    elif selected_product in ACCOUNT_PRODUCTS:
        with c1:
            profile["customer_type"] = st.selectbox("Customer Type", ["Individual", "NRI", "Minor (Guardian-operated)", "HUF"], key="e_cust_a")
            profile["age"]           = st.number_input("Age (years)", 0, 120, 30, key="e_age_a")
        with c2:
            profile["kyc_documents"] = st.multiselect(
                "KYC Documents Available",
                ["Aadhaar", "PAN", "Passport", "Voter ID", "Driving License", "Utility Bill"],
                default=["Aadhaar", "PAN"],
                key="e_kycdocs",
            )
        with c3:
            profile["annual_income"] = st.number_input("Annual Income (₹)", 0, 100_000_000, 600_000, step=10_000, key="e_aninc")
        if selected_product == "current_account":
            c4, c5 = st.columns(2)
            with c4:
                profile["business_type"] = st.selectbox("Business Type", ["Proprietorship", "Partnership", "Private Ltd", "LLP", "Trust / NGO"], key="e_btype")
            with c5:
                profile["business_vintage_years"] = st.number_input("Business Vintage (years)", 0, 100, 3, key="e_bvint")
        elif selected_product == "salary_account":
            c4, c5 = st.columns(2)
            with c4:
                profile["employer_name"]   = st.text_input("Employer Name", key="e_empn")
                profile["employer_tie_up"] = st.selectbox("Employer Tie-Up with FinTrust?", ["Yes", "No", "Unknown"], key="e_emptu")
            with c5:
                profile["monthly_salary"] = st.number_input("Monthly Salary (₹)", 0, 10_000_000, 50_000, step=1_000, key="e_sal")

    elif selected_product in CARD_PRODUCTS:
        with c1:
            profile["age"]             = st.number_input("Age (years)", 18, 75, 30, key="e_age_c")
            profile["employment_type"] = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed Professional", "Self-Employed Business"],
                key="e_emp_c",
            )
        with c2:
            profile["monthly_income"] = st.number_input("Monthly Income (₹)", 0, 10_000_000, 60_000, step=5_000, key="e_inc_c")
            profile["cibil_score"]    = st.number_input("CIBIL Score", 300, 900, 700, key="e_cibil_c")
        with c3:
            if selected_product == "credit_card":
                profile["existing_credit_cards"] = st.number_input("Existing Credit Cards", 0, 20, 1, key="e_cc")
                profile["card_type_requested"]   = st.selectbox("Card Type", ["Everyday", "Classic", "Premium", "Super Premium", "Business"], key="e_ctype")
            elif selected_product == "debit_card":
                profile["savings_account_held"] = st.selectbox("Holds FinTrust Savings Account?", ["Yes", "No"], key="e_sah")
                profile["account_status"]       = st.selectbox("Account Status", ["Active", "Inactive", "Frozen"], key="e_astat")

    notes = st.text_area("Additional notes / special circumstances (optional)", height=70, key="e_notes")
    if notes.strip():
        profile["additional_notes"] = notes.strip()

    st.markdown("---")
    run_col, _ = st.columns([2, 5])
    with run_col:
        run_button = st.button("🔍 Check Eligibility", type="primary", use_container_width=True, key="e_run")

    if run_button:
        if not profile:
            st.warning("Please fill in at least one profile field before checking.")
        else:
            with st.spinner(f"Evaluating eligibility for **{selected_label}**..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/eligibility",
                        json={"product": selected_product, "profile": profile},
                        timeout=300,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                except requests.exceptions.ConnectionError:
                    st.error("API server is offline. Start it with: `uvicorn api.main:app --reload`")
                    return
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
            st.session_state["elig_result"]        = result
            st.session_state["elig_product_label"] = selected_label
            st.session_state["elig_finalised"]     = None

    elig_result = st.session_state.get("elig_result")
    elig_label  = st.session_state.get("elig_product_label", "")

    if elig_result:
        result   = elig_result
        decision = result["decision"]
        decision_icons  = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌", "ERROR": "🚫"}
        decision_labels = {
            "PASS": "ELIGIBLE", "CONDITIONAL": "CONDITIONALLY ELIGIBLE",
            "FAIL": "NOT ELIGIBLE", "ERROR": "ERROR",
        }
        st.markdown(
            f'<div class="decision-{decision}">'
            f'{decision_icons.get(decision,"ℹ️")}&nbsp; {decision_labels.get(decision, decision)}'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {result['decision_reason']}")

        if result.get("conditions"):
            st.warning("**Items requiring further review or documentation:**")
            for cond in result["conditions"]:
                st.markdown(f"- {cond}")

        st.markdown("---")
        st.markdown("#### 📊 Criterion-by-Criterion Breakdown")
        css_map  = {"PASS": "elig-pass", "CONDITIONAL": "elig-cond", "FAIL": "elig-fail"}
        icon_map = {"PASS": "✅", "CONDITIONAL": "⚠️", "FAIL": "❌"}

        for c in result.get("criteria", []):
            status  = c["status"]
            css_cls = css_map.get(status, "elig-ns")
            icon    = icon_map.get(status, "")
            st.markdown(
                f'<div class="{css_cls}">'
                f'<b>{icon} {c["criterion"]}</b><br>'
                f'<span style="font-size:12px;color:#555">'
                f'<b>Requirement:</b> {c["requirement"]}&nbsp;&nbsp;|&nbsp;&nbsp;'
                f'<b>Applicant:</b> {c["applicant_value"]}</span><br>'
                f'<span style="font-size:13px">{c["reason"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 🧑‍💼 Officer Review & Decision Override")
        st.caption(
            "The AI check is advisory. The relationship manager can override the decision "
            "and add a justification before finalising."
        )

        if st.session_state.get("elig_finalised"):
            fin = st.session_state["elig_finalised"]
            banner_fn = st.success if "APPROVE" in fin["decision"] or fin["decision"] == "PASS" else (
                st.error if "REJECT" in fin["decision"] or fin["decision"] == "FAIL" else st.info
            )
            banner_fn(
                f"✔️ Decision recorded: **{fin['decision']}** for **{fin['product']}**"
                + (f"\n\n🗒️ Officer notes: {fin['notes']}" if fin["notes"] else "")
            )
            if st.button("🔄 Revise Decision", key="elig_revise"):
                st.session_state["elig_finalised"] = None
                st.rerun()
        else:
            override_col, note_col = st.columns([1, 2])
            with override_col:
                officer_decision = st.radio(
                    "Officer Decision",
                    ["Accept AI Decision", "Override → APPROVE", "Override → REJECT",
                     "Escalate to Credit Committee"],
                    index=0,
                    key="elig_officer_radio",
                )
            with note_col:
                officer_notes = st.text_area(
                    "Officer Justification / Notes", height=120,
                    placeholder="Add your reasoning here if overriding the AI decision...",
                    key="elig_officer_notes",
                )
            finalize_col, _ = st.columns([2, 5])
            with finalize_col:
                if st.button("📝 Finalise & Save Decision", use_container_width=True, key="elig_finalise_btn"):
                    final = (officer_decision if officer_decision != "Accept AI Decision" else decision)
                    final_clean = (
                        final.replace("Override → ", "")
                             .replace("Escalate to Credit Committee", "ESCALATED")
                    )
                    st.session_state["elig_finalised"] = {
                        "decision": final_clean,
                        "product":  elig_label,
                        "notes":    officer_notes.strip(),
                    }
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Document Checklist Generator
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("📋 Document Checklist Generator", width="large")
def show_checklist():
    st.caption(
        "Select a product and applicant type. The AI agent retrieves the exact "
        "documents required from FinTrust's official policy documents."
    )

    cl_products   = _fetch_products()
    cl_categories = _fetch_applicant_categories()

    if not cl_products or not cl_categories:
        st.error("Cannot load options. Make sure the API server is running.")
        return

    cl_product_map = {p["label"]: p["product_id"] for p in cl_products}

    cl_col1, cl_col2 = st.columns(2)
    with cl_col1:
        cl_selected_label   = st.selectbox("Product", list(cl_product_map.keys()), key="cl_product")
        cl_selected_product = cl_product_map[cl_selected_label]
    with cl_col2:
        cl_selected_category = st.selectbox("Applicant Type", cl_categories, key="cl_category")

    cl_extra = st.text_input(
        "Additional context (optional)",
        placeholder="e.g. under-construction property, co-applicant is spouse, NRI remittance account",
        key="cl_extra",
    )

    cl_btn_col, _ = st.columns([2, 5])
    with cl_btn_col:
        cl_run = st.button("📋 Generate Checklist", type="primary", use_container_width=True, key="cl_run")

    if cl_run:
        with st.spinner(f"Generating checklist for **{cl_selected_label}** — **{cl_selected_category}**..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/checklist",
                    json={
                        "product": cl_selected_product,
                        "applicant_category": cl_selected_category,
                        "additional_context": cl_extra,
                    },
                    timeout=300,
                )
                resp.raise_for_status()
                cl_result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("API server is offline.")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if cl_result.get("error"):
            st.error(cl_result["error"])
            return

        checklist = cl_result.get("checklist", [])
        if not checklist:
            st.warning("No documents returned. Try adjusting the product or applicant type.")
            return

        mandatory_count = sum(1 for c in checklist if c["mandatory"])
        optional_count  = len(checklist) - mandatory_count
        st.success(
            f"**{cl_result['product_label']}** · **{cl_result['applicant_category']}** — "
            f"{len(checklist)} documents ({mandatory_count} mandatory, {optional_count} optional)"
        )

        categories: dict = {}
        for item in checklist:
            categories.setdefault(item["category"], []).append(item)

        for cat_name, items in categories.items():
            mandatory_in_cat = sum(1 for i in items if i["mandatory"])
            with st.expander(
                f"**{cat_name}** — {len(items)} docs ({mandatory_in_cat} mandatory)",
                expanded=True,
            ):
                for item in items:
                    badge_color = "#dc3545" if item["mandatory"] else "#6c757d"
                    badge_label = "Mandatory" if item["mandatory"] else "Optional"
                    alt_text = (
                        f"<br><span style='font-size:12px;color:#1565C0'>"
                        f"✦ Acceptable alternatives: {' &nbsp;|&nbsp; '.join(item['alternatives'])}</span>"
                        if item.get("alternatives") else ""
                    )
                    notes_text = (
                        f"<br><span style='font-size:12px;color:#555'>📝 {item['notes']}</span>"
                        if item.get("notes") else ""
                    )
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-left:4px solid {badge_color};'
                        f'padding:10px 14px;border-radius:4px;margin:4px 0">'
                        f'<span style="background:{badge_color};color:white;font-size:11px;'
                        f'font-weight:600;padding:2px 8px;border-radius:10px">{badge_label}</span>'
                        f'&nbsp;&nbsp;<b>{item["document"]}</b>'
                        f'{alt_text}{notes_text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("---")
        st.markdown("#### 📤 Export Checklist")
        lines = [
            "FINTRUST BANK — DOCUMENT CHECKLIST",
            f"Product        : {cl_result['product_label']}",
            f"Applicant Type : {cl_result['applicant_category']}",
            f"Total Documents: {len(checklist)} ({mandatory_count} mandatory, {optional_count} optional)",
            f"{'=' * 60}",
            "",
        ]
        for cat_name, items in categories.items():
            lines.append(cat_name.upper())
            lines.append("-" * len(cat_name))
            for item in items:
                flag = "[MANDATORY]" if item["mandatory"] else "[OPTIONAL] "
                lines.append(f"  {flag}  {item['document']}")
                if item.get("alternatives"):
                    lines.append(f"             Alternatives: {' | '.join(item['alternatives'])}")
                if item.get("notes"):
                    lines.append(f"             Note: {item['notes']}")
            lines.append("")
        plain_text = "\n".join(lines)
        st.download_button(
            label="⬇️ Download as .txt",
            data=plain_text,
            file_name=f"checklist_{cl_selected_product}_{cl_selected_category.lower().replace(' ','_')}.txt",
            mime="text/plain",
        )
        with st.expander("Preview text export"):
            st.code(plain_text, language=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Fee & Interest Calculator
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("🧮 Fee & Interest Calculator", width="large")
def show_calculator():
    st.caption(
        "Enter your product parameters. The calculator computes exact figures using "
        "standard banking formulas and fetches current fee/rate information from "
        "FinTrust's official policy documents."
    )

    calc_products = _fetch_calc_products()
    if not calc_products:
        st.error("Cannot load calculator products. Make sure the API server is running.")
        return

    calc_product_map = {p["label"]: p for p in calc_products}

    selected_calc_label   = st.selectbox("Select Product", list(calc_product_map.keys()), key="calc_product")
    selected_calc_product = calc_product_map[selected_calc_label]

    st.markdown(
        f"**Calculation types:** {', '.join(selected_calc_product['calc_types'])}",
        help="These are the values computed for this product.",
    )

    st.markdown("#### Parameters")
    input_cols = st.columns(2)
    param_values: dict = {}

    for idx, inp in enumerate(selected_calc_product["inputs"]):
        col = input_cols[idx % 2]
        with col:
            if inp["type"] == "number":
                param_values[inp["key"]] = st.number_input(
                    inp["label"],
                    value=float(inp["default"]),
                    min_value=0.0,
                    step=1000.0 if inp["default"] >= 1000 else 0.1,
                    format="%.2f" if inp["default"] < 100 else "%.0f",
                    key=f"calc_{inp['key']}",
                )
            elif inp["type"] == "select":
                param_values[inp["key"]] = st.selectbox(
                    inp["label"],
                    inp["options"],
                    index=inp["options"].index(inp["default"]) if inp["default"] in inp["options"] else 0,
                    key=f"calc_{inp['key']}",
                )

    calc_btn_col, _ = st.columns([2, 5])
    with calc_btn_col:
        calc_run = st.button("🧮 Calculate", type="primary", use_container_width=True, key="calc_run")

    if calc_run:
        with st.spinner(f"Computing **{selected_calc_label}** figures + retrieving policy rates..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/calculator",
                    json={"product": selected_calc_product["product_id"], "params": param_values},
                    timeout=300,
                )
                resp.raise_for_status()
                calc_result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("API server is offline.")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if calc_result.get("error"):
            st.error(calc_result["error"])
            return

        results = calc_result.get("results", {})
        if results:
            st.success(f"**{calc_result['product_label']}** — {calc_result['summary']}")
            st.markdown("#### 📊 Computed Values")
            metric_cols = st.columns(min(len(results), 3))
            for i, (k, v) in enumerate(results.items()):
                metric_cols[i % 3].metric(label=k, value=v)

        schedule = calc_result.get("schedule", [])
        if schedule:
            st.markdown("#### 📅 Period-wise Breakdown")
            df = pd.DataFrame(schedule)
            col_rename = {
                "month": "Month", "year": "Year", "emi": "EMI (₹)",
                "principal": "Principal (₹)", "interest": "Interest (₹)",
                "balance": "Balance (₹)", "value": "Value (₹)",
                "invested": "Invested (₹)",
            }
            df.rename(columns={c: col_rename.get(c, c) for c in df.columns}, inplace=True)
            st.dataframe(df, use_container_width=True, hide_index=True)

        policy_notes = calc_result.get("policy_notes", [])
        if policy_notes:
            with st.expander("📄 Policy Notes — Fees, Charges & Conditions", expanded=True):
                for note in policy_notes:
                    st.markdown(f"- {note}")

        if results:
            st.markdown("---")
            lines = [
                "FINTRUST BANK — FEE & INTEREST CALCULATION",
                f"Product  : {calc_result['product_label']}",
                f"{'=' * 50}",
                "",
                "COMPUTED VALUES",
                "---------------",
            ]
            for k, v in results.items():
                lines.append(f"  {k:<35} {v}")
            if schedule:
                lines += ["", "PERIOD-WISE BREAKDOWN", "---------------------"]
                headers = list(schedule[0].keys())
                lines.append("  " + "  ".join(str(h).ljust(16) for h in headers))
                for row in schedule:
                    lines.append("  " + "  ".join(str(row.get(h, "")).ljust(16) for h in headers))
            if policy_notes:
                lines += ["", "POLICY NOTES", "------------"]
                for note in policy_notes:
                    lines.append(f"  {note}")
            st.download_button(
                label="⬇️ Download as .txt",
                data="\n".join(lines),
                file_name=f"calculation_{selected_calc_product['product_id']}.txt",
                mime="text/plain",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Section: Policy Comparison Tool
# ═══════════════════════════════════════════════════════════════════════════════

@st.dialog("⚖️ Policy Comparison Tool", width="large")
def show_comparison():
    st.caption(
        "Select two products. The AI agent pulls key parameters from FinTrust's "
        "policy documents and builds a side-by-side comparison table with a "
        "recommendation on which product suits which customer."
    )

    cmp_products_list, cmp_quick_pairs = _fetch_compare_products()
    if not cmp_products_list:
        st.error("Cannot load products. Make sure the API server is running.")
        return

    cmp_product_map = {p["label"]: p["product_id"] for p in cmp_products_list}

    st.markdown("#### ⚡ Quick Compare")
    pair_cols = st.columns(min(len(cmp_quick_pairs), 4))
    for idx, pair in enumerate(cmp_quick_pairs):
        with pair_cols[idx % 4]:
            if st.button(pair["label"], key=f"qpair_{idx}", use_container_width=True):
                st.session_state["cmp_product_a"] = pair["a"]
                st.session_state["cmp_product_b"] = pair["b"]
                st.session_state["cmp_trigger"]   = True

    st.markdown("---")
    st.markdown("#### 🔧 Custom Comparison")
    col_a, col_vs, col_b = st.columns([5, 1, 5])

    def _label_from_id(pid: str) -> str:
        for p in cmp_products_list:
            if p["product_id"] == pid:
                return p["label"]
        return list(cmp_product_map.keys())[0]

    default_a_label = _label_from_id(st.session_state.get("cmp_product_a", "fixed_deposit"))
    default_b_label = _label_from_id(st.session_state.get("cmp_product_b", "recurring_deposit"))

    with col_a:
        cmp_label_a = st.selectbox(
            "Product A",
            list(cmp_product_map.keys()),
            index=list(cmp_product_map.keys()).index(default_a_label),
            key="cmp_select_a",
        )
    with col_vs:
        st.markdown("<div style='text-align:center;padding-top:32px;font-size:22px'>vs</div>", unsafe_allow_html=True)
    with col_b:
        cmp_label_b = st.selectbox(
            "Product B",
            list(cmp_product_map.keys()),
            index=list(cmp_product_map.keys()).index(default_b_label),
            key="cmp_select_b",
        )

    cmp_product_a = cmp_product_map[cmp_label_a]
    cmp_product_b = cmp_product_map[cmp_label_b]

    if cmp_product_a == cmp_product_b:
        st.warning("Please select two different products to compare.")
        return

    cmp_btn_col, _ = st.columns([3, 5])
    with cmp_btn_col:
        cmp_run = st.button(
            f"⚖️ Compare {cmp_label_a} vs {cmp_label_b}",
            type="primary",
            use_container_width=True,
            key="cmp_run",
        ) or st.session_state.pop("cmp_trigger", False)

    if cmp_run:
        with st.spinner(f"Comparing **{cmp_label_a}** vs **{cmp_label_b}** — retrieving policies..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/compare",
                    json={"product_a": cmp_product_a, "product_b": cmp_product_b},
                    timeout=300,
                )
                resp.raise_for_status()
                cmp_result = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("API server is offline.")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        if cmp_result.get("error"):
            st.error(cmp_result["error"])
            return

        label_a        = cmp_result["label_a"]
        label_b        = cmp_result["label_b"]
        parameters     = cmp_result.get("parameters", [])
        highlights     = cmp_result.get("highlights", [])
        recommendation = cmp_result.get("recommendation", "")

        if not parameters:
            st.warning("No comparison data returned.")
            return

        banner_col1, banner_col2, banner_col3 = st.columns([5, 1, 5])
        with banner_col1:
            st.markdown(
                f'<div style="background:#1565C0;color:white;padding:16px;'
                f'border-radius:8px;text-align:center;font-size:18px;font-weight:700">'
                f'🅐 {label_a}</div>',
                unsafe_allow_html=True,
            )
        with banner_col2:
            st.markdown(
                '<div style="text-align:center;padding-top:12px;font-size:20px;font-weight:700">vs</div>',
                unsafe_allow_html=True,
            )
        with banner_col3:
            st.markdown(
                f'<div style="background:#6a1b9a;color:white;padding:16px;'
                f'border-radius:8px;text-align:center;font-size:18px;font-weight:700">'
                f'🅑 {label_b}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("#### 📊 Parameter-by-Parameter Comparison")

        def _winner_badge(winner: str, la: str, lb: str) -> str:
            if winner == la:
                return f'<span style="background:#1565C0;color:white;font-size:11px;padding:2px 8px;border-radius:10px">✓ {la}</span>'
            elif winner == lb:
                return f'<span style="background:#6a1b9a;color:white;font-size:11px;padding:2px 8px;border-radius:10px">✓ {lb}</span>'
            elif winner == "Neutral":
                return '<span style="background:#607d8b;color:white;font-size:11px;padding:2px 8px;border-radius:10px">Neutral</span>'
            else:
                return '<span style="background:#e65100;color:white;font-size:11px;padding:2px 8px;border-radius:10px">Depends</span>'

        st.markdown(
            f'<div style="display:grid;grid-template-columns:2fr 2fr 2fr 1.5fr;gap:6px;'
            f'font-weight:700;background:#f0f4f8;padding:8px 12px;border-radius:6px;margin-bottom:4px">'
            f'<span>Parameter</span>'
            f'<span style="color:#1565C0">🅐 {label_a}</span>'
            f'<span style="color:#6a1b9a">🅑 {label_b}</span>'
            f'<span>Advantage</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        for i, param in enumerate(parameters):
            bg           = "#ffffff" if i % 2 == 0 else "#f8f9fa"
            winner_badge = _winner_badge(param["winner"], label_a, label_b)
            note_text    = f'<br><span style="font-size:11px;color:#777">{param["note"]}</span>' if param.get("note") else ""
            st.markdown(
                f'<div style="display:grid;grid-template-columns:2fr 2fr 2fr 1.5fr;gap:6px;'
                f'background:{bg};padding:10px 12px;border-radius:4px;margin:2px 0;align-items:center">'
                f'<span style="font-weight:600">{param["parameter"]}</span>'
                f'<span>{param["value_a"]}</span>'
                f'<span>{param["value_b"]}</span>'
                f'<span>{winner_badge}{note_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        wins_a = sum(1 for p in parameters if p["winner"] == label_a)
        wins_b = sum(1 for p in parameters if p["winner"] == label_b)
        st.markdown(
            f'<div style="background:#f0f4f8;padding:10px 12px;border-radius:6px;margin-top:8px">'
            f'<b>Score:</b> &nbsp;'
            f'<span style="color:#1565C0">🅐 {label_a}: {wins_a}</span> &nbsp;|&nbsp; '
            f'<span style="color:#6a1b9a">🅑 {label_b}: {wins_b}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if highlights:
            st.markdown("---")
            st.markdown("#### 💡 Key Differences")
            for h in highlights:
                st.markdown(f"- {h}")

        if recommendation:
            st.markdown("---")
            st.info(f"**🎯 Recommendation**\n\n{recommendation}")

        st.markdown("---")
        export_lines = [
            "FINTRUST BANK — POLICY COMPARISON",
            f"Product A : {label_a}",
            f"Product B : {label_b}",
            f"{'=' * 70}",
            "",
            f"{'Parameter':<30}  {'':1}  {label_a:<25}  {label_b:<25}  {'Advantage':<15}  Note",
            f"{'-' * 130}",
        ]
        for p in parameters:
            export_lines.append(
                f"{p['parameter']:<30}  {'':1}  {p['value_a']:<25}  {p['value_b']:<25}  {p['winner']:<15}  {p['note']}"
            )
        export_lines += [
            "",
            f"Score: {label_a}: {wins_a}  |  {label_b}: {wins_b}",
            "",
            "KEY DIFFERENCES",
            "---------------",
        ]
        export_lines += [f"  - {h}" for h in highlights]
        if recommendation:
            export_lines += ["", "RECOMMENDATION", "--------------", f"  {recommendation}"]

        st.download_button(
            label="⬇️ Download Comparison as .txt",
            data="\n".join(export_lines),
            file_name=f"comparison_{cmp_product_a}_vs_{cmp_product_b}.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main area — Chat (default view)
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🧭 FinTrust Compass")
st.caption("AI-powered policy assistant for FinTrust bank employees")


# Scrollable message history
chat_container = st.container(height=520, border=False)
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                domain     = msg.get("domain", "")
                color      = DOMAIN_COLORS.get(domain, "#607d8b")
                icon       = DOMAIN_ICONS.get(domain, "🤖")
                confidence = msg.get("confidence", "")
                conf_class = f"confidence-{confidence}"
                st.markdown(
                    f'<span class="domain-badge" style="background:{color}">'
                    f'{icon} {domain.replace("_", " ").title()}'
                    f'</span> '
                    f'<span class="{conf_class}">({confidence} confidence)</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} source chunks", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>{src["source_file"]}</b> · Page {src["page"]}<br>'
                            f'<i>{src["snippet"][:150]}...</i>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

# ---------------------------------------------------------------------------
# Chat input + FinTrust tool menu
# ---------------------------------------------------------------------------

_sidebar_query = st.session_state.pending_query
if _sidebar_query:
    st.session_state.pending_query = None

composer_left, _ = st.columns([1, 12], vertical_alignment="bottom")
with composer_left:
    if st.button("➕", key="toggle_tool_sheet", help="Open FinTrust tools", use_container_width=True):
        st.session_state.show_tool_sheet = not st.session_state.show_tool_sheet

if st.session_state.show_tool_sheet:
    with st.container(border=True):
        st.caption("FinTrust functionalities")
        tool_col1, tool_col2 = st.columns(2)
        with tool_col1:
            if st.button("✅ Eligibility Checker", key="sheet_elig", use_container_width=True):
                st.session_state.dialog_to_open = "eligibility"
                st.session_state.show_tool_sheet = False
                st.rerun()
            if st.button("📋 Checklist Generator", key="sheet_checklist", use_container_width=True):
                st.session_state.dialog_to_open = "checklist"
                st.session_state.show_tool_sheet = False
                st.rerun()
        with tool_col2:
            if st.button("🧮 Fee & Interest Calculator", key="sheet_calc", use_container_width=True):
                st.session_state.dialog_to_open = "calculator"
                st.session_state.show_tool_sheet = False
                st.rerun()
            if st.button("⚖️ Policy Comparison", key="sheet_compare", use_container_width=True):
                st.session_state.dialog_to_open = "comparison"
                st.session_state.show_tool_sheet = False
                st.rerun()

tool_to_open = st.session_state.get("dialog_to_open")
if tool_to_open:
    # Consume the click-event before opening to avoid accidental reopen loops.
    st.session_state.dialog_to_open = None
    if tool_to_open == "eligibility":
        show_eligibility()
    elif tool_to_open == "checklist":
        show_checklist()
    elif tool_to_open == "calculator":
        show_calculator()
    elif tool_to_open == "comparison":
        show_comparison()

prompt = st.chat_input("Ask about any FinTrust policy...")
active_prompt = prompt or _sidebar_query

# ---------------------------------------------------------------------------
# Handle chat message
# ---------------------------------------------------------------------------

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(active_prompt)

    with chat_container:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            domain      = ""
            confidence  = ""
            sources     = []
            full_answer = ""

            status_placeholder.markdown("_Classifying query..._")

            try:
                with requests.post(
                    f"{API_BASE}/query/stream",
                    json={
                        "question": active_prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                    stream=True,
                    timeout=300,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        if isinstance(line, bytes):
                            line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        data = json.loads(line[6:])

                        if data["type"] == "domain":
                            domain     = data["domain"]
                            confidence = data["confidence"]
                            color      = DOMAIN_COLORS.get(domain, "#607d8b")
                            icon       = DOMAIN_ICONS.get(domain, "🤖")
                            conf_class = f"confidence-{confidence}"
                            status_placeholder.markdown(
                                f'<span class="domain-badge" style="background:{color}">'
                                f'{icon} {domain.replace("_", " ").title()}'
                                f'</span> '
                                f'<span class="{conf_class}">({confidence} confidence)</span>',
                                unsafe_allow_html=True,
                            )

                        elif data["type"] == "token":
                            full_answer += data["content"]
                            answer_placeholder.markdown(full_answer + "▌")

                        elif data["type"] == "done":
                            sources = data["sources"]
                            answer_placeholder.markdown(full_answer)

            except requests.exceptions.ConnectionError:
                status_placeholder.empty()
                answer_placeholder.error(
                    "Cannot connect to API server. Start it with:\n"
                    "```\nuvicorn api.main:app --reload\n```"
                )
                full_answer = "_(API unavailable)_"
            except Exception as e:
                status_placeholder.empty()
                answer_placeholder.error(f"Error: {e}")
                full_answer = f"_(Error: {e})_"

            if sources:
                with st.expander(f"📄 {len(sources)} source chunks", expanded=False):
                    for src in sources:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>{src["source_file"]}</b> · Page {src["page"]}<br>'
                            f'<i>{src["snippet"][:150]}...</i>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    full_answer,
        "domain":     domain,
        "confidence": confidence,
        "sources":    sources,
    })

# Tool dialogs are opened via the click-event dispatcher above.


