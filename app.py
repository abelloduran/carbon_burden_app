import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(page_title="Global Carbon Burden", layout="wide")

# =============================================================================
# Load data
# =============================================================================

# aggregate level 

df = pd.read_csv("cb_dataset_final.csv")

# firm level

firm_parts = [
    pd.read_parquet("cb_mkt_firm_app_part1.parquet"),
    pd.read_parquet("cb_mkt_firm_app_part2.parquet"),
    pd.read_parquet("cb_mkt_firm_app_part3.parquet")
]

cb_mkt_firm_app = pd.concat(firm_parts, ignore_index=True)

# =============================================================================
# Standardize names
# =============================================================================

if "market_value" in df.columns and "mkt_value" not in df.columns:
    df = df.rename(columns={"market_value": "mkt_value"})

# =============================================================================
# Labels
# =============================================================================

SCENARIO_LABELS = {
    "B2C": "Below 2°C",
    "CP": "Current Policies",
    "NDC": "Nationally Determined Contributions"
}

SCENARIO_EXPLANATIONS = {
    "B2C": "Scenario aligned with climate policies consistent with limiting warming below 2°C. Requires strong and relatively early policy action.",
    "CP": "Baseline scenario reflecting currently implemented policies only. No additional climate action beyond what is already in place.",
    "NDC": "Reflects countries' pledged climate targets under the Paris Agreement. Moderate transition effort, not sufficient for 2°C."
}

SCENARIO_ORDER = ["CP", "NDC", "B2C"]

FIRM_PATHWAY_LABELS = {
    "Target": "ISS Target",
    "Benchmark": "ISS Benchmark",
    "Historical": "ISS Historical",
    "NGFSRMCP": "NGFS Current Policies",
    "NGFSRMNDC": "NGFS Nationally Determined Contributions",
    "NGFSRMB2C": "NGFS Below 2°C",
    "NGFSRMNZ": "NGFS Net Zero 2050"
}

FIRM_SCOPE_LABELS = {
    "s1": "Scope 1",
    "s12": "Scope 1+2",
    "s123": "Scope 1+2+3"
}

FIRM_SCOPE_ORDER = ["s1", "s12", "s123"]
FIRM_DISCOUNT_ORDER = [0.015, 0.020, 0.025]

MODEL_EXPLANATION = """
The NGFS scenarios are produced by the Network for Greening the Financial System, 
a group of central banks and supervisors working on climate- and environment-related 
risk management in the financial sector.

The scenarios are generated using integrated assessment models such as GCAM, 
MESSAGE-GLOBIOM and REMIND-MAgPIE, together with the NiGEM macroeconomic model.

[Open NGFS model documentation](https://zenodo.org/records/17901363)
"""

ISS_EXPLANATION = """
**ISS projected emissions**

**Historical:** assumes that the historical rate of emissions evolution will continue for the issuer. This is done separately for each emissions scope.

**Benchmark:** assumes that issuers follow the industry's emissions evolution under a neutral scenario based on current government commitments, excluding additional ambitions not yet reflected in actual policies.

**Target:** assumes that issuers reach their disclosed emissions targets.
"""

VARIABLE_LABELS = {
    "carbon_burden": "Carbon Burden",
    "carbon_burden_net": "Net Carbon Burden",
    "tax_burden_reduction": "Tax Burden Reduction",
    "mkt_value": "Market Value",
    "cb_mkt_value": "Carbon Burden / Market Value",
    "cb_net_mkt_value": "Net Carbon Burden / Market Value"
}

VARIABLES = list(VARIABLE_LABELS.keys())

# =============================================================================
# Territorial classification
# =============================================================================

EU24_MEMBERS = [
    "AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN",
    "FRA", "DEU", "GRC", "HUN", "ITA", "LVA", "LTU", "MLT", "POL",
    "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"
]

EU_EXCEPTIONS = ["LUX", "NLD", "IRL"]

STANDALONE_COUNTRIES = [
    "USA", "CHN", "IND", "RUS", "JPN", "CAN", "GBR", "AUS", "KOR",
    "CHE", "NOR", "TUR", "SAU", "BRA", "MEX", "IDN", "ZAF",
    "ARG", "ARE", "SGP", "TWN", "HKG"
] + EU_EXCEPTIONS

LATIN_AMERICA = [
    "BOL", "CHL", "COL", "CRI", "CUB", "DOM", "ECU", "GTM", "HND",
    "HTI", "JAM", "NIC", "PAN", "PER", "PRY", "SLV", "SUR", "TTO",
    "URY", "VEN"
]

AFRICA = [
    "AGO", "BEN", "BWA", "CIV", "CMR", "COD", "COG", "DZA", "EGY",
    "ERI", "ETH", "GAB", "GHA", "KEN", "LBY", "MAR", "MLI", "MOZ",
    "MUS", "NAM", "NER", "NGA", "SDN", "SEN", "SSD", "TGO", "TUN",
    "TZA", "UGA", "ZMB", "ZWE", "MWI", "GMB"
]

MIDDLE_EAST = [
    "BHR", "IRN", "IRQ", "ISR", "JOR", "KWT", "LBN", "OMN", "QAT",
    "SYR", "YEM", "PSE"
]

REST_OF_ASIA = [
    "ARM", "AZE", "BGD", "BRN", "GEO", "KAZ", "KGZ", "KHM", "LKA",
    "MMR", "MNG", "MYS", "NPL", "PAK", "PHL", "PRK", "THA", "TJK",
    "TKM", "UZB", "VNM", "PNG"
]

REST_OF_EUROPE = [
    "ALB", "BIH", "BLR", "GIB", "ISL", "MDA", "MKD", "MNE", "SRB",
    "UKR", "LIE", "MCO", "JEY", "GGY", "IMN"
]

EXCLUDE_REGIONS = [
    "World", "EU27", "OECD", "Downscaling|Countries without IEA statistics"
]

def assign_display_unit(region, country):
    if region in EXCLUDE_REGIONS:
        return None
    if region == "EU24":
        return "EU24"
    if region in EU24_MEMBERS:
        return None
    if region in STANDALONE_COUNTRIES:
        return country
    if region in LATIN_AMERICA:
        return "Latin America"
    if region in AFRICA:
        return "Africa"
    if region in MIDDLE_EAST:
        return "Middle East"
    if region in REST_OF_ASIA:
        return "Rest of Asia"
    if region in REST_OF_EUROPE:
        return "Rest of Europe"
    return "Rest of World"

def assign_map_display_unit(region, country):
    if region in EXCLUDE_REGIONS or region == "EU24":
        return None
    if region in EU24_MEMBERS:
        return "EU24"
    return assign_display_unit(region, country)

def assign_firm_display_unit(country_iso3, country_name):
    if country_iso3 in EU24_MEMBERS:
        return "EU24"
    if country_iso3 in STANDALONE_COUNTRIES:
        return country_name
    if country_iso3 in LATIN_AMERICA:
        return "Latin America"
    if country_iso3 in AFRICA:
        return "Africa"
    if country_iso3 in MIDDLE_EAST:
        return "Middle East"
    if country_iso3 in REST_OF_ASIA:
        return "Rest of Asia"
    if country_iso3 in REST_OF_EUROPE:
        return "Rest of Europe"
    return "Rest of World"

def assign_view_type(display_unit):
    if display_unit == "EU24":
        return "eu_submap"
    if display_unit in [
        "Latin America", "Africa", "Middle East",
        "Rest of Asia", "Rest of Europe", "Rest of World"
    ]:
        return "regional_pies"
    return "country"

# =============================================================================
# Aggregate data construction
# =============================================================================

df["display_unit"] = df.apply(lambda x: assign_display_unit(x["region"], x["country"]), axis=1)
df["view_type"] = df["display_unit"].apply(assign_view_type)

df_clean = df[
    df["display_unit"].notna() &
    df["scenario"].isin(SCENARIO_ORDER)
].copy()

SUM_COLS = [
    "carbon_burden",
    "carbon_burden_net",
    "tax_burden_reduction",
    "mkt_value"
]

GROUP_COLS = [
    "display_unit",
    "view_type",
    "scenario",
    "model",
    "horizon_label",
    "discount_rate"
]

global_view_df = (
    df_clean
    .groupby(GROUP_COLS, as_index=False)[SUM_COLS]
    .sum()
)

global_view_df["cb_mkt_value"] = global_view_df["carbon_burden"] / global_view_df["mkt_value"]
global_view_df["cb_net_mkt_value"] = global_view_df["carbon_burden_net"] / global_view_df["mkt_value"]

map_base = df[df["scenario"].isin(SCENARIO_ORDER)].copy()

map_base["display_unit"] = map_base.apply(
    lambda x: assign_map_display_unit(x["region"], x["country"]),
    axis=1
)

map_base["view_type"] = map_base["display_unit"].apply(assign_view_type)

map_base = (
    map_base[
        map_base["display_unit"].notna()
    ][["region", "country", "display_unit", "view_type"]]
    .drop_duplicates()
)

global_map_df = map_base.merge(
    global_view_df,
    on=["display_unit", "view_type"],
    how="left"
)

eu_detail_df = df[
    df["region"].isin(EU24_MEMBERS) &
    df["scenario"].isin(SCENARIO_ORDER)
].copy()

eu_detail_df["display_unit"] = eu_detail_df["country"]
eu_detail_df["view_type"] = "country"

regional_detail_df = df[
    df["scenario"].isin(SCENARIO_ORDER)
].copy()

regional_detail_df["display_unit"] = regional_detail_df.apply(
    lambda x: assign_map_display_unit(x["region"], x["country"]),
    axis=1
)

regional_detail_df["view_type"] = regional_detail_df["display_unit"].apply(assign_view_type)

regional_detail_df = regional_detail_df[
    regional_detail_df["view_type"].eq("regional_pies")
].copy()

# =============================================================================
# Firm-level data construction
# =============================================================================

cb_mkt_firm_app["display_unit"] = cb_mkt_firm_app.apply(
    lambda x: assign_firm_display_unit(x["country_iso3"], x["country_name"]),
    axis=1
)

cb_mkt_firm_app["pathway"] = cb_mkt_firm_app["pathway"].astype(str)
cb_mkt_firm_app["scope"] = cb_mkt_firm_app["scope"].astype(str)
cb_mkt_firm_app["horizon_label"] = cb_mkt_firm_app["horizon_label"].astype(str)
cb_mkt_firm_app["discount_rate"] = cb_mkt_firm_app["discount_rate"].astype(float)

FIRM_ALLOWED_PATHWAYS = [
    "Target", "Benchmark", "Historical",
    "NGFSRMCP", "NGFSRMNDC", "NGFSRMB2C", "NGFSRMNZ"
]

firm_df = cb_mkt_firm_app[
    cb_mkt_firm_app["pathway"].isin(FIRM_ALLOWED_PATHWAYS)
].copy()

# =============================================================================
# Helper functions
# =============================================================================

def clean_horizon_label(x):
    x = str(x)
    if x == "all_future_years":
        return "All Future Years"
    if x.startswith("to_"):
        return "To " + x.replace("to_", "")
    return x.replace("_", " ").title()

def clean_discount_label(x):
    try:
        return f"{float(x) * 100:.1f}%"
    except Exception:
        return str(x)

def clean_model_label(x):
    return str(x).replace("_", " ").title()

def is_percentage_variable(variable):
    return variable in ["cb_mkt_value", "cb_net_mkt_value"]

def format_percentage(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:,.2f}%"

def format_trillion(x):
    if pd.isna(x):
        return "N/A"
    return f"${x / 1e12:,.2f}T"

def default_index(options, target):
    for i, x in enumerate(options):
        try:
            if float(x) == float(target):
                return i
        except Exception:
            if str(x) == str(target):
                return i
    return 0

def get_home_ratio_values(data, scenario):
    temp = data.copy()
    temp = temp[
        (temp["region"] == "World") &
        (temp["scenario"] == scenario) &
        (temp["discount_rate"].astype(float).round(4) == 0.02)
    ].copy()

    if "all_future_years" in temp["horizon_label"].astype(str).unique():
        temp = temp[temp["horizon_label"].astype(str) == "all_future_years"].copy()

    temp = temp.sort_values("model")
    return temp["cb_mkt_value"].tolist()

def get_firm_home_ratio(data, pathway, scope="s123", horizon="all_future_years", discount_rate=0.02):
    temp = data[
        (data["pathway"] == pathway) &
        (data["scope"] == scope) &
        (data["horizon_label"] == horizon) &
        (data["discount_rate"].round(4) == round(discount_rate, 4))
    ].copy()

    temp = temp.drop_duplicates(subset=["ISIN"])

    if temp.empty or temp["market_cap"].sum() == 0:
        return np.nan

    return temp["carbon_burden"].sum() / temp["market_cap"].sum()

def make_pie_data(data, value_col, label_col="display_unit", top_n=10):
    temp = data[[label_col, value_col]].dropna().copy()
    temp = temp.groupby(label_col, as_index=False)[value_col].sum()
    temp = temp.sort_values(value_col, ascending=False)

    top = temp.head(top_n).copy()
    rest = temp.iloc[top_n:].copy()

    if not rest.empty:
        rest_row = pd.DataFrame({
            label_col: ["Rest"],
            value_col: [rest[value_col].sum()]
        })
        temp = pd.concat([top, rest_row], ignore_index=True)
    else:
        temp = top

    total = temp[value_col].sum()
    temp["share"] = temp[value_col] / total if total != 0 else 0

    return temp

def get_composition_baseline(data, scenario="CP"):
    temp = data[
        (data["scenario"] == scenario) &
        (data["discount_rate"].astype(float).round(4) == 0.02)
    ].copy()

    if "all_future_years" in temp["horizon_label"].astype(str).unique():
        temp = temp[temp["horizon_label"].astype(str) == "all_future_years"].copy()

    temp = (
        temp
        .groupby("display_unit", as_index=False)[SUM_COLS]
        .mean()
    )

    return temp

def prepare_plot_values(plot_df, selected_variable):
    plot_df = plot_df.copy()

    if is_percentage_variable(selected_variable):
        plot_df[selected_variable] = plot_df[selected_variable] * 100
        colorbar_title = "%"
        hover_suffix = "%"
    elif selected_variable in ["carbon_burden", "carbon_burden_net", "tax_burden_reduction", "mkt_value"]:
        plot_df[selected_variable] = plot_df[selected_variable] / 1e12
        colorbar_title = "Trillion 2023 USD"
        hover_suffix = " trillion 2023 USD"
    else:
        colorbar_title = VARIABLE_LABELS.get(selected_variable, selected_variable)
        hover_suffix = ""

    return plot_df, colorbar_title, hover_suffix

def smooth_density(values, bins=80):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().astype(float)

    if len(values) < 5:
        return None, None

    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2

    kernel = np.array([1, 2, 4, 6, 4, 2, 1], dtype=float)
    kernel = kernel / kernel.sum()

    smooth = np.convolve(counts, kernel, mode="same")

    return centers, smooth

def filter_firms_for_territory(data, territory):
    if territory is None or territory == "World":
        return data.copy()
    return data[data["display_unit"] == territory].copy()

def get_unique_firm_rows(data):
    return data.drop_duplicates(subset=["ISIN"]).copy()

# =============================================================================
# Session state
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "home_stage" not in st.session_state:
    st.session_state.home_stage = "Intro"

if "selected_display_unit" not in st.session_state:
    st.session_state.selected_display_unit = "World"

if "results_stage" not in st.session_state:
    st.session_state.results_stage = "Global"

# =============================================================================
# CSS
# =============================================================================

st.markdown(
    """
    <style>
    html, body, .stMarkdown, .stText, label, p {
        font-family: Georgia, 'Times New Roman', serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: Georgia, 'Times New Roman', serif !important;
    }

    div[data-testid="stMarkdownContainer"] {
        font-family: Georgia, 'Times New Roman', serif !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 68px;
        font-weight: 700;
        margin-top: 90px;
        margin-bottom: 24px;
        color: #252634;
        letter-spacing: -1px;
    }

    .page-title {
        font-size: 44px;
        font-weight: 500;
        color: #7b2431;
        margin-bottom: 30px;
    }

    .sub-page-title {
        font-size: 32px;
        font-weight: 600;
        color: #252634;
        margin-top: 35px;
        margin-bottom: 16px;
    }

    .section-note {
        font-size: 17px;
        color: #444444;
        margin-bottom: 25px;
        line-height: 1.45;
    }

    .home-subtitle-clean {
        max-width: 950px;
        margin: auto;
        text-align: center;
        font-size: 24px;
        line-height: 1.55;
        color: #4d4d55;
        margin-bottom: 15px;
    }

    .scenario-wrapper {
        max-width: 1180px;
        margin: auto;
        margin-top: 6px;
        margin-bottom: 30px;
    }

    .scenario-card {
        border: 1px solid #ddd9db;
        border-radius: 22px;
        padding: 34px 20px 32px 20px;
        background-color: #fcfcfc;
        box-shadow: 0 12px 30px rgba(20,20,30,0.04);
        text-align: center;
        height: 100%;
    }

    .scenario-title {
        font-size: 17px;
        letter-spacing: 1.6px;
        font-weight: 700;
        color: #7b2431;
        margin-bottom: 22px;
        text-transform: uppercase;
    }

    .scenario-range {
        font-size: 36px;
        font-weight: 700;
        color: #252634;
        line-height: 1.25;
        margin-bottom: 14px;
    }

    .scenario-note {
        font-size: 17px;
        color: #666666;
        font-style: italic;
    }

    .home-menu-subtitle {
        max-width: 950px;
        margin: auto;
        text-align: center;
        font-size: 24px;
        line-height: 1.55;
        color: #4d4d55;
        margin-top: 12px;
        margin-bottom: 62px;
    }

    .nav-wrapper {
        padding: 12px 18px;
        border: 1px solid #e3e0e0;
        border-radius: 20px;
        background: #fbfbfc;
        box-shadow: 0 6px 18px rgba(30, 30, 40, 0.04);
        margin-bottom: 34px;
    }

    div.stButton > button {
        width: 100%;
        height: 54px;
        border-radius: 14px;
        border: 1px solid #d9d6d7;
        background-color: #fafafa;
        font-size: 18px;
        font-weight: 600;
        color: #2b2d3a;
        font-family: Georgia, 'Times New Roman', serif !important;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        border: 1px solid #7b2431;
        background-color: #f4eef0;
        color: #7b2431;
        transform: translateY(-1px);
    }

    .continue-button button {
        width: 220px !important;
        height: 64px !important;
        font-size: 22px !important;
        border-radius: 999px !important;
        background: #7b2431 !important;
        color: white !important;
        border: 1px solid #7b2431 !important;
    }

    .continue-button button:hover {
        background: #5f1b26 !important;
        color: white !important;
    }

    .home-menu-card button {
        height: 118px !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        border-radius: 24px !important;
        background: #fcfcfc !important;
        border: 1px solid #ddd9db !important;
        color: #252634 !important;
        box-shadow: 0 12px 30px rgba(20,20,30,0.045) !important;
        transition: all 0.2s ease !important;
    }

    .home-menu-card button:hover {
        background: #f4eef0 !important;
        border-color: #7b2431 !important;
        color: #7b2431 !important;
        transform: translateY(-2px);
    }

    .header-text {
        font-size: 26px;
        font-weight: 700;
        color: #111111;
    }

    .concept-text {
        font-size: 24px;
        color: #111111;
        padding-top: 20px;
    }

    .meaning-text {
        font-size: 21px;
        line-height: 1.35;
        color: #111111;
        padding-top: 20px;
    }

    .paper-link {
        color: #7b2431;
        font-weight: 600;
        text-decoration: none;
    }

    .rule {
        border-top: 2px solid #111111;
        margin-top: 12px;
        margin-bottom: 20px;
    }

    .light-rule {
        border-top: 1px solid #dddddd;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .note {
        font-size: 18px;
        color: #444444;
        margin-top: 35px;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Navigation
# =============================================================================

def navigation_bar():
    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "Home"
            st.session_state.home_stage = "Intro"
            st.session_state.results_stage = "Global"
            st.session_state.selected_display_unit = "World"
            st.rerun()

    with col2:
        if st.button("Methodology", key="nav_methodology"):
            st.session_state.page = "Methodology"
            st.rerun()

    with col3:
        if st.button("Results", key="nav_results"):
            st.session_state.page = "Results"
            st.session_state.results_stage = "Global"
            st.session_state.selected_display_unit = "World"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Filters
# =============================================================================

def show_aggregate_filters(data, include_variable=True, key_prefix="agg"):
    st.sidebar.markdown("## Aggregate country-level controls")

    model_options = sorted(data["model"].dropna().unique())
    model_reverse = {clean_model_label(x): x for x in model_options}

    selected_model_label = st.sidebar.selectbox(
        "Model",
        list(model_reverse.keys()),
        key=f"{key_prefix}_model"
    )
    selected_model = model_reverse[selected_model_label]

    with st.sidebar.expander("About NGFS models"):
        st.markdown(MODEL_EXPLANATION)

    scenario_options = [s for s in SCENARIO_ORDER if s in data["scenario"].dropna().unique()]
    scenario_display = [f"{x} — {SCENARIO_LABELS.get(x, x)}" for x in scenario_options]

    selected_scenario_display = st.sidebar.selectbox(
        "Scenario",
        scenario_display,
        key=f"{key_prefix}_scenario"
    )
    selected_scenario = selected_scenario_display.split(" — ")[0]

    with st.sidebar.expander("Scenario explanation"):
        st.markdown(
            f"""
            **{selected_scenario} — {SCENARIO_LABELS.get(selected_scenario, selected_scenario)}**

            {SCENARIO_EXPLANATIONS.get(selected_scenario, "No explanation available.")}
            """
        )

    horizon_options = sorted(data["horizon_label"].dropna().unique())
    horizon_reverse = {clean_horizon_label(x): x for x in horizon_options}

    default_horizon_index = 0
    if "All Future Years" in horizon_reverse.keys():
        default_horizon_index = list(horizon_reverse.keys()).index("All Future Years")

    selected_horizon_label = st.sidebar.selectbox(
        "Horizon",
        list(horizon_reverse.keys()),
        index=default_horizon_index,
        key=f"{key_prefix}_horizon"
    )
    selected_horizon = horizon_reverse[selected_horizon_label]

    discount_options = sorted(data["discount_rate"].dropna().unique())
    discount_reverse = {clean_discount_label(x): x for x in discount_options}

    selected_discount_label = st.sidebar.selectbox(
        "Discount Rate",
        list(discount_reverse.keys()),
        index=default_index(list(discount_reverse.values()), 0.02),
        key=f"{key_prefix}_discount"
    )
    selected_discount = discount_reverse[selected_discount_label]

    selected_variable = None
    selected_variable_label = None

    if include_variable:
        variable_reverse = {VARIABLE_LABELS[v]: v for v in VARIABLES if v in data.columns}

        selected_variable_label = st.sidebar.selectbox(
            "Variable",
            list(variable_reverse.keys()),
            index=list(variable_reverse.keys()).index("Carbon Burden / Market Value")
            if "Carbon Burden / Market Value" in variable_reverse.keys()
            else 0,
            key=f"{key_prefix}_variable"
        )
        selected_variable = variable_reverse[selected_variable_label]

    filtered = data[
        (data["model"] == selected_model) &
        (data["scenario"] == selected_scenario) &
        (data["horizon_label"] == selected_horizon) &
        (data["discount_rate"] == selected_discount)
    ].copy()

    return (
        filtered,
        selected_model_label,
        selected_scenario,
        selected_horizon_label,
        selected_discount_label,
        selected_variable,
        selected_variable_label
    )

def show_firm_filters(key_prefix="firm"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Firm-level controls")

    family = st.sidebar.selectbox(
        "Projection family",
        ["ISS projections", "NGFS pathways"],
        index=0,
        key=f"{key_prefix}_family"
    )

    if family == "ISS projections":
        pathway_options = ["Target", "Benchmark", "Historical"]
    else:
        pathway_options = ["NGFSRMCP", "NGFSRMNDC", "NGFSRMB2C", "NGFSRMNZ"]

    pathway_labels = [FIRM_PATHWAY_LABELS[p] for p in pathway_options]

    selected_pathway_label = st.sidebar.selectbox(
        "Scenario / projection",
        pathway_labels,
        index=0,
        key=f"{key_prefix}_pathway"
    )

    selected_pathway = {
        FIRM_PATHWAY_LABELS[p]: p for p in pathway_options
    }[selected_pathway_label]

    horizon_options = sorted(firm_df["horizon_label"].dropna().unique())
    horizon_reverse = {clean_horizon_label(x): x for x in horizon_options}

    default_horizon_index = 0
    if "All Future Years" in horizon_reverse.keys():
        default_horizon_index = list(horizon_reverse.keys()).index("All Future Years")

    selected_horizon_label = st.sidebar.selectbox(
        "Firm-level horizon",
        list(horizon_reverse.keys()),
        index=default_horizon_index,
        key=f"{key_prefix}_horizon"
    )

    selected_horizon = horizon_reverse[selected_horizon_label]

    with st.sidebar.expander("About firm-level projections"):
        st.markdown(ISS_EXPLANATION)
        st.markdown(
            """
            **NGFS firm-level pathways**

            The NGFS firm-level pathways use projected emissions paths aligned with selected NGFS scenarios. 
            Current Policies, NDC, Below 2°C and Net Zero 2050 are kept as the main comparable pathways.
            """
        )

    return selected_pathway, selected_pathway_label, selected_horizon, selected_horizon_label

# =============================================================================
# Firm-level charts and tables
# =============================================================================

def render_firm_density_panel(
    data,
    territory,
    selected_pathway,
    selected_pathway_label,
    selected_horizon,
    selected_horizon_label
):
    territory_label = territory if territory else "World"

    base = filter_firms_for_territory(data, territory_label)
    base = base[
        (base["pathway"] == selected_pathway) &
        (base["horizon_label"] == selected_horizon)
    ].copy()

    st.markdown(
        f"""
        <div class="sub-page-title">
        Firm-Level Results using Company Data: {territory_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-note">
        Densities show the distribution of log(Carbon Burden / Market Cap) across firms for 
        <b>{selected_pathway_label}</b>, through <b>{selected_horizon_label}</b>. 
        Each chart compares the three discount rates. The vertical line marks CB / Market Cap = 1.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    for col, scope in zip([col1, col2, col3], FIRM_SCOPE_ORDER):
        with col:
            fig = go.Figure()

            scope_base = base[base["scope"] == scope].copy()

            for dr in FIRM_DISCOUNT_ORDER:
                temp = scope_base[scope_base["discount_rate"].round(4) == round(dr, 4)].copy()
                temp = get_unique_firm_rows(temp)

                x, y = smooth_density(temp["log_cb_market_cap"], bins=90)

                if x is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            name=clean_discount_label(dr),
                            line=dict(width=2)
                        )
                    )

            baseline = scope_base[
                scope_base["discount_rate"].round(4) == 0.020
            ].copy()

            baseline = get_unique_firm_rows(baseline)
            n_firms = baseline["ISIN"].nunique()
            share_above = (
                (baseline["cb_market_cap"] > 1).mean()
                if n_firms > 0 else np.nan
            )

            fig.add_vline(
                x=0,
                line_dash="dot",
                line_color="black",
                opacity=0.8
            )

            fig.add_annotation(
                x=0.98,
                y=0.96,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="right",
                text=f"N = {n_firms:,}<br>{share_above * 100:,.1f}% with CB/MktCap > 1"
                if pd.notna(share_above) else f"N = {n_firms:,}",
                font=dict(size=13)
            )

            fig.update_layout(
                title=f"{FIRM_SCOPE_LABELS[scope]}",
                xaxis_title="ln(Carbon Burden / Market Cap)",
                yaxis_title="Density",
                height=420,
                font=dict(family="Georgia, Times New Roman, serif"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.35,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=60, b=80)
            )

            st.plotly_chart(fig, use_container_width=True)

    table_base = base[
        (base["scope"] == "s123") &
        (base["discount_rate"].round(4) == 0.020)
    ].copy()

    table_base = get_unique_firm_rows(table_base)

    if table_base.empty:
        st.warning("No firm-level observations available for this selection.")
        return

    total_cb = table_base["carbon_burden"].sum()
    total_mkt = table_base["market_cap"].sum()
    aggregate_ratio = total_cb / total_mkt if total_mkt != 0 else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Number of firms", f"{table_base['ISIN'].nunique():,}")
    m2.metric("Total Carbon Burden", format_trillion(total_cb))
    m3.metric("Total Market Cap", format_trillion(total_mkt))
    m4.metric("CB / Market Cap", format_percentage(aggregate_ratio))

    if territory_label in [
        "EU24", "Latin America", "Africa", "Middle East",
        "Rest of Asia", "Rest of Europe", "Rest of World", "World"
    ]:
        st.markdown("### Firm-level sample composition")

        country_firms = (
            table_base
            .groupby("country_name", as_index=False)
            .agg(
                n_firms=("ISIN", "nunique"),
                carbon_burden=("carbon_burden", "sum")
            )
        )

        pie_firms = make_pie_data(country_firms, "n_firms", label_col="country_name", top_n=10)
        pie_cb = make_pie_data(country_firms, "carbon_burden", label_col="country_name", top_n=10)

        c1, c2 = st.columns(2)

        with c1:
            fig_firms = px.pie(
                pie_firms,
                names="country_name",
                values="n_firms",
                title="Share of firms by country",
                hole=0.35
            )
            fig_firms.update_traces(textposition="inside", textinfo="percent+label")
            fig_firms.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=520)
            st.plotly_chart(fig_firms, use_container_width=True)

        with c2:
            fig_cb = px.pie(
                pie_cb,
                names="country_name",
                values="carbon_burden",
                title="Share of firm-level carbon burden by country",
                hole=0.35
            )
            fig_cb.update_traces(textposition="inside", textinfo="percent+label")
            fig_cb.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=520)
            st.plotly_chart(fig_cb, use_container_width=True)

    st.markdown("### Top 50 firms by market cap")

    top_mkt = (
        table_base
        .sort_values("market_cap", ascending=False)
        .head(50)
        [[
            "IssuerName",
            "country_name",
            "carbon_burden",
            "market_cap",
            "cb_market_cap"
        ]]
        .rename(columns={
            "IssuerName": "Firm",
            "country_name": "Country",
            "carbon_burden": "Carbon Burden",
            "market_cap": "Market Cap",
            "cb_market_cap": "CB / Market Cap"
        })
    )

    st.dataframe(
        top_mkt,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Carbon Burden": st.column_config.NumberColumn(format="$%.2e"),
            "Market Cap": st.column_config.NumberColumn(format="$%.2e"),
            "CB / Market Cap": st.column_config.NumberColumn(format="%.2f")
        }
    )

    st.markdown("### Top 50 firms by carbon burden")

    top_cb = (
        table_base
        .sort_values("carbon_burden", ascending=False)
        .head(50)
        [[
            "IssuerName",
            "country_name",
            "carbon_burden",
            "market_cap",
            "cb_market_cap"
        ]]
        .rename(columns={
            "IssuerName": "Firm",
            "country_name": "Country",
            "carbon_burden": "Carbon Burden",
            "market_cap": "Market Cap",
            "cb_market_cap": "CB / Market Cap"
        })
    )

    st.dataframe(
        top_cb,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Carbon Burden": st.column_config.NumberColumn(format="$%.2e"),
            "Market Cap": st.column_config.NumberColumn(format="$%.2e"),
            "CB / Market Cap": st.column_config.NumberColumn(format="%.2f")
        }
    )

# =============================================================================
# Home
# =============================================================================

if st.session_state.page == "Home":

    cp_values = sorted(get_home_ratio_values(df, "CP"), reverse=True)
    b2c_values = sorted(get_home_ratio_values(df, "B2C"), reverse=True)

    cp_low = format_percentage(min(cp_values)) if cp_values else "N/A"
    cp_high = format_percentage(max(cp_values)) if cp_values else "N/A"

    b2c_low = format_percentage(min(b2c_values)) if b2c_values else "N/A"
    b2c_high = format_percentage(max(b2c_values)) if b2c_values else "N/A"

    firm_target = format_percentage(get_firm_home_ratio(firm_df, "Target"))
    firm_cp = format_percentage(get_firm_home_ratio(firm_df, "NGFSRMCP"))
    firm_nz = format_percentage(get_firm_home_ratio(firm_df, "NGFSRMNZ"))

    if st.session_state.home_stage == "Intro":

        st.markdown('<div class="main-title">Global Carbon Burden</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="home-subtitle-clean">
            Measuring future climate damages relative to the market value
            of the global corporate sector.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sub-page-title" style="text-align:center;">Aggregate Data</div>', unsafe_allow_html=True)

        st.markdown('<div class="scenario-wrapper">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">Current Policies Scenario</div>
                    <div class="scenario-range">{cp_low} — {cp_high}</div>
                    <div class="scenario-note">carbon burden as % of market value</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">Below 2°C Scenario</div>
                    <div class="scenario-range">{b2c_low} — {b2c_high}</div>
                    <div class="scenario-note">carbon burden as % of market value</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-page-title" style="text-align:center;">Firm-Level Data</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">ISS Target</div>
                    <div class="scenario-range">{firm_target}</div>
                    <div class="scenario-note">firm-level CB / market cap</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">NGFS Current Policies</div>
                    <div class="scenario-range">{firm_cp}</div>
                    <div class="scenario-note">firm-level CB / market cap</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">NGFS Net Zero 2050</div>
                    <div class="scenario-range">{firm_nz}</div>
                    <div class="scenario-note">firm-level CB / market cap</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        col1, col2 = st.columns([5.3, 1.2])

        with col2:
            st.markdown('<div class="continue-button">', unsafe_allow_html=True)
            if st.button("Continue →", key="continue_to_composition"):
                st.session_state.home_stage = "Composition"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.home_stage == "Composition":

        st.markdown('<div class="page-title">Global Composition</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-note">
            The charts below use aggregate country-level data and show how global carbon burden and corporate market value are distributed across corrected territorial units.
            The baseline uses Current Policies, a 2% discount rate, all future years, and averages across NGFS downscaling models.
            EU24 is treated as a single block, while Luxembourg, the Netherlands and Ireland remain separate countries.
            </div>
            """,
            unsafe_allow_html=True
        )

        baseline = get_composition_baseline(global_view_df, scenario="CP")

        pie_cb = make_pie_data(baseline, "carbon_burden", top_n=10)
        pie_mv = make_pie_data(baseline, "mkt_value", top_n=10)

        col1, col2 = st.columns(2)

        with col1:
            fig_cb = px.pie(
                pie_cb,
                names="display_unit",
                values="carbon_burden",
                title="Share of Global Carbon Burden",
                hole=0.35
            )
            fig_cb.update_traces(textposition="inside", textinfo="percent+label")
            fig_cb.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=560)
            st.plotly_chart(fig_cb, use_container_width=True)

        with col2:
            fig_mv = px.pie(
                pie_mv,
                names="display_unit",
                values="mkt_value",
                title="Share of Global Corporate Market Value",
                hole=0.35
            )
            fig_mv.update_traces(textposition="inside", textinfo="percent+label")
            fig_mv.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=560)
            st.plotly_chart(fig_mv, use_container_width=True)

        col1, col2 = st.columns([5.3, 1.2])

        with col2:
            st.markdown('<div class="continue-button">', unsafe_allow_html=True)
            if st.button("Continue →", key="continue_to_menu"):
                st.session_state.home_stage = "Menu"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    else:

        st.markdown('<div class="main-title">Global Carbon Burden</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="home-menu-subtitle">
            Explore the methodology or examine aggregate and firm-level results.
            </div>
            """,
            unsafe_allow_html=True
        )

        col_left, col_mid_left, col_mid_right, col_right = st.columns([1.0, 1.6, 1.6, 1.0])

        with col_mid_left:
            st.markdown('<div class="home-menu-card">', unsafe_allow_html=True)
            if st.button("Methodology", key="home_methodology"):
                st.session_state.page = "Methodology"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with col_mid_right:
            st.markdown('<div class="home-menu-card">', unsafe_allow_html=True)
            if st.button("Results", key="home_results"):
                st.session_state.page = "Results"
                st.session_state.results_stage = "Global"
                st.session_state.selected_display_unit = "World"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Methodology
# =============================================================================

elif st.session_state.page == "Methodology":

    navigation_bar()

    st.markdown('<div class="page-title">Firm Value vs Carbon Burden</div>', unsafe_allow_html=True)

    h1, h2, h3 = st.columns([1.4, 3.0, 2.3])
    h1.markdown('<div class="header-text">Concept</div>', unsafe_allow_html=True)
    h2.markdown('<div class="header-text">Meaning</div>', unsafe_allow_html=True)
    h3.markdown('<div class="header-text" style="text-align:center;">Formula</div>', unsafe_allow_html=True)

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 3.0, 2.3])
    c1.markdown('<div class="concept-text">Firm Value</div>', unsafe_allow_html=True)
    c2.markdown('<div class="meaning-text">Present value of future corporate cash flows.</div>', unsafe_allow_html=True)
    c3.latex(r"\sum_{\tau=1}^{\infty} \frac{CashFlow_{\tau}}{(1+r)^{\tau}}")

    st.markdown('<div class="light-rule"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 3.0, 2.3])
    c1.markdown('<div class="concept-text">Carbon Burden</div>', unsafe_allow_html=True)
    c2.markdown(
        """
        <div class="meaning-text">
        Present value of social costs generated by future greenhouse gas emissions,
        following the carbon burden framework of
        <a class="paper-link" href="https://www.nber.org/papers/w33110" target="_blank">
        Pástor, Stambaugh and Taylor (2024)</a>.
        </div>
        """,
        unsafe_allow_html=True
    )
    c3.latex(r"CB = \sum_{\tau=1}^{T} \frac{GHG_{\tau} \times SCC_{\tau}}{(1+\rho)^{\tau}}")

    st.markdown('<div class="light-rule"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 3.0, 2.3])
    c1.markdown('<div class="concept-text">Net Carbon Burden</div>', unsafe_allow_html=True)
    c2.markdown(
        '<div class="meaning-text">Carbon burden net of projected carbon tax payments, interpreted as the portion of climate damages not internalized through carbon pricing.</div>',
        unsafe_allow_html=True
    )
    c3.latex(r"CB_{net} = \sum_{\tau=1}^{T} \frac{GHG_{\tau} \times (SCC_{\tau} - t_{\tau})}{(1+\rho)^{\tau}}")

    st.markdown('<div class="light-rule"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 3.0, 2.3])
    c1.markdown('<div class="concept-text">Tax Burden Reduction</div>', unsafe_allow_html=True)
    c2.markdown('<div class="meaning-text">Reduction in carbon burden generated by projected carbon tax payments.</div>', unsafe_allow_html=True)
    c3.latex(r"TaxReduction = CB - CB_{net}")

    st.markdown('<div class="light-rule"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 3.0, 2.3])
    c1.markdown('<div class="concept-text">Carbon Burden / Market Value</div>', unsafe_allow_html=True)
    c2.markdown(
        '<div class="meaning-text">Carbon burden scaled by the market value of domestic corporations. This expresses climate liabilities relative to the value of the corporate sector.</div>',
        unsafe_allow_html=True
    )
    c3.latex(r"\frac{CB}{MarketValue}")

    st.markdown(
        """
        <div class="note">
        The dashboard combines aggregate country-level data with firm-level projected emissions data.
        Aggregate results use NGFS Phase 5 emissions projections, social cost of carbon estimates, projected carbon pricing, and country-level corporate market value data.
        Firm-level results use issuer-level projected emissions pathways and market capitalization.
        Monetary values are expressed in 2023 U.S. dollars.
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# Results
# =============================================================================

elif st.session_state.page == "Results":

    navigation_bar()

    if st.session_state.results_stage != "Global":
        if st.button("← Back to global results", key="back_to_global_results"):
            st.session_state.results_stage = "Global"
            st.session_state.selected_display_unit = "World"
            st.rerun()

    if st.session_state.results_stage == "Global":

        st.markdown('<div class="page-title">Results</div>', unsafe_allow_html=True)

        st.markdown('<div class="sub-page-title">Aggregate Results using Country-Level Data</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="section-note">
            The global map uses corrected territorial units. EU24 is shown as a single block;
            Luxembourg, the Netherlands and Ireland remain separate countries. Regional blocks are used where country-level ratios are less informative.
            Click on a territory to open its detail page. Firm-level distributions for the selected territory are shown below.
            </div>
            """,
            unsafe_allow_html=True
        )

        (
            filtered_df,
            selected_model,
            selected_scenario,
            selected_horizon,
            selected_discount,
            selected_variable,
            selected_variable_label
        ) = show_aggregate_filters(global_map_df, key_prefix="global_agg")

        (
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        ) = show_firm_filters(key_prefix="global_firm")

        plot_df = filtered_df.copy()
        plot_df, colorbar_title, hover_suffix = prepare_plot_values(plot_df, selected_variable)

        fig = px.choropleth(
            plot_df,
            locations="region",
            color=selected_variable,
            hover_name="display_unit",
            locationmode="ISO-3",
            color_continuous_scale="RdYlBu_r",
            custom_data=["display_unit", "view_type"],
            title=f"{selected_variable_label} | {selected_model} | {selected_scenario} | {selected_horizon}"
        )

        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>"
            + selected_variable_label
            + ": %{z:.2f}"
            + hover_suffix
            + "<extra></extra>"
        )

        fig.update_layout(
            geo=dict(showframe=False, showcoastlines=True),
            height=700,
            font=dict(family="Georgia, Times New Roman, serif"),
            coloraxis_colorbar=dict(title=colorbar_title),
            title=dict(font=dict(size=22))
        )

        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="global_results_map",
            on_select="rerun",
            selection_mode="points"
        )

        try:
            selected_points = event["selection"]["points"]
            if selected_points:
                clicked_unit = selected_points[0]["customdata"][0]
                clicked_type = selected_points[0]["customdata"][1]

                st.session_state.selected_display_unit = clicked_unit

                if clicked_type == "eu_submap":
                    st.session_state.results_stage = "EU24"
                elif clicked_type == "regional_pies":
                    st.session_state.results_stage = "Regional"
                else:
                    st.session_state.results_stage = "Country"

                st.rerun()
        except Exception:
            pass

        st.markdown("### Aggregate ranking")

        ranking_df = filtered_df.drop_duplicates(subset=["display_unit"]).copy()
        ranking_df = ranking_df.sort_values(selected_variable, ascending=False).reset_index(drop=True)
        ranking_df.insert(0, "Ranking", ranking_df.index + 1)

        ranking_df = ranking_df[[
            "Ranking",
            "display_unit",
            "carbon_burden",
            "carbon_burden_net",
            "tax_burden_reduction",
            "mkt_value",
            "cb_mkt_value",
            "cb_net_mkt_value"
        ]].rename(columns={
            "display_unit": "Territory",
            "carbon_burden": "Carbon Burden",
            "carbon_burden_net": "Net Carbon Burden",
            "tax_burden_reduction": "Tax Burden Reduction",
            "mkt_value": "Market Value",
            "cb_mkt_value": "Carbon Burden / Market Value",
            "cb_net_mkt_value": "Net Carbon Burden / Market Value"
        })

        st.dataframe(
            ranking_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Carbon Burden": st.column_config.NumberColumn(format="$%.2e"),
                "Net Carbon Burden": st.column_config.NumberColumn(format="$%.2e"),
                "Tax Burden Reduction": st.column_config.NumberColumn(format="$%.2e"),
                "Market Value": st.column_config.NumberColumn(format="$%.2e"),
                "Carbon Burden / Market Value": st.column_config.NumberColumn(format="%.2%"),
                "Net Carbon Burden / Market Value": st.column_config.NumberColumn(format="%.2%")
            }
        )

        render_firm_density_panel(
            firm_df,
            "World",
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        )

    elif st.session_state.results_stage == "EU24":

        st.markdown('<div class="page-title">EU24 Detail</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-note">
            This submap decomposes the EU24 aggregate into member countries.
            Luxembourg, the Netherlands and Ireland are not included in EU24 and remain separate countries in the global layer.
            </div>
            """,
            unsafe_allow_html=True
        )

        (
            filtered_df,
            selected_model,
            selected_scenario,
            selected_horizon,
            selected_discount,
            selected_variable,
            selected_variable_label
        ) = show_aggregate_filters(eu_detail_df, key_prefix="eu_agg")

        (
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        ) = show_firm_filters(key_prefix="eu_firm")

        plot_df = filtered_df.copy()
        plot_df, colorbar_title, hover_suffix = prepare_plot_values(plot_df, selected_variable)

        fig = px.choropleth(
            plot_df,
            locations="region",
            color=selected_variable,
            hover_name="country",
            locationmode="ISO-3",
            color_continuous_scale="RdYlBu_r",
            title=f"EU24 | {selected_variable_label} | {selected_model} | {selected_scenario} | {selected_horizon}"
        )

        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>"
            + selected_variable_label
            + ": %{z:.2f}"
            + hover_suffix
            + "<extra></extra>"
        )

        fig.update_layout(
            geo=dict(scope="europe", showframe=False, showcoastlines=True),
            height=700,
            font=dict(family="Georgia, Times New Roman, serif"),
            coloraxis_colorbar=dict(title=colorbar_title),
            title=dict(font=dict(size=22))
        )

        st.plotly_chart(fig, use_container_width=True)

        render_firm_density_panel(
            firm_df,
            "EU24",
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        )

    elif st.session_state.results_stage == "Regional":

        selected_region = st.session_state.selected_display_unit

        st.markdown(f'<div class="page-title">{selected_region}</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="section-note">
            For regional blocks, the dashboard shows the internal distribution of carbon burden and market value rather than emphasizing country-level ratios.
            </div>
            """,
            unsafe_allow_html=True
        )

        (
            filtered_df,
            selected_model,
            selected_scenario,
            selected_horizon,
            selected_discount,
            selected_variable,
            selected_variable_label
        ) = show_aggregate_filters(regional_detail_df, include_variable=False, key_prefix="regional_agg")

        (
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        ) = show_firm_filters(key_prefix="regional_firm")

        region_df = filtered_df[
            filtered_df["display_unit"] == selected_region
        ].copy()

        pie_cb = make_pie_data(region_df, "carbon_burden", label_col="country", top_n=10)
        pie_mv = make_pie_data(region_df, "mkt_value", label_col="country", top_n=10)

        col1, col2 = st.columns(2)

        with col1:
            fig_cb = px.pie(
                pie_cb,
                names="country",
                values="carbon_burden",
                title=f"{selected_region}: Share of Carbon Burden",
                hole=0.35
            )
            fig_cb.update_traces(textposition="inside", textinfo="percent+label")
            fig_cb.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=560)
            st.plotly_chart(fig_cb, use_container_width=True)

        with col2:
            fig_mv = px.pie(
                pie_mv,
                names="country",
                values="mkt_value",
                title=f"{selected_region}: Share of Market Value",
                hole=0.35
            )
            fig_mv.update_traces(textposition="inside", textinfo="percent+label")
            fig_mv.update_layout(font=dict(family="Georgia, Times New Roman, serif"), height=560)
            st.plotly_chart(fig_mv, use_container_width=True)

        render_firm_density_panel(
            firm_df,
            selected_region,
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        )

    elif st.session_state.results_stage == "Country":

        selected_country = st.session_state.selected_display_unit

        st.markdown(f'<div class="page-title">{selected_country}</div>', unsafe_allow_html=True)

        (
            filtered_df,
            selected_model,
            selected_scenario,
            selected_horizon,
            selected_discount,
            selected_variable,
            selected_variable_label
        ) = show_aggregate_filters(global_view_df, key_prefix="country_agg")

        (
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        ) = show_firm_filters(key_prefix="country_firm")

        country_df = filtered_df[
            filtered_df["display_unit"] == selected_country
        ].copy()

        if country_df.empty:
            st.warning("No aggregate data available for this territory under the selected filters.")
        else:
            row = country_df.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric("Aggregate Carbon Burden", format_trillion(row["carbon_burden"]))
            col2.metric("Aggregate Market Value", format_trillion(row["mkt_value"]))
            col3.metric("Aggregate CB / Market Value", format_percentage(row["cb_mkt_value"]))

        render_firm_density_panel(
            firm_df,
            selected_country,
            firm_pathway,
            firm_pathway_label,
            firm_horizon,
            firm_horizon_label
        )



