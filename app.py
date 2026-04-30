import streamlit as st
import pandas as pd
import plotly.express as px

# =============================================================================
# Page configuration
# =============================================================================

st.set_page_config(page_title="Global Carbon Burden", layout="wide")

# =============================================================================
# Load data
# =============================================================================

df = pd.read_csv("cb_dataset_final.csv")
df_map = df[~df["region"].isin(["EU27", "World", "OECD"])].copy()

# =============================================================================
# Labels
# =============================================================================

SCENARIO_LABELS = {
    "B2C": "Below 2°C",
    "CP": "Current Policies",
    "DT": "Delayed Transition",
    "FW": "Fragmented World",
    "LD": "Low Demand",
    "NDC": "Nationally Determined Contributions",
    "NZ": "Net Zero 2050"
}

SCENARIO_EXPLANATIONS = {
    "B2C": "Scenario aligned with climate policies consistent with limiting warming below 2°C. Requires strong and relatively early policy action.",
    "CP": "Baseline scenario reflecting currently implemented policies only. No additional climate action beyond what is already in place.",
    "DT": "Climate policies are delayed, then introduced abruptly. Leads to higher transition risks and more disruptive adjustments.",
    "FW": "No coordinated global climate policy. Countries act independently, leading to high physical and transition risks.",
    "LD": "Strong behavioral and efficiency changes reduce energy demand. Lower emissions achieved with less reliance on high carbon prices.",
    "NDC": "Reflects countries' pledged climate targets under the Paris Agreement. Moderate transition effort, not sufficient for 2°C.",
    "NZ": "Ambitious scenario targeting net-zero emissions globally by 2050. Requires immediate and stringent climate policies."
}

MODEL_EXPLANATION = """
The NGFS scenarios are produced by the Network for Greening the Financial System, 
a group of central banks and supervisors working on climate- and environment-related 
risk management in the financial sector.

The scenarios are generated using integrated assessment models such as GCAM, 
MESSAGE-GLOBIOM and REMIND-MAgPIE, together with the NiGEM macroeconomic model.

[Open NGFS model documentation](https://zenodo.org/records/17901363)
"""

VARIABLE_LABELS = {
    "carbon_burden": "Carbon Burden",
    "carbon_burden_net": "Net Carbon Burden",
    "tax_burden_reduction": "Tax Burden Reduction",
    "market_value": "Market Value",
    "cb_mkt_value": "Carbon Burden / Market Value",
    "cb_net_mkt_value": "Net Carbon Burden / Market Value"
}

VARIABLES = list(VARIABLE_LABELS.keys())

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
    except:
        return str(x)

def clean_model_label(x):
    return str(x).replace("_", " ").title()

def is_percentage_variable(variable):
    return variable in ["cb_mkt_value", "cb_net_mkt_value"]

def format_trillion(x):
    if pd.isna(x):
        return ""
    return f"${x / 1e12:,.2f}T"

def format_home_values(values):
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "not available"
    return ", ".join([format_trillion(v) for v in values])

def get_home_scenario_values(data, scenario):
    temp = data.copy()

    temp = temp[
        (temp["region"] == "World") &
        (temp["scenario"] == scenario) &
        (temp["discount_rate"].astype(float).round(4) == 0.02)
    ].copy()

    if "all_future_years" in temp["horizon_label"].astype(str).unique():
        temp = temp[temp["horizon_label"].astype(str) == "all_future_years"].copy()

    temp = temp.sort_values("model")

    return temp["carbon_burden"].tolist()

# =============================================================================
# Session state
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "home_stage" not in st.session_state:
    st.session_state.home_stage = "Intro"

if "selected_country" not in st.session_state:
    st.session_state.selected_country = "All Countries"

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
        margin-top: 110px;
        margin-bottom: 24px;
        color: #252634;
        letter-spacing: -1px;
    }

    .home-statement {
        max-width: 980px;
        margin: auto;
        text-align: center;
        font-size: 25px;
        line-height: 1.55;
        color: #4d4d55;
    }

    .home-statement strong {
        color: #7b2431;
        font-weight: 700;
    }

    .continue-area {
        margin-top: 110px;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-right: 70px;
    }

    .continue-text {
        font-size: 22px;
        color: #7b2431;
        font-weight: 600;
        margin-right: 16px;
    }

    .continue-arrow {
        font-size: 42px;
        color: #7b2431;
    }

    .page-title {
        font-size: 44px;
        font-weight: 500;
        color: #7b2431;
        margin-bottom: 30px;
    }

    .section-note {
        font-size: 17px;
        color: #444444;
        margin-bottom: 25px;
        line-height: 1.45;
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

    .nav-wrapper {
        padding: 12px 18px;
        border: 1px solid #e3e0e0;
        border-radius: 20px;
        background: #fbfbfc;
        box-shadow: 0 6px 18px rgba(30, 30, 40, 0.04);
        margin-bottom: 34px;
    }

    .menu-button button {
        height: 92px !important;
        font-size: 28px !important;
        border-radius: 22px !important;
        background: #fbfbfc !important;
        border: 1px solid #d8d3d5 !important;
        box-shadow: 0 10px 26px rgba(30, 30, 40, 0.06);
    }

    .menu-button button:hover {
        background: #f3ecef !important;
        border-color: #7b2431 !important;
        color: #7b2431 !important;
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
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# Navigation
# =============================================================================

def navigation_bar():
    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Home", key="nav_home"):
            st.session_state.page = "Home"
            st.session_state.home_stage = "Intro"
            st.rerun()

    with col2:
        if st.button("Methodology", key="nav_methodology"):
            st.session_state.page = "Methodology"
            st.rerun()

    with col3:
        if st.button("Map", key="nav_map"):
            st.session_state.page = "Map"
            st.rerun()

    with col4:
        if st.button("World", key="nav_world"):
            st.session_state.page = "World"
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Filters
# =============================================================================

def show_filters(data):
    st.sidebar.title("Controls")

    model_options = sorted(data["model"].dropna().unique())
    model_reverse = {clean_model_label(x): x for x in model_options}

    selected_model_label = st.sidebar.selectbox("Model", list(model_reverse.keys()))
    selected_model = model_reverse[selected_model_label]

    with st.sidebar.expander("About NGFS models"):
        st.markdown(MODEL_EXPLANATION)

    scenario_options = sorted(data["scenario"].dropna().unique())
    scenario_display = [f"{x} — {SCENARIO_LABELS.get(x, x)}" for x in scenario_options]

    selected_scenario_display = st.sidebar.selectbox("Scenario", scenario_display)
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

    selected_horizon_label = st.sidebar.selectbox("Horizon", list(horizon_reverse.keys()))
    selected_horizon = horizon_reverse[selected_horizon_label]

    discount_options = sorted(data["discount_rate"].dropna().unique())
    discount_reverse = {clean_discount_label(x): x for x in discount_options}

    selected_discount_label = st.sidebar.selectbox("Discount Rate", list(discount_reverse.keys()))
    selected_discount = discount_reverse[selected_discount_label]

    variable_reverse = {VARIABLE_LABELS[v]: v for v in VARIABLES if v in data.columns}

    selected_variable_label = st.sidebar.selectbox("Variable", list(variable_reverse.keys()))
    selected_variable = variable_reverse[selected_variable_label]

    country_options = ["All Countries"] + sorted(data["country"].dropna().unique())

    default_country = st.session_state.get("selected_country", "All Countries")
    if default_country not in country_options:
        default_country = "All Countries"

    selected_country = st.sidebar.selectbox(
        "Country",
        country_options,
        index=country_options.index(default_country),
        help="Start typing to search for a country."
    )

    st.session_state.selected_country = selected_country

    filtered = data[
        (data["model"] == selected_model) &
        (data["scenario"] == selected_scenario) &
        (data["horizon_label"] == selected_horizon) &
        (data["discount_rate"] == selected_discount)
    ].copy()

    if selected_country != "All Countries":
        filtered = filtered[filtered["country"] == selected_country].copy()

    return (
        filtered,
        selected_model_label,
        selected_scenario,
        selected_horizon_label,
        selected_discount_label,
        selected_variable,
        selected_variable_label,
        selected_country
    )

# =============================================================================
# Home
# =============================================================================


if st.session_state.page == "Home":

    cp_values = sorted(get_home_scenario_values(df, "CP"), reverse=True)
    nz_values = sorted(get_home_scenario_values(df, "NZ"), reverse=True)

    cp_low = format_trillion(min(cp_values)) if cp_values else "N/A"
    cp_high = format_trillion(max(cp_values)) if cp_values else "N/A"

    nz_low = format_trillion(min(nz_values)) if nz_values else "N/A"
    nz_high = format_trillion(max(nz_values)) if nz_values else "N/A"

    if st.session_state.home_stage == "Intro":

        st.markdown(
            """
            <style>
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
                max-width: 1100px;
                margin: auto;
                margin-top: 6px;
                margin-bottom: 30px;
            }

            .scenario-card {
                border: 1px solid #ddd9db;
                border-radius: 22px;
                padding: 42px 20px 38px 20px;
                background-color: #fcfcfc;
                box-shadow: 0 12px 30px rgba(20,20,30,0.04);
                text-align: center;
                height: 100%;
            }

            .scenario-title {
                font-size: 18px;
                letter-spacing: 1.8px;
                font-weight: 700;
                color: #7b2431;
                margin-bottom: 26px;
                text-transform: uppercase;
            }

            .scenario-range {
                font-size: 42px;
                font-weight: 700;
                color: #252634;
                line-height: 1.25;
                margin-bottom: 16px;
            }

            .scenario-note {
                font-size: 18px;
                color: #666666;
                font-style: italic;
            }

            .continue-clean {
                display: flex;
                justify-content: flex-end;
                margin-top: 25px;
                padding-right: 55px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Main title
        st.markdown(
            '<div class="main-title">Global Carbon Burden</div>',
            unsafe_allow_html=True
        )

        # Subtitle
        st.markdown(
            """
            <div class="home-subtitle-clean">
            Measuring the present value of future climate damages
            across countries, scenarios, and climate-economy models.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Scenario cards
        st.markdown('<div class="scenario-wrapper">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">Current Policies Scenario</div>
                    <div class="scenario-range">{cp_low} — {cp_high}</div>
                    <div class="scenario-note">estimated global carbon burden</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div class="scenario-card">
                    <div class="scenario-title">Net Zero 2050 Scenario</div>
                    <div class="scenario-range">{nz_low} — {nz_high}</div>
                    <div class="scenario-note">estimated global carbon burden</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # Continue button
        col1, col2 = st.columns([5.3, 1.2])

        with col2:
            st.markdown('<div class="continue-button">', unsafe_allow_html=True)
            if st.button("Continue →", key="continue_button"):
                st.session_state.home_stage = "Menu"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    else:

        st.markdown(
            '<div class="main-title">Global Carbon Burden</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="home-subtitle-clean" style="margin-bottom: 55px;">
            Explore the methodology, compare countries on the global map,
            or examine the full world ranking.
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([1.4, 2.2, 1.4])

        with col2:

            st.markdown('<div class="menu-button">', unsafe_allow_html=True)

            if st.button("Methodology", key="home_methodology"):
                st.session_state.page = "Methodology"
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Map", key="home_map"):
                st.session_state.page = "Map"
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("World", key="home_world"):
                st.session_state.page = "World"
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
        The dashboard currently combines NGFS Phase 5 emissions projections,
        social cost of carbon estimates, projected carbon pricing, and country-level
        corporate market value data. Monetary values are expressed in 2023 U.S. dollars.
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# Map
# =============================================================================

elif st.session_state.page == "Map":

    navigation_bar()

    st.markdown('<div class="page-title">Carbon Burden World Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Monetary values are expressed in 2023 U.S. dollars. Ratio variables are shown as percentages of corporate market value. Click on a country to open its table entry.</div>',
        unsafe_allow_html=True
    )

    (
        filtered_df,
        selected_model,
        selected_scenario,
        selected_horizon,
        selected_discount,
        selected_variable,
        selected_variable_label,
        selected_country
    ) = show_filters(df_map)

    plot_df = filtered_df.copy()

    if is_percentage_variable(selected_variable):
        plot_df[selected_variable] = plot_df[selected_variable] * 100
        colorbar_title = "%"
        hover_suffix = "%"
    elif selected_variable in ["carbon_burden", "carbon_burden_net", "tax_burden_reduction", "market_value"]:
        plot_df[selected_variable] = plot_df[selected_variable] / 1e12
        colorbar_title = "Trillion 2023 USD"
        hover_suffix = " trillion 2023 USD"
    else:
        colorbar_title = selected_variable_label
        hover_suffix = ""

    fig = px.choropleth(
        plot_df,
        locations="region",
        color=selected_variable,
        hover_name="country",
        locationmode="ISO-3",
        color_continuous_scale="RdYlBu_r",
        title=f"{selected_variable_label} | {selected_model} | {selected_scenario} | {selected_horizon}"
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>"
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
        key="carbon_map",
        on_select="rerun",
        selection_mode="points"
    )

    try:
        selected_points = event["selection"]["points"]
        if selected_points:
            clicked_country = selected_points[0]["hovertext"]
            st.session_state.selected_country = clicked_country
            st.session_state.page = "World"
            st.rerun()
    except Exception:
        pass

# =============================================================================
# World table
# =============================================================================

elif st.session_state.page == "World":

    navigation_bar()

    st.markdown('<div class="page-title">Carbon Burden World Ranking</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Monetary values are expressed in trillions of 2023 U.S. dollars. Ratio variables are expressed as percentages of corporate market value.</div>',
        unsafe_allow_html=True
    )

    (
        filtered_df,
        selected_model,
        selected_scenario,
        selected_horizon,
        selected_discount,
        selected_variable,
        selected_variable_label,
        selected_country
    ) = show_filters(df)

    table_df = filtered_df.copy()

    with st.expander("Search country in table", expanded=False):
        search_country = st.text_input("Type a country name", "")
        if search_country.strip():
            table_df = table_df[
                table_df["country"].str.contains(search_country.strip(), case=False, na=False)
            ].copy()

    table_df = table_df.sort_values(by=selected_variable, ascending=False).reset_index(drop=True)
    table_df.insert(0, "Ranking", table_df.index + 1)

    keep_cols = [
        "Ranking",
        "country",
        "carbon_burden",
        "carbon_burden_net",
        "tax_burden_reduction",
        "market_value",
        "cb_mkt_value",
        "cb_net_mkt_value"
    ]

    keep_cols = [c for c in keep_cols if c in table_df.columns]
    table_df = table_df[keep_cols].copy()

    table_df = table_df.rename(columns={
        "country": "Country",
        "carbon_burden": "Carbon Burden",
        "carbon_burden_net": "Net Carbon Burden",
        "tax_burden_reduction": "Tax Burden Reduction",
        "market_value": "Market Value",
        "cb_mkt_value": "Carbon Burden / Market Value",
        "cb_net_mkt_value": "Net Carbon Burden / Market Value"
    })

    for col in ["Carbon Burden", "Net Carbon Burden", "Tax Burden Reduction", "Market Value"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: f"${x / 1e12:,.2f}T" if pd.notna(x) else "")

    for col in ["Carbon Burden / Market Value", "Net Carbon Burden / Market Value"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].apply(lambda x: f"{x * 100:,.2f}%" if pd.notna(x) else "")

    st.dataframe(table_df, use_container_width=True, hide_index=True)
