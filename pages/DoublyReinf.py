import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head 3 – Doubly Reinforced Beam Design", layout="wide")

# --- CSS for Tighter Mobile/Print Layout ---
st.markdown("""
<style>
/* Overall reduction in spacing (padding and margin) for mobile/A4 */
.st-emotion-cache-18ni7ap, 
.st-emotion-cache-1wb9b6h { 
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
}

/* Smaller font size for markdown headers and labels */
h1, h2, h3, h4, .stMarkdown, .st-emotion-cache-vdhr9j {
    font-size: 0.9rem; 
    line-height: 1.2;
}

/* Tighter text and smaller fonts for labels/results */
span, label {
    font-size: 0.8rem !important;
}

/* Smaller selectboxes and number inputs */
.stSelectbox, .stNumberInput {
    height: 30px; 
}
.stSelectbox>div>div, .stNumberInput>div>div>input {
    min-height: 30px !important;
    padding: 2px 5px !important; 
    font-size: 0.8rem !important;
}

/* Tighter columns (less spacing between elements) */
.st-emotion-cache-p2n2mc, .st-emotion-cache-16ya12x { 
    gap: 0.5rem; 
}

/* Tighter table/dataframe */
.stDataFrame {
    font-size: 0.75rem; 
}

/* Print-friendly: Ensure content is legible on A4 and hide Streamlit UI */
@media print {
    .st-emotion-cache-6v09g0, 
    .st-emotion-cache-1avcm0d {
        display: none !important;
    }
    .st-emotion-cache-1vq4p4c {
        max-width: 100% !important;
    }
    body {
        -webkit-print-color-adjust: exact;
        background-color: white !important;
        color: black !important;
    }
}
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------


# ---------- Styles ----------
BLUE = "#1f6feb"
RED = "#d11a2a"
OK = "<span style='color:#0a8a0a;font-weight:700'>Okay</span>"
NOT_OK = "<span style='color:#c1121f;font-weight:700'>Not OK</span>"

def blue(s):  return f"<span style='color:{BLUE};font-weight:600'>{s}</span>"
def red(s):   return f"**<span style='color:{RED}'>{s}</span>**"
def label(md): st.markdown(md, unsafe_allow_html=True)

# ---------- Helpers (IS 456-2000 Specific) ----------

# Max xu/d ratio based on steel grade (Cl 38.1)
def xu_max_ratio(fy):
    if fy <= 250: return 0.53
    if fy <= 415: return 0.48
    return 0.46

# Limiting moment coefficient (0.36 fck (xu/d) * (1 - 0.42 xu/d)) * b*d^2
def R_lim(fck, fy):
    xu_d = xu_max_ratio(fy)
    return 0.36 * fck * xu_d * (1 - 0.42 * xu_d)

# Stress in compression steel (fsc) in N/mm² based on fy and d'/d
# Reference: Annex E, IS 456:2000 for strain and corresponding stress tables
# This function approximates the value based on standard tables
def fsc_calc(fy, d_prime_over_d):
    if fy == 250: # Fe 250 (fsc is always 0.87 fy = 217.5)
        return 217.5
    
    # Fe 415 (approximate/interpolated values for common d'/d ratios)
    if fy == 415:
        if d_prime_over_d <= 0.05: return 355.0
        if d_prime_over_d <= 0.10: return 352.0
        if d_prime_over_d <= 0.15: return 342.0
        if d_prime_over_d <= 0.20: return 329.0
        return 300.0 # Default for higher ratios

    # Fe 500 (approximate/interpolated values for common d'/d ratios)
    if fy == 500:
        if d_prime_over_d <= 0.05: return 424.0
        if d_prime_over_d <= 0.10: return 412.0
        if d_prime_over_d <= 0.15: return 395.0
        if d_prime_over_d <= 0.20: return 370.0
        return 350.0 # Default for higher ratios
    
    # Simple calculation for custom fy where strain > yield strain
    # Strain = 0.0035 * (1 - d'/xu_max)
    # If strain >= yield, use 0.87 fy. Else, calculate stress.
    xu_max = xu_max_ratio(fy) * 1 # Assuming d=1 for ratio calculation
    if d_prime_over_d <= xu_max * (1 - (0.002 + 0.87*fy/200000)/0.0035):
         return 0.87 * fy
         
    # More complex interpolation is needed for accuracy, but for a simple app:
    return 0.87 * fy - 0.05 * fy / d_prime_over_d


# ---------- Title ----------
default_title = "Item 3: Design of Doubly Reinforced Beam (IS 456:2000)"
header_text = st.text_input("Header/Title", value=default_title)
st.title(header_text)

st.markdown("---")

# ---------- Materials & Geometry ----------
st.header("Materials & Geometry (Inputs)")
st.markdown("""
**📝 NARRATIVE:** Required when $M_u$ exceeds the limiting moment capacity $M_{u, \lim}$.
- **$d$**: Effective depth to tension steel.
- **$d'$**: Effective cover to compression steel.
""")
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.markdown(blue("fck (MPa)"), unsafe_allow_html=True)
    fck = st.selectbox("fck", [20, 25, 30, 35, 40], index=0)
with c2:
    st.markdown(blue("fy (MPa)"), unsafe_allow_html=True)
    fy = st.selectbox("fy", [415, 500, 250], index=0)
with c3:
    st.markdown(blue("b (mm)"), unsafe_allow_html=True)
    b = st.number_input("b", value=250.0, step=10.0, min_value=100.0, format="%f")
with c4:
    st.markdown(blue("Effective depth d (mm)"), unsafe_allow_html=True)
    d = st.number_input("d", value=500.0, step=10.0, min_value=100.0, format="%f")
with c5:
    st.markdown(blue("Effective cover d' (mm)"), unsafe_allow_html=True)
    d_prime = st.number_input("d_prime", value=50.0, step=5.0, min_value=20.0, format="%f", help="Cover to centroid of Asc")

st.markdown("---")

# ---------- Actions (Factored) ----------
st.header("Factored Moment $\mathbf{M_u}$")
c1, c2 = st.columns(2)
with c1:
    st.markdown(blue("Mu (kNm)"), unsafe_allow_html=True)
    Mu = st.number_input("Mu", value=200.0, step=5.0, min_value=0.0, format="%f")

# --- CORE DOUBLY REINFORCED CALCULATIONS ---

# 1. Calculate Mu,lim
R_lim_val = R_lim(fck, fy)
Mu_lim_kNm = R_lim_val * b * d**2 / 1e6

# 2. Check if doubly reinforced section is required
is_doubly_required = Mu > Mu_lim_kNm

st.info(f"Limiting Moment $\mathbf{{M_{{u, \lim}}}}$: **{Mu_lim_kNm:.2f} kNm**")

if not is_doubly_required:
    st.success(f"**$\mathbf{{M_u}}$ ({Mu:.2f} kNm) $\leq$ $\mathbf{{M_{{u, \lim}}}}$ ({Mu_lim_kNm:.2f} kNm). Doubly reinforced section is {OK}. Proceed with singly reinforced design.")
    st.stop()

st.error(f"**$\mathbf{{M_u}}$ ({Mu:.2f} kNm) $>$ $\mathbf{{M_{{u, \lim}}}}$ ({Mu_lim_kNm:.2f} kNm). Doubly reinforced section is {NOT_OK}.")
st.markdown("---")

# 3. Splitting the moment
Mu2_kNm = Mu - Mu_lim_kNm
st.header("Moment Components")
c_m1, c_m2 = st.columns(2)
with c_m1:
    st.markdown(f"Moment resisted by concrete & $\mathbf{{A_{{st1}}}}$ ($\mathbf{{M_{{u, \lim}}}}$): {blue(f'{Mu_lim_kNm:.2f} kNm')}")
with c_m2:
    st.markdown(f"Remaining moment resisted by $\mathbf{{A_{{sc}}}}$ & $\mathbf{{A_{{st2}}}}$ ($\mathbf{{M_{{u2}}}}$): {red(f'{Mu2_kNm:.2f} kNm')}")

# 4. Stress in Compression Steel (fsc)
d_prime_over_d = d_prime / d
fsc = fsc_calc(fy, d_prime_over_d)

st.header("Compression Steel Stress Check")
label(f"{blue('d\'/d ratio')}: {d_prime_over_d:.3f}")
label(f"{blue('Stress in Compression Steel (fsc)')}: **{fsc:.2f} N/mm²**")

# 5. Calculate Required Steel Areas (Ast1, Asc, Ast2)

# Ast1 (Tension steel to balance Mu,lim)
Ast1 = (0.36 * fck * b * xu_max_ratio(fy) * d) / (0.87 * fy)

# Asc (Compression steel to resist Mu2)
# Mu2 = Asc * (fsc - 0.45 fck) * (d - d')
Asc = (Mu2_kNm * 1e6) / ((fsc - 0.45 * fck) * (d - d_prime))

# Ast2 (Tension steel to balance Asc)
Ast2 = (Asc * (fsc - 0.45 * fck)) / (0.87 * fy)

# Total Tension Steel
Ast_total = Ast1 + Ast2

# --- FINAL RESULTS ---
st.markdown("---")
st.header("Required Reinforcement Areas (IS 456:2000)")

c_ast1, c_asc, c_ast2, c_ast_total = st.columns(4)
with c_ast1:
    st.markdown(f"**$\mathbf{{A_{{st1}}}}$ (mm²)**: (from $\text{M}_{u, \lim}$)")
    st.info(f"{Ast1:.2f}")
with c_asc:
    st.markdown(f"**$\mathbf{{A_{{sc}}}}$ (mm²)**: (Compression Steel)")
    st.info(f"{Asc:.2f}")
with c_ast2:
    st.markdown(f"**$\mathbf{{A_{{st2}}}}$ (mm²)**: (from $\text{M}_{u2}$)")
    st.info(f"{Ast2:.2f}")
with c_ast_total:
    st.markdown(f"**$\mathbf{{A_{{st, total}}}}$ (mm²)**: ($\text{A}_{st1} + \text{A}_{st2}$)")
    st.info(f"{Ast_total:.2f}")

# --- Summary ---
st.markdown("---")
st.header("Design Summary")
st.markdown(f"""
- **Design Factored Moment $\mathbf{{M_u}}$**: {Mu:.2f} kNm
- **Limiting Moment $\mathbf{{M_{{u, \lim}}}}$**: {Mu_lim_kNm:.2f} kNm
- **Required Tension Steel ($\mathbf{{A_{{st, total}}}}$)**: **{Ast_total:.2f} mm²**
- **Required Compression Steel ($\mathbf{{A_{{sc}}}}$)**: **{Asc:.2f} mm²**
- *Note: $\text{{A}}_{{st}}$ must also satisfy minimum and maximum area requirements.*
""")
