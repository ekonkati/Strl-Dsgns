import math
import streamlit as st

# Set page configuration for mobile-friendly layout
st.set_page_config(page_title="Head 3 – Doubly Reinforced Beam Design", layout="wide")

# ====================================================================
# *** CONSTANTS AND DATA TABLES (DEFINED AT THE TOP FOR STABILITY) ***
# ====================================================================

# Styles
BLUE = "#1f6feb"
RED = "#d11a2a"
OK = "<span style='color:#0a8a0a;font-weight:700'>Okay</span>"
NOT_OK = "<span style='color:#c1121f;font-weight:700'>Not OK</span>"

# IS 456 Annex E Data: d'/xu,max * 100 vs fsc (N/mm²)
FSC_RATIOS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] 
FY415_FSC_VALS = [360.9, 360.9, 351.8, 342.8, 333.7, 324.6, 315.5, 306.4]
FY500_FSC_VALS = [434.8, 434.8, 424.4, 411.3, 395.1, 370.5, 347.5, 324.5]


# ====================================================================
# *** HELPER FUNCTIONS ***
# ====================================================================

def blue(s):  return f"<span style='color:{BLUE};font-weight:600'>{s}</span>"
def red(s):   return f"**<span style='color:{RED}'>{s}</span>**"
def label(md): st.markdown(md, unsafe_allow_html=True)

# Max xu/d ratio based on steel grade (Cl 38.1)
def xu_max_ratio(fy):
    if fy <= 250: return 0.53
    if fy <= 415: return 0.48
    return 0.46

# Limiting moment coefficient 
def R_lim(fck, fy):
    xu_d = xu_max_ratio(fy)
    return 0.36 * fck * xu_d * (1 - 0.42 * xu_d)

# Stress in compression steel (fsc) in N/mm² based on fy and d'/d (IS 456 Annex E)
def fsc_calc(fy, d_prime_over_d):
    """Calculates fsc using robust, basic Python interpolation."""
    
    if fy == 250:
        return 0.87 * fy

    if fy == 415:
        fsc_vals = FY415_FSC_VALS
    elif fy == 500:
        fsc_vals = FY500_FSC_VALS
    else:
        return 0.87 * fy

    xu_d_max = xu_max_ratio(fy)
    d_prime_over_xu_max = d_prime_over_d / xu_d_max
    ratio_pct = d_prime_over_xu_max * 100 

    # Boundary checks
    if ratio_pct <= FSC_RATIOS[0]: return fsc_vals[0]
    if ratio_pct >= FSC_RATIOS[-1]: return fsc_vals[-1]

    # Find the interpolation interval using simple index search
    idx = 0
    for i in range(1, len(FSC_RATIOS)):
        if FSC_RATIOS[i] > ratio_pct:
            idx = i
            break

    r0, r1 = FSC_RATIOS[idx - 1], FSC_RATIOS[idx]
    f0, f1 = fsc_vals[idx - 1], fsc_vals[idx]
    
    # Linear interpolation formula
    fsc = f0 + (f1 - f0) * (ratio_pct - r0) / (r1 - r0)
    
    return min(fsc, 0.87 * fy)


# ====================================================================
# *** STREAMLIT APP LAYOUT & CORE LOGIC ***
# ====================================================================

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


# ---------- Title ----------
default_title = "Item 3: Design of Doubly Reinforced Beam (IS 456:2000)"
header_text = st.text_input("Header/Title", value=default_title)
st.title(header_text)

st.markdown("---")

# ---------- Materials & Geometry ----------
st.header("Materials & Geometry (Inputs) 🧱")
st.markdown(r"""
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
st.header(r"Factored Moment $\mathbf{M_u}$ ⚙️")
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

st.info(rf"Limiting Moment $\mathbf{{M_{{u, \lim}}}}$: **{Mu_lim_kNm:.2f} kNm**")

if not is_doubly_required:
    st.success(rf"**$\mathbf{{M_u}}$ ({Mu:.2f} kNm) $\leq$ $\mathbf{{M_{{u, \lim}}}}$ ({Mu_lim_kNm:.2f} kNm). Doubly reinforced section is {OK}. Proceed with singly reinforced design.")
    st.stop() # Stops execution if singly reinforced design is sufficient

st.error(rf"**$\mathbf{{M_u}}$ ({Mu:.2f} kNm) $>$ $\mathbf{{M_{{u, \lim}}}}$ ({Mu_lim_kNm:.2f} kNm). Doubly reinforced section is {NOT_OK}.")
st.markdown("---")

# 3. Splitting the moment
Mu2_kNm = Mu - Mu_lim_kNm
st.header("Moment Components")
c_m1, c_m2 = st.columns(2)
with c_m1:
    st.markdown(rf"Moment resisted by concrete & $\mathbf{{A_{{st1}}}}$ ($\mathbf{{M_{{u, \lim}}}}$): {blue(f'{Mu_lim_kNm:.2f} kNm')}")
with c_m2:
    st.markdown(rf"Remaining moment resisted by $\mathbf{{A_{{sc}}}}$ & $\mathbf{{A_{{st2}}}}$ ($\mathbf{{M_{{u2}}}}$): {red(f'{Mu2_kNm:.2f} kNm')}")

# 4. Stress in Compression Steel (fsc)
d_prime_over_d = d_prime / d
fsc = fsc_calc(fy, d_prime_over_d)

st.header("Compression Steel Stress Check")
label(f"{blue('d\'/d ratio')}: {d_prime_over_d:.3f}")
d_prime_over_xu_max = d_prime_over_d / xu_max_ratio(fy)
# Use double backslashes for the $\mathbf{x_{u,max}}$ LaTeX in f-strings for robustness
label(f"{blue('d\' / x$_{{u,max}}$ ratio')}: {d_prime_over_xu_max:.3f} (Used for $f_{{sc}}$ determination from IS 456 Annex E)")

# Display fsc check result
if fsc >= 0.87 * fy - 1e-3: 
    label(f"{blue('Stress in Compression Steel (fsc)')}: **{fsc:.2f} N/mm²** (Steel is yielding)")
else:
    label(f"{blue('Stress in Compression Steel (fsc)')}: **{fsc:.2f} N/mm²** (Steel is NOT yielding)")


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
st.header("Required Reinforcement Areas (IS 456:2000) 🎯")

c_ast1, c_asc, c_ast2, c_ast_total = st.columns(4)
with c_ast1:
    # Removed \text{M} and switched to $M$
    st.markdown(r"**$\mathbf{A_{st1}}$ (mm²)**: (from $\mathbf{M}_{u, \lim}$)") 
    st.info(f"{Ast1:.2f}")
with c_asc:
    st.markdown(r"**$\mathbf{A_{sc}}$ (mm²)**: (Compression Steel)")
    st.info(f"{Asc:.2f}")
with c_ast2:
    st.markdown(r"**$\mathbf{A_{st2}}$ (mm²)**: (from $\mathbf{M}_{u2}$)")
    st.info(f"{Ast2:.2f}")
with c_ast_total:
    st.markdown(r"**$\mathbf{A_{st, total}}$ (mm²)**: ($\mathbf{A}_{st1} + \mathbf{A}_{st2}$)")
    st.info(f"{Ast_total:.2f}")

# --- Summary ---
st.markdown("---")
st.header("Design Summary 📝")
st.markdown(rf"""
- **Design Factored Moment $\mathbf{{M_u}}$**: {Mu:.2f} kNm
- **Limiting Moment $\mathbf{{M_{{u, \lim}}}}$**: {Mu_lim_kNm:.2f} kNm
- **Required Tension Steel ($\mathbf{{A_{{st, total}}}}$)**: **{Ast_total:.2f} mm²**
- **Required Compression Steel ($\mathbf{{A_{{sc}}}}$)**: **{Asc:.2f} mm²**
- *Note: $\mathbf{{A}}_{{st}}$ must also satisfy minimum and maximum area requirements.*
""")
