import math
import streamlit as st

# Set page configuration for mobile-friendly layout
st.set_page_config(page_title="Item 3 – Doubly Reinforced Beam Design (Full Checks)", layout="wide")

# ====================================================================
# *** CONSTANTS AND DATA TABLES (DEFINED AT THE TOP FOR STABILITY) ***
# ====================================================================

# Styles
BLUE = "#1f6feb"
RED = "#d11a2a"
OK = "<span style='color:#0a8a0a;font-weight:700'>Okay</span>"
NOT_OK = "<span style='color:#c1121f;font-weight=700'>Not OK</span>"
WARNING = "<span style='color:#ffc107;font-weight=700'>Warning</span>"

# IS 456 Annex E Data: d'/xu,max * 100 vs fsc (N/mm²)
FSC_RATIOS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] 
FY415_FSC_VALS = [360.9, 360.9, 351.8, 342.8, 333.7, 324.6, 315.5, 306.4]
FY500_FSC_VALS = [434.8, 434.8, 424.4, 411.3, 395.1, 370.5, 347.5, 324.5]

# IS 456 Table 19: Design Shear Strength of Concrete, τc 
PT_VALS = [0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.50, 2.75, 3.0]
TC_TABLE = {
    20: [0.36, 0.48, 0.56, 0.62, 0.67, 0.71, 0.75, 0.78, 0.81, 0.82, 0.82, 0.82],
    25: [0.37, 0.51, 0.60, 0.67, 0.73, 0.79, 0.84, 0.88, 0.91, 0.94, 0.96, 0.98],
    30: [0.37, 0.53, 0.63, 0.71, 0.78, 0.84, 0.89, 0.93, 0.96, 0.99, 1.01, 1.03],
    35: [0.37, 0.55, 0.66, 0.74, 0.81, 0.88, 0.93, 0.97, 1.01, 1.04, 1.06, 1.08],
    40: [0.37, 0.56, 0.68, 0.76, 0.84, 0.90, 0.96, 1.01, 1.05, 1.08, 1.10, 1.12],
}

# IS 456 Table 20: Maximum Shear Stress (τc,max)
TAU_C_MAX_VALS = {20: 2.8, 25: 3.1, 30: 3.5, 35: 3.7, 40: 4.0}


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
    """Calculates fsc using interpolation from IS 456 Annex E tables."""
    
    if fy == 250: return 0.87 * fy

    if fy == 415: fsc_vals = FY415_FSC_VALS
    elif fy == 500: fsc_vals = FY500_FSC_VALS
    else: return 0.87 * fy

    xu_d_max = xu_max_ratio(fy)
    d_prime_over_xu_max = d_prime_over_d / xu_d_max
    ratio_pct = d_prime_over_xu_max * 100 

    if ratio_pct <= FSC_RATIOS[0]: return fsc_vals[0]
    if ratio_pct >= FSC_RATIOS[-1]: return fsc_vals[-1]

    idx = 0
    for i in range(1, len(FSC_RATIOS)):
        if FSC_RATIOS[i] > ratio_pct:
            idx = i
            break

    r0, r1 = FSC_RATIOS[idx - 1], FSC_RATIOS[idx]
    f0, f1 = fsc_vals[idx - 1], fsc_vals[idx]
    
    fsc = f0 + (f1 - f0) * (ratio_pct - r0) / (r1 - r0)
    return min(fsc, 0.87 * fy)

def calculate_tau_c(fck, pt):
    """Interpolates the design shear strength tau_c from IS 456 Table 19."""
    if fck not in TC_TABLE: return 0.0 

    tc_values = TC_TABLE[fck]
    
    if pt <= PT_VALS[0]: return tc_values[0]
    if pt >= PT_VALS[-1]: return tc_values[-1]

    idx = 0
    for i in range(1, len(PT_VALS)):
        if PT_VALS[i] > pt:
            idx = i
            break
    
    p0, p1 = PT_VALS[idx - 1], PT_VALS[idx]
    t0, t1 = tc_values[idx - 1], tc_values[idx]
    
    tau_c = t0 + (t1 - t0) * (pt - p0) / (p1 - p0)
    return tau_c

# --- CSS for Tighter Mobile/Print Layout ---
st.markdown(r"""
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


# ====================================================================
# *** STREAMLIT APP LAYOUT & CORE LOGIC ***
# ====================================================================

# ---------- Title ----------
default_title = "Item 3: Design of Doubly Reinforced Beam (IS 456:2000) - Full Checks"
header_text = st.text_input("Header/Title", value=default_title)
st.title(header_text)
st.markdown("---")

# ---------- Materials & Geometry ----------
st.header("Materials & Geometry (Inputs) 🧱")
st.markdown(r"""
**📝 NARRATIVE:** Doubly reinforced section is required when $M_u > M_{u, \lim}$.
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
    st.markdown(blue("Width b (mm)"), unsafe_allow_html=True)
    b = st.number_input("b", value=250.0, step=10.0, min_value=100.0, format="%f")
with c4:
    st.markdown(blue("Effective depth d (mm)"), unsafe_allow_html=True)
    d = st.number_input("d", value=500.0, step=10.0, min_value=100.0, format="%f")
with c5:
    st.markdown(blue("Effective cover d' (mm)"), unsafe_allow_html=True)
    d_prime = st.number_input("d_prime", value=50.0, step=5.0, min_value=20.0, format="%f", help="Cover to centroid of Asc")

# ---------- Actions (Factored) ----------
st.markdown("---")
st.header(r"Factored Actions (Inputs) ⚙️")
c_m, c_v, c_l, c_type = st.columns(4)
with c_m:
    st.markdown(blue("Mu (kNm)"), unsafe_allow_html=True)
    Mu = st.number_input("Mu", value=200.0, step=5.0, min_value=0.0, format="%f")
with c_v:
    st.markdown(blue("Vu (kN)"), unsafe_allow_html=True)
    Vu = st.number_input("Vu", value=150.0, step=5.0, min_value=0.0, format="%f")
with c_l:
    st.markdown(blue("Clear Span L (mm)"), unsafe_allow_html=True)
    L = st.number_input("L", value=7500.0, step=100.0, min_value=1000.0, format="%f")
with c_type:
    st.markdown(blue("Support Type"), unsafe_allow_html=True)
    beam_type = st.selectbox("beam_type", ["Simply Supported", "Continuous"], index=1)

# --- CORE DOUBLY REINFORCED CALCULATIONS ---
# 1. Calculate Mu,lim
R_lim_val = R_lim(fck, fy)
Mu_lim_kNm = R_lim_val * b * d**2 / 1e6

# 2. Check requirement and calculate Mu2
is_doubly_required = Mu > Mu_lim_kNm
st.markdown("---")
st.info(rf"Limiting Moment $\\mathbf{{M_{{u, \lim}}}}$: **{Mu_lim_kNm:.2f} kNm**")

if not is_doubly_required:
    # **CRITICAL FIX 1 (SUCCESS):** Wrap the entire formula/text in double dollar signs ($$ $$) 
    # for robust LaTeX rendering, and simplify the Python f-string logic.
    st.success(f"$$M_u ({Mu:.2f}\\,\\text{{kNm}}) \\le M_{{u, \\lim}} ({Mu_lim_kNm:.2f}\\,\\text{{kNm}})$$ Doubly reinforced section is {OK}. Proceed with singly reinforced design.")
    st.stop() 

# **CRITICAL FIX 2 (ERROR):** Wrap the entire formula/text in double dollar signs ($$ $$)
st.error(f"$$M_u ({Mu:.2f}\\,\\text{{kNm}}) > M_{{u, \\lim}} ({Mu_lim_kNm:.2f}\\,\\text{{kNm}})$$ Doubly reinforced section is {NOT_OK}. **(Requires $A_{{sc}}$)**")
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

# 5. Calculate Required Steel Areas (Ast1, Asc, Ast2)
Ast1 = (0.36 * fck * b * xu_max_ratio(fy) * d) / (0.87 * fy)
Asc = (Mu2_kNm * 1e6) / ((fsc - 0.45 * fck) * (d - d_prime))
Ast2 = (Asc * (fsc - 0.45 * fck)) / (0.87 * fy)
Ast_total = Ast1 + Ast2

# --- FINAL RESULTS (CALCULATED) ---
st.markdown("---")
st.header("Required Reinforcement Areas (IS 456:2000) 🎯")
c_ast1, c_asc, c_ast2, c_ast_total = st.columns(4)
with c_ast1:
    st.markdown(r"**$\mathbf{A_{{st1}}}$ (mm²)**: (from $\mathbf{M_{{u, \lim}}}$)") 
    st.info(f"{Ast1:.2f}")
with c_asc:
    st.markdown(r"**$\mathbf{A_{{sc}}}$ (mm²)**: (Compression Steel)")
    st.info(f"{Asc:.2f}")
with c_ast2:
    st.markdown(r"**$\mathbf{A_{{st2}}}$ (mm²)**: (from $\mathbf{M_{{u2}}}$)")
    st.info(f"{Ast2:.2f}")
with c_ast_total:
    st.markdown(r"**$\mathbf{A_{{st, total}}}$ (mm²)**: ($\mathbf{A_{{st1}}} + \mathbf{A_{{st2}}}$)")
    st.info(f"{Ast_total:.2f}")


# ====================================================================
# *** DESIGN CHECKS ***
# ====================================================================
st.markdown("---")
st.header("Design Checks (IS 456:2000) ✅")

# Assume Overall Depth D for Max Steel Check (D = d + d_prime)
D = d + d_prime 

# --- 1. Minimum and Maximum Area Checks ---
st.subheader("1. Area of Steel Limits (Cl 26.5.1)")
c_a1, c_a2, c_a3 = st.columns(3)

# Minimum Tension Steel (Cl 26.5.1.1)
Ast_min = (0.85 * b * d) / fy
with c_a1:
    st.markdown(blue("Min $A_{{st}}$"))
    label(r"**{:.2f} mm²** ($\geq 0.85 \cdot b \cdot d / f_y$)".format(Ast_min))
    result_min = Ast_total >= Ast_min
    st.markdown(f"**Result**: {OK if result_min else NOT_OK}")

# Maximum Tension Steel (Cl 26.5.1.1)
Ast_max = 0.04 * b * D # Use Overall Depth D
with c_a2:
    st.markdown(blue("Max $A_{{st}}$"))
    label(r"**{:.2f} mm²** ($\leq 0.04 \cdot b \cdot D$)".format(Ast_max))
    result_max_t = Ast_total <= Ast_max
    st.markdown(f"**Result**: {OK if result_max_t else NOT_OK}")

# Maximum Compression Steel (Cl 26.5.1.2)
Asc_max = 0.04 * b * D # Use Overall Depth D
with c_a3:
    st.markdown(blue("Max $A_{{sc}}$"))
    label(r"**{:.2f} mm²** ($\leq 0.04 \cdot b \cdot D$)".format(Asc_max))
    result_max_c = Asc <= Asc_max
    st.markdown(f"**Result**: {OK if result_max_c else NOT_OK}")

# --- 2. Shear Design Check ---
st.subheader("2. Shear Check (Cl 40)")
c_s1, c_s2, c_s3 = st.columns(3)

# Nominal Shear Stress (τv)
tau_v = Vu * 1000 / (b * d)
with c_s1:
    st.markdown(blue("Nominal Shear Stress $\\tau_v$"))
    label(f"**{tau_v:.3f} N/mm²** ($V_u / (b \\cdot d)$)")

# Design Shear Strength (τc)
pt = (Ast_total * 100) / (b * d)
tau_c = calculate_tau_c(fck, pt)
with c_s2:
    st.markdown(blue("Design Shear Strength $\\tau_c$ (Table 19)"))
    label(f"For $\mathbf{{p_t={pt:.3f}\%}}$: **{tau_c:.3f} N/mm²**")

# Maximum Shear Stress (τc,max)
tau_c_max = TAU_C_MAX_VALS.get(fck, 2.8)
with c_s3:
    st.markdown(blue("Maximum Shear $\\tau_{{c, max}}$ (Table 20)"))
    label(f"**{tau_c_max:.1f} N/mm²**")
    
    # Check 1: τv vs τc,max
    result_max_shear = tau_v <= tau_c_max
    st.markdown(f"**$\mathbf{{\\tau_v}} \leq \mathbf{{\\tau_{{c, max}}}}$**: {OK if result_max_shear else NOT_OK}")

# Shear Reinforcement Decision (Cl 40.3)
if tau_v <= tau_c:
    st.success(f"**$\mathbf{{\\tau_v}}$ ({tau_v:.3f}) $\leq \mathbf{{\\tau_c}}$ ({tau_c:.3f})**. Nominal shear reinforcement only (Cl 40.3).")
else:
    Vs_req = (tau_v - tau_c) * b * d / 1000
    st.warning(f"**$\mathbf{{\\tau_v}}$ ({tau_v:.3f}) $>\mathbf{{\\tau_c}}$ ({tau_c:.3f})**. Shear reinforcement is **required**. $\mathbf{{V_s}} = \mathbf{{ {Vs_req:.2f} kN}}$")


# --- 3. Deflection Control Check ---
st.subheader("3. Deflection Check (L/d) (Cl 23.2)")
c_d1, c_d2 = st.columns(2)

# Basic L/d ratio (Cl 23.2.1)
basic_ld = 20 if beam_type == "Simply Supported" else 26
with c_d1:
    st.markdown(blue("Basic L/d Ratio (Table 15)"))
    label(f"**{basic_ld}** ({beam_type})")

# Calculate Modification Factor for Tension Steel (Mf_t) - Simplified approach (Fig 4)
fs = 0.58 * fy 
Mf_t = 0.55 + (477 - fs) / (120 * (0.9 + Ast_total * 100 / (b * d))) 
if Mf_t > 2.0: Mf_t = 2.0

# Calculate Modification Factor for Compression Steel (Mf_c) - Simplified approach (Fig 5)
pc = Asc * 100 / (b * d)
Mf_c = 1.0 + (pc / 100) / (0.36 + 20 * d_prime / d) 
if Mf_c > 1.5: Mf_c = 1.5

permitted_ld = basic_ld * Mf_t * Mf_c
actual_ld = L / d

with c_d2:
    st.markdown(blue("Permitted L/d Ratio"))
    # FIX: Corrected using f-string and fully escaped LaTeX braces for subscripts.
    label(f"Basic $\\times M_{{f,t}} \\cdot M_{{f,c}} = {basic_ld:.0f} \\cdot {Mf_t:.2f} \\cdot {Mf_c:.2f} = \\mathbf{{ {permitted_ld:.2f} }}$")
    
# Final Check
result_deflection = actual_ld <= permitted_ld
st.markdown(f"**Actual L/d ({actual_ld:.2f}) $\\mathbf{{\\leq}}$ Permitted L/d ({permitted_ld:.2f})**")
st.markdown(f"**Deflection Check Result**: {OK if result_deflection else NOT_OK}")
