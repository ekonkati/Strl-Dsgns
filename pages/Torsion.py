import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head 2 – Torsion & Shear Design (Single Page v1)", layout="wide")

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
def red(s):   return f"<span style='color:{RED};font-weight:700'>{s}</span>"
def label(md): st.markdown(md, unsafe_allow_html=True)

STIRRUP_OPTIONS = [6, 8, 10, 12]

# ---------- Helpers (Shear/Torsion Specific) ----------

# Tau_c Table (Table 19, IS 456)
TAU_C_TABLE = pd.DataFrame(
    data=[
        [0.28, 0.29, 0.30, 0.31, 0.31],
        [0.38, 0.40, 0.41, 0.42, 0.42],
        [0.47, 0.50, 0.51, 0.52, 0.52],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
        [0.62, 0.62, 0.62, 0.62, 0.62],
    ],
    index=[0.25,0.50,0.75,1.0,1.25,1.5,1.75,2.0,3.0,4.0],
    columns=[20,25,30,35,40],
)
def tau_c_interp(p_t, fck):
    xs = np.array(TAU_C_TABLE.index, dtype=float)
    ys = np.array(TAU_C_TABLE.columns, dtype=float)
    x = float(np.clip(p_t, xs.min(), xs.max()))
    y = float(np.clip(fck, ys.min(), ys.max()))
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = int(np.clip(i, 0, len(xs)-2))
    j = int(np.clip(j, 0, len(ys)-2))
    x0, x1 = xs[i], xs[i+1];
    y0, y1 = ys[j], ys[j+1]
    q11 = TAU_C_TABLE.iloc[i, j]; q12 = TAU_C_TABLE.iloc[i, j+1]
    q21 = TAU_C_TABLE.iloc[i+1, j]; q22 = TAU_C_TABLE.iloc[i+1, j+1]
    if x1==x0 and y1==y0: return float(q11)
    if x1==x0: return float(q11 + (q12-q11)*(y-y0)/(y1-y0))
    if y1==y0: return float(q11 + (q21-q11)*(x-x0)/(x1-x0))
    return float(q11*(x1-x)*(y1-y)/((x1-x0)*(y1-y0)) +
                 q21*(x-x0)*(y1-y)/((x1-x0)*(y1-y0)) +
                 q12*(x1-x)*(y-y0)/((x1-x0)*(y1-y0)) +
                 q22*(x-x0)*(y-y0)/((x1-x0)*(y1-y0)))

# Max shear stress (Table 20, IS 456)
def tau_cmax(fck):
    if fck <= 20: return 2.8
    if fck == 25: return 3.1
    if fck == 30: return 3.5
    if fck == 35: return 3.7
    return 4.0

# ---------- Title ----------
default_title = "Item 2: Design of Beam Subjected to Combined Torsion and Shear (IS 456:2000 Cl 41)"
header_text = st.text_input("Header/Title", value=default_title)
st.title(header_text)

st.markdown("---")

# ---------- Materials & Geometry ----------
st.header("Materials & Geometry (Inputs)")
st.markdown("""
**📝 NARRATIVE: Input Parameters**
- **$f_{ck}$** is the characteristic strength of the concrete.
- **$b$ and $D$** are the beam's width and overall depth.
- **$d$ (Effective Depth)** is the distance from the compression face to the centroid of the tension reinforcement.
""")
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(blue("fck (MPa)"), unsafe_allow_html=True)
    fck_pick = st.selectbox("fck_pick", [20, 25, 30, 35, 40, "Custom"], index=0, label_visibility="collapsed")
    fck = float(fck_pick) if fck_pick != "Custom" else st.number_input("fck", value=20.0, step=1.0, min_value=15.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("b (mm)"), unsafe_allow_html=True)
    b = st.number_input("b", value=300.0, step=10.0, min_value=100.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("Overall depth D (mm)"), unsafe_allow_html=True)
    D = st.number_input("D", value=600.0, step=10.0, min_value=150.0, format="%f", label_visibility="collapsed")
with c4:
    st.markdown(blue("Effective depth d (mm)"), unsafe_allow_html=True)
    d = st.number_input("d", value=550.0, step=5.0, min_value=50.0, format="%f", help="Based on tension steel c.g.", label_visibility="collapsed")

st.markdown("---")

# ---------- Actions (Factored) ----------
st.header("Actions (Factored $M_u, V_u, T_u$)")
st.markdown("""
**📝 NARRATIVE: Design Actions**
- **$M_u$**: Factored Bending Moment.
- **$V_u$**: Factored Shear Force.
- **$T_u$**: Factored Torsional Moment.
""")
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(blue("Mu (kNm)"), unsafe_allow_html=True)
    Mu = st.number_input("Mu", value=181.0, step=5.0, min_value=0.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("Vu (kN)"), unsafe_allow_html=True)
    Vu = st.number_input("Vu", value=110.0, step=5.0, min_value=0.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("Tu (kNm)"), unsafe_allow_html=True)
    Tu = st.number_input("Tu", value=75.0, step=5.0, min_value=0.0, format="%f", label_visibility="collapsed")


# --- CORE TORSION CALCULATIONS ---

# 1. Equivalent Moment Mt (Clause 41.4.2)
Mt = Tu * (1 + (D/b)) / 1.7
Me1 = Mu + Mt
Me2 = Mu - Mt

# 2. Equivalent Shear Ve (Clause 41.3.1)
Ve = Vu + 1.6 * (Tu * 1e6) / b / 1e3 # Vu + 1.6 * Tu/b (units conversion: Tu*1e6/1e3 converts kNm to N-mm, then to kN-mm/mm/1e3 to get kN)

# 3. Nominal Shear Stress Tve (Clause 41.3.1)
Tve = Ve * 1e3 / (b * d) # N/mm2

st.markdown("---")

# ---------- Equivalent Forces Results ----------
st.header("Equivalent Design Forces (IS 456:2000 Cl 41)")
st.markdown("""
The applied forces are converted into equivalent design forces ($\mathbf{M_{e1}, M_{e2}}$ and $\mathbf{V_e}$) for use in standard flexural and shear design methods.
""")

label(f"{blue('Torsional Moment (Mt)')} = $T_u(1 + D/b)/1.7$")
label(f"{blue('Equivalent Moment (Me1, tension)')} = $M_u + M_t$")
label(f"{blue('Equivalent Moment (Me2, compression)')} = $M_u - M_t$")

c_eq = st.columns(4)
with c_eq[0]: st.info(f"**$\mathbf{{M_t}}$ (kNm)**: {Mt:.2f}")
with c_eq[1]: st.info(f"**$\mathbf{{M_{{e1}}}}$ (kNm)**: {Me1:.2f}")
with c_eq[2]: st.info(f"**$\mathbf{{M_{{e2}}}}$ (kNm)**: {Me2:.2f}")

label(f"{blue('Equivalent Shear (Ve)')} = $V_u + 1.6 T_u/b$")
label(f"{blue('Nominal Equivalent Shear Stress (τve)')} = $V_e / (b \cdot d)$")

c_ve = st.columns(4)
with c_ve[0]: st.info(f"**$\mathbf{{V_e}}$ (kN)**: {Ve:.2f}")
with c_ve[1]: st.info(f"**$\mathbf{{\\tau_{{ve}}}}$ ($\mathbf{{N/mm^2}}$)**: {Tve:.3f}")

# --- Shear Strength Check (Cl 41.3.2) ---
Tc_max = tau_cmax(fck)
shear_check_ok = Tve <= Tc_max
with c_ve[2]: st.info(f"**$\mathbf{{\\tau_{{c,max}}}}$ ($\mathbf{{N/mm^2}}$)**: {Tc_max:.3f}")
with c_ve[3]: 
    if shear_check_ok:
        st.success("Torsion Check: OK")
    else:
        st.error(f"Torsion Check: FAIL (Increase section size as $\\tau_{{ve}} > \\tau_{{c,max}}$)")

st.markdown("---")

# ---------- Transverse Shear Design (Stirrups) ----------
st.header("Transverse Shear Design (Based on $V_e$)")
st.markdown("""
**📝 NARRATIVE: Stirrup Requirement**
The required shear reinforcement ($\text{A}_{sv}$ spacing) is determined using the **Equivalent Shear, $\text{V}_e$**. The actual tension steel ratio ($\text{p}_t$) based on $\text{M}_{e1}$ is required to find the concrete shear capacity ($\tau_c$).
""")

c_pt, c_phi, c_fy = st.columns(3)
with c_pt:
    st.markdown(blue("pt (Tension Steel Ratio % prov.)"), unsafe_allow_html=True)
    pt_prov = st.number_input("pt_prov", value=0.5, step=0.05, min_value=0.0, format="%f", help="Based on Me1 design", label_visibility="collapsed")
with c_phi:
    st.markdown(blue("Stirrup dia (mm)"), unsafe_allow_html=True)
    phi_sv_local = st.selectbox("phi_sv_local", STIRRUP_OPTIONS, index=1, label_visibility="collapsed")
with c_fy:
    st.markdown(blue("Stirrup fy (MPa)"), unsafe_allow_html=True)
    fy_sv_val = st.number_input("fy_sv_val", value=415.0, step=5.0, min_value=250.0, format="%f", label_visibility="collapsed")

# 1. Concrete Shear Capacity (Vc)
pt_use = max(0.01, pt_prov)
tau_c = tau_c_interp(pt_use, fck)
k = min(2.0, 1.0 + 200.0 / d)
tau_c_min = 0.035 * k * (fck ** 0.5)
tau_c_use = max(tau_c, tau_c_min)
Vc = tau_c_use * b * d / 1e3 # kN

# 2. Shear Resistance required from Stirrups (Vus)
Vus_req = max(0.0, Ve - Vc) # kN

st.success(f"Concrete Shear Capacity ($\mathbf{{V_c}}$) = **{Vc:.2f} kN**")
st.error(f"Required Stirrup Resistance ($\mathbf{{V_{{us,req}}}}$) = **{Vus_req:.2f} kN**")

st.markdown("#### Stirrup Spacing Check")
c_stir, c_s_user = st.columns(2)
with c_stir:
    st.markdown(blue("Legs"), unsafe_allow_html=True)
    legs = st.selectbox("legs", [2,4], index=0, label_visibility="collapsed")
with c_s_user:
    st.markdown(blue("User spacing s (mm)"), unsafe_allow_html=True)
    s_user = st.number_input("s_user", value=150, min_value=25, step=5, label_visibility="collapsed")

# ----------------------------------------------------
# MOVED: Longitudinal Bar Configuration (must come BEFORE s_max calculation)
# ----------------------------------------------------
st.markdown("##### Longitudinal Bar Configuration (for Spacing Check)")
c_dim = st.columns(2)
with c_dim[0]:
    st.markdown(blue("$x_1$ (c/c horizontal dim, mm)"), unsafe_allow_html=True)
    x1 = st.number_input("x1", value=240.0, step=5.0, min_value=1.0, format="%f", help="Centre-to-centre distance of the outermost longitudinal bars in the direction of width 'b'", label_visibility="collapsed")
with c_dim[1]:
    st.markdown(blue("$y_1$ (c/c vertical dim, mm)"), unsafe_allow_html=True)
    y1 = st.number_input("y1", value=540.0, step=5.0, min_value=1.0, format="%f", help="Centre-to-centre distance of the outermost longitudinal bars in the direction of depth 'D'", label_visibility="collapsed")
# ----------------------------------------------------

Asv = legs * (0.25 * math.pi * float(phi_sv_local)**2)
Vus_user_N = Vus_req * 1e3 # Convert Vus_req to N

# Calculate required spacing based on Vus_req
s_demand = (0.87 * fy_sv_val * Asv * d) / Vus_user_N if Vus_user_N > 1e-9 else 1e9

# Check Spacing Limits (Cl 41.4.3)
s_max_41 = min(x1, y1) / 4.0 + (D / 10.0)
s_max_code_cl_26 = min(0.75 * d, 300.0)
s_max_ctrl = min(s_max_41, s_max_code_cl_26, x1) # Cl 41.4.3 (c) limits spacing to min(s_max_41, x1, 300)

provide_ok = (s_user <= s_demand) and (s_user <= s_max_ctrl)

label(f"{blue('s by Vus demand (≤)')} = {red(f'{s_demand:.0f} mm')} &nbsp; "
      f"{blue('s by Cl 41.4.3 (≤)')} = {red(f'{s_max_ctrl:.0f} mm')}" )
label(f"{blue('User s check')} = {OK if provide_ok else NOT_OK}" )

# --- Summary for Torsion App ---
st.markdown("---")
st.header("Summary (Torsion Design)")
items = []
items.append(("Equivalent Moment Me1", f"{Me1:.2f} kNm", "N/A", True))
items.append(("Equivalent Moment Me2", f"{Me2:.2f} kNm", "N/A", True))
items.append(("Equivalent Shear Ve", f"{Ve:.2f} kN", f"{Vc:.2f} kN", Ve < Vc))
items.append(("Torsion Max Stress Check", f"Max {Tc_max:.3f} N/mm²", f"Actual {Tve:.3f} N/mm²", shear_check_ok))
items.append(("Stirrups Spacing", f"≤ {min(s_demand, s_max_ctrl):.0f} mm", f"{s_user:.0f} mm", provide_ok))

df = pd.DataFrame([{"Check": n, "Required/Limit": r, "Provided/Actual": p, "OK": "Yes" if ok else "No"} for (n,r,p,ok) in items])
st.dataframe(df, hide_index=True, use_container_width=True)
overall = all(ok for (_,_,_,ok) in items if n != "Equivalent Moment Me1" and n != "Equivalent Moment Me2")
st.success("Overall: PASS ✅" if overall else "Overall: CHECK ❌")
