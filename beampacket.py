
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head 1 – Vertical Packet UI", layout="wide")

# ---------- Styles (blue inputs / red outputs) ----------
BLUE = "#1f6feb"
RED = "#d11a2a"

def blue(s):  return f"<span style='color:{BLUE};font-weight:600'>{s}</span>"
def red(s):   return f"<span style='color:{RED};font-weight:700'>{s}</span>"
def hsub(title): st.markdown(f"### {title}")

# ---------- Design helpers ----------
def xu_max_ratio(fy):
    if fy <= 260: return 0.53
    if fy <= 450: return 0.48
    return 0.46

def mu_lim_kNm(fck, fy, b, d):
    xu = xu_max_ratio(fy) * d
    return 0.36 * fck * b * xu * (d - 0.42 * xu) / 1e6  # kNm

def ast_singly_for_Mu(Mu_kNm, fck, fy, b, d):
    Mu = Mu_kNm * 1e6
    jd = 0.9 * d
    Ast = Mu / (0.87 * fy * jd) if jd > 0 else float('nan')
    if Ast <= 0 or math.isnan(Ast):
        return float('nan'), float('nan'), float('nan'), mu_lim_kNm(fck, fy, b, d)
    xu_try = (0.87 * fy * Ast) / (0.36 * fck * b)
    jd = d * (1 - 0.42 * (xu_try / d))
    if jd <= 0: jd = 0.9 * d
    Ast = Mu / (0.87 * fy * jd)
    return Ast, (xu_try/d), jd, mu_lim_kNm(fck, fy, b, d)

def d_required_for_Mu_lim(Mu_kNm, fck, fy, b):
    xu_max = xu_max_ratio(fy)
    C = 0.36 * fck * b * xu_max * (1 - 0.42 * xu_max)
    if C <= 0: return float('nan')
    Mu = Mu_kNm * 1e6
    return math.sqrt(Mu / C)

# tau_c table (representative IS 456 Table 19 values).
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

def bilinear_interp(df, x, y):
    xs = np.array(df.index, dtype=float)
    ys = np.array(df.columns, dtype=float)
    x = float(np.clip(x, xs.min(), xs.max()))
    y = float(np.clip(y, ys.min(), ys.max()))
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = int(np.clip(i, 0, len(xs)-2))
    j = int(np.clip(j, 0, len(ys)-2))
    x0, x1 = xs[i], xs[i+1]
    y0, y1 = ys[j], ys[j+1]
    q11 = df.iloc[i, j]
    q12 = df.iloc[i, j+1]
    q21 = df.iloc[i+1, j]
    q22 = df.iloc[i+1, j+1]
    if x1==x0 and y1==y0:
        return float(q11)
    if x1==x0:
        return float(q11 + (q12-q11)*(y-y0)/(y1-y0))
    if y1==y0:
        return float(q11 + (q21-q11)*(x-x0)/(x1-x0))
    return float(
        q11*(x1-x)*(y1-y)/((x1-x0)*(y1-y0)) +
        q21*(x-x0)*(y1-y)/((x1-x0)*(y1-y0)) +
        q12*(x1-x)*(y-y0)/((x1-x0)*(y1-y0)) +
        q22*(x-x0)*(y-y0)/((x1-x0)*(y1-y0))
    )

def area_bars(diams, nos):
    A = 0.0
    for dia, n in zip(diams, nos):
        if n and dia:
            A += n * (math.pi * (dia ** 2) / 4.0)
    return A

# ---------- PAGE ----------
st.title("Head 1 – Vertical Subhead Layout")

# Subhead 1: Materials & Geometry
hsub("Materials & Geometry")
c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
with c1:
    st.markdown(blue("fy (MPa)"), unsafe_allow_html=True)
    fy = st.number_input("fy (MPa)", value=415.0, step=1.0, min_value=200.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("fck (MPa)"), unsafe_allow_html=True)
    fck = st.number_input("fck (MPa)", value=20.0, step=1.0, min_value=15.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("b (mm)"), unsafe_allow_html=True)
    b = st.number_input("b (mm)", value=230.0, step=10.0, min_value=100.0, format="%f", label_visibility="collapsed")
with c4:
    st.markdown(blue("d_prov (mm)"), unsafe_allow_html=True)
    d = st.number_input("d_prov (mm)", value=319.0, step=5.0, min_value=100.0, format="%f", label_visibility="collapsed")
with c5:
    st.markdown(blue("cover (mm)"), unsafe_allow_html=True)
    cover = st.number_input("cover (mm)", value=25.0, step=1.0, min_value=0.0, format="%f", label_visibility="collapsed")
with c6:
    st.markdown(blue("main bar dia (mm)"), unsafe_allow_html=True)
    phi_main = st.number_input("main bar dia (mm)", value=12.0, step=1.0, min_value=6.0, format="%f", label_visibility="collapsed")
with c7:
    st.markdown(blue("clear span L (mm)"), unsafe_allow_html=True)
    Lspan = st.number_input("clear span L (mm)", value=5000.0, step=50.0, min_value=500.0, format="%f", label_visibility="collapsed")

st.markdown("---")

# Subhead 2: Actions/Design inputs
hsub("Actions (factored)")
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(blue("Mu_support (kNm)"), unsafe_allow_html=True)
    Mu_support = st.number_input("Mu_support (kNm)", value=20.625, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("Mu_span (kNm)"), unsafe_allow_html=True)
    Mu_span = st.number_input("Mu_span (kNm)", value=17.2, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("Vu (kN)"), unsafe_allow_html=True)
    Vu = st.number_input("Vu (kN)", value=64.60, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")

st.markdown("---")

# Subhead 3: Required steel (Support & Span)
hsub("Required Steel – Support")
Ast_spt, xu_spt_ratio, jd_spt, Mu_lim = ast_singly_for_Mu(Mu_support, fck, fy, b, d)
pt_spt = 100.0 * Ast_spt / (b * d) if b*d>0 else float('nan')
st.markdown(
    f"{blue('Mu_support')} kNm = {red(f'{Mu_support:.3f}')} &nbsp;&nbsp; "
    f"{blue('Ast req, spt')} mm² = {red(f'{Ast_spt:.2f}')} &nbsp;&nbsp; "
    f"{blue('pt req, spt')} % = {red(f'{pt_spt:.2f}')}",
    unsafe_allow_html=True
)

hsub("Required Steel – Span")
Ast_span, xu_span_ratio, jd_span, _ = ast_singly_for_Mu(Mu_span, fck, fy, b, d)
pt_span = 100.0 * Ast_span / (b * d) if b*d>0 else float('nan')
st.markdown(
    f"{blue('Mu_span')} kNm = {red(f'{Mu_span:.3f}')} &nbsp;&nbsp; "
    f"{blue('Ast span')} mm² = {red(f'{Ast_span:.2f}')} &nbsp;&nbsp; "
    f"{blue('pt req, span')} % = {red(f'{pt_span:.2f}')}",
    unsafe_allow_html=True
)

st.markdown("---")

# Subhead 4: Depth check
hsub("Check for Depth")
d_req_ctrl = max(d_required_for_Mu_lim(Mu_support, fck, fy, b),
                 d_required_for_Mu_lim(Mu_span, fck, fy, b))
depth_ok = d >= d_req_ctrl
st.markdown(
    f"{blue('d req mm')} = {red(f'{d_req_ctrl:.2f}')} &nbsp;&nbsp; "
    f"{blue('d prov mm')} = {red(f'{d:.0f}')} &nbsp;&nbsp; "
    f"{blue('Result')} = {red('Okay' if depth_ok else 'Not OK')}",
    unsafe_allow_html=True
)

st.markdown("---")

# Subhead 5: Reinforcement details provided (support & span)
hsub("Reinf. details at support")
c = st.columns(6)
with c[0]:
    st.markdown(blue("Nos1"), unsafe_allow_html=True)
    ns1 = st.number_input("Nos1", value=2, step=1, min_value=0, label_visibility="collapsed")
with c[1]:
    st.markdown(blue("dia1 (mm)"), unsafe_allow_html=True)
    ds1 = st.number_input("dia1 (mm)", value=12, step=1, min_value=0, label_visibility="collapsed")
with c[2]:
    st.markdown(blue("Nos2"), unsafe_allow_html=True)
    ns2 = st.number_input("Nos2", value=0, step=1, min_value=0, label_visibility="collapsed")
with c[3]:
    st.markdown(blue("dia2 (mm)"), unsafe_allow_html=True)
    ds2 = st.number_input("dia2 (mm)", value=0, step=1, min_value=0, label_visibility="collapsed")
with c[4]:
    st.markdown(blue("Nos3"), unsafe_allow_html=True)
    ns3 = st.number_input("Nos3", value=0, step=1, min_value=0, label_visibility="collapsed")
with c[5]:
    st.markdown(blue("dia3 (mm)"), unsafe_allow_html=True)
    ds3 = st.number_input("dia3 (mm)", value=0, step=1, min_value=0, label_visibility="collapsed")

Ast_spt_prov = area_bars([ds1, ds2, ds3], [ns1, ns2, ns3])
pt_spt_prov = 100.0 * Ast_spt_prov / (b * d) if b*d>0 else float('nan')
spt_ok = Ast_spt_prov >= Ast_spt
st.markdown(
    f"{blue('Nos.')} = {red(f'{ns1}+{ns2}+{ns3}')} &nbsp;&nbsp; "
    f"{blue('dia')} mm = {red(f'{ds1}/{ds2}/{ds3}')} &nbsp;&nbsp; "
    f"{blue('Ast support')} mm² = {red(f'{Ast_spt_prov:.2f}')} &nbsp;&nbsp; "
    f"{blue('pt support')} % = {red(f'{pt_spt_prov:.2f}')} &nbsp;&nbsp; "
    f"{blue('Result')} = {red('okay' if spt_ok else 'not okay')}",
    unsafe_allow_html=True
)

hsub("Reinf. details at span")
c2 = st.columns(6)
with c2[0]:
    st.markdown(blue("Nos1"), unsafe_allow_html=True)
    nn1 = st.number_input("Nos1 ", value=2, step=1, min_value=0, key="n1", label_visibility="collapsed")
with c2[1]:
    st.markdown(blue("dia1 (mm)"), unsafe_allow_html=True)
    dn1 = st.number_input("dia1 (mm) ", value=12, step=1, min_value=0, key="d1", label_visibility="collapsed")
with c2[2]:
    st.markdown(blue("Nos2"), unsafe_allow_html=True)
    nn2 = st.number_input("Nos2 ", value=0, step=1, min_value=0, key="n2", label_visibility="collapsed")
with c2[3]:
    st.markdown(blue("dia2 (mm)"), unsafe_allow_html=True)
    dn2 = st.number_input("dia2 (mm) ", value=0, step=1, min_value=0, key="d2", label_visibility="collapsed")
with c2[4]:
    st.markdown(blue("Nos3"), unsafe_allow_html=True)
    nn3 = st.number_input("Nos3 ", value=0, step=1, min_value=0, key="n3", label_visibility="collapsed")
with c2[5]:
    st.markdown(blue("dia3 (mm)"), unsafe_allow_html=True)
    dn3 = st.number_input("dia3 (mm) ", value=0, step=1, min_value=0, key="d3", label_visibility="collapsed")

Ast_span_prov = area_bars([dn1, dn2, dn3], [nn1, nn2, nn3])
pt_span_prov = 100.0 * Ast_span_prov / (b * d) if b*d>0 else float('nan')
span_ok = Ast_span_prov >= Ast_span
st.markdown(
    f"{blue('Nos.')} = {red(f'{nn1}+{nn2}+{nn3}')} &nbsp;&nbsp; "
    f"{blue('dia')} mm = {red(f'{dn1}/{dn2}/{dn3}')} &nbsp;&nbsp; "
    f"{blue('Ast span')} mm² = {red(f'{Ast_span_prov:.2f}')} &nbsp;&nbsp; "
    f"{blue('pt span')} % = {red(f'{pt_span_prov:.2f}')} &nbsp;&nbsp; "
    f"{blue('Result')} = {red('okay' if span_ok else 'not okay')}",
    unsafe_allow_html=True
)

st.markdown("---")

# Subhead 6: Shear Check and Stirrups
hsub("Shear Check and Stirrups")
c = st.columns(5)
with c[0]:
    st.markdown(blue("stirrup dia (mm)"), unsafe_allow_html=True)
    phi_sv = st.number_input("stirrup dia (mm)", value=8.0, step=1.0, min_value=6.0, format="%f", label_visibility="collapsed")
with c[1]:
    st.markdown(blue("legs"), unsafe_allow_html=True)
    legs = st.selectbox("legs", [2,4], index=0, label_visibility="collapsed")
with c[2]:
    st.markdown(blue("fy_stirrup (MPa)"), unsafe_allow_html=True)
    fy_sv = st.number_input("fy_stirrup (MPa)", value=415.0, step=1.0, min_value=200.0, format="%f", label_visibility="collapsed")
with c[3]:
    st.markdown(blue("web cover (mm)"), unsafe_allow_html=True)
    cover_sv = st.number_input("web cover (mm)", value=25.0, step=1.0, min_value=0.0, format="%f", label_visibility="collapsed")
with c[4]:
    note_toggle = st.checkbox("Show note", value=True)

VuN = Vu * 1e3
tau_v = VuN / (b * d) if b*d>0 else float('nan')  # MPa
pt_use = max(0.01, (pt_spt + pt_span)/2.0)  # rough % tension
# Interpolated tau_c
def _bilin(x, y):
    xs = np.array(TAU_C_TABLE.index, dtype=float)
    ys = np.array(TAU_C_TABLE.columns, dtype=float)
    x = float(np.clip(x, xs.min(), xs.max()))
    y = float(np.clip(y, ys.min(), ys.max()))
    i = np.searchsorted(xs, x) - 1
    j = np.searchsorted(ys, y) - 1
    i = int(np.clip(i, 0, len(xs)-2))
    j = int(np.clip(j, 0, len(ys)-2))
    x0, x1 = xs[i], xs[i+1]; y0, y1 = ys[j], ys[j+1]
    q11 = TAU_C_TABLE.iloc[i, j]; q12 = TAU_C_TABLE.iloc[i, j+1]
    q21 = TAU_C_TABLE.iloc[i+1, j]; q22 = TAU_C_TABLE.iloc[i+1, j+1]
    if x1==x0 and y1==y0: return float(q11)
    if x1==x0: return float(q11 + (q12-q11)*(y-y0)/(y1-y0))
    if y1==y0: return float(q11 + (q21-q11)*(x-x0)/(x1-x0))
    return float(q11*(x1-x)*(y1-y)/((x1-x0)*(y1-y0)) +
                 q21*(x-x0)*(y1-y)/((x1-x0)*(y1-y0)) +
                 q12*(x1-x)*(y-y0)/((x1-x0)*(y1-y0)) +
                 q22*(x-x0)*(y-y0)/((x1-x0)*(y1-y0)))
try:
    tau_c = _bilin(pt_use, fck)
except Exception:
    tau_c = TAU_C_TABLE.iloc[-1,-1]
k = min(2.0, 1.0 + 200.0 / max(1.0, d))
tau_c_min = 0.035 * k * (fck ** 0.5)
tau_c_use = max(tau_c, tau_c_min)
tau_cmax = 0.62 * (fck ** 0.5)  # IS 456 limit

st.markdown(
    f"{blue('τ_v')} MPa = {red(f'{tau_v:.3f}')} &nbsp;&nbsp; "
    f"{blue('τ_c,design')} MPa = {red(f'{tau_c_use:.3f}')} &nbsp;&nbsp; "
    f"{blue('τ_c,max')} MPa = {red(f'{tau_cmax:.3f}')}",
    unsafe_allow_html=True
)

if note_toggle:
    if tau_v > tau_c_use and tau_v < tau_cmax:
        st.markdown(red("τ_v > τ_c,design for shear; τ_v < τ_c,max, Ok – provide shear steel"), unsafe_allow_html=True)
    elif tau_v <= tau_c_use:
        st.markdown(red("τ_v ≤ τ_c,design – minimum shear steel only"), unsafe_allow_html=True)
    else:
        st.markdown(red("τ_v ≥ τ_c,max – Increase section or grade"), unsafe_allow_html=True)

Vc = tau_c_use * b * d
Vus = max(0.0, VuN - Vc)
Asv = legs * (0.25 * math.pi * phi_sv**2)
if Vus > 0 and Asv > 0:
    s = (0.87 * fy_sv * Asv * d) / Vus  # mm
    s_lim = min(0.75*d, 300.0)
    s_use = min(s, s_lim)
    st.markdown(
        f"{blue('Design s')} mm = {red(f'{s:.0f}')} &nbsp;&nbsp; "
        f"{blue('Provide')} = {red(f'{int(s_use)} mm c/c (≤ {int(s_lim)} mm)')}",
        unsafe_allow_html=True
    )
else:
    st.markdown(red("Provide minimum shear reinforcement as per IS 456."), unsafe_allow_html=True)

st.markdown("---")

# Subhead 7: Summary
overall = all([ (d >= d_req_ctrl), (Ast_spt_prov >= Ast_spt), (Ast_span_prov >= Ast_span),
                (Mu_support <= Mu_lim and Mu_span <= Mu_lim) ])
st.success("Overall: PASS ✅" if overall else "Overall: CHECK ❌")
