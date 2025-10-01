
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Head 1 – Design & Shear (Single Page v6)", layout="wide")

# ---------- Styles ----------
BLUE = "#1f6feb"
RED = "#d11a2a"
OK = "<span style='color:#0a8a0a;font-weight:700'>Okay</span>"
NOT_OK = "<span style='color:#c1121f;font-weight:700'>Not OK</span>"

def blue(s):  return f"<span style='color:{BLUE};font-weight:600'>{s}</span>"
def red(s):   return f"<span style='color:{RED};font-weight:700'>{s}</span>"
def label(md): st.markdown(md, unsafe_allow_html=True)

BAR_OPTIONS = [8, 10, 12, 16, 20, 25, 32]
STIRRUP_OPTIONS = [6, 8, 10, 12]

# ---------- Helpers ----------
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

def area_bars(diams, nos):
    A = 0.0
    for dia, n in zip(diams, nos):
        if n and dia:
            A += n * (math.pi * (dia ** 2) / 4.0)
    return A

def d_effective_from_layers(D, clear_cover, stirrup_dia, layer_info):
    ys, As = [], []
    offset0 = clear_cover + stirrup_dia
    for idx, (nos, dia) in enumerate(layer_info):
        if nos <= 0 or dia <= 0: continue
        if idx == 0:
            y = offset0 + 0.5*dia
        else:
            prev_dia = layer_info[idx-1][1]
            clear_spacing = max(10.0, 0.5*(dia + prev_dia))
            y = ys[-1] + 0.5*prev_dia + clear_spacing + 0.5*dia
        A = nos * (math.pi * dia**2 / 4.0)
        ys.append(y); As.append(A)
    if not ys: return None
    ycg = sum(A*y for A, y in zip(As, ys)) / sum(As)
    return D - ycg

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

# ---------- Title ----------
default_title = "Design for area of steel and shear for singly reinforced beam by limit state design method"
header_text = st.text_input("Header/Title", value=default_title)
st.title(header_text)

st.markdown("---")

# ---------- Materials ----------
st.header("Materials & Geometry")
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(blue("fy (MPa)"), unsafe_allow_html=True)
    fy_pick = st.selectbox("fy_pick", [415, 500, 550, "Custom"], index=0, label_visibility="collapsed")
    fy = float(fy_pick) if fy_pick != "Custom" else st.number_input("fy", value=415.0, step=1.0, min_value=200.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("fck (MPa)"), unsafe_allow_html=True)
    fck_pick = st.selectbox("fck_pick", [20, 25, 30, 35, 40, "Custom"], index=0, label_visibility="collapsed")
    fck = float(fck_pick) if fck_pick != "Custom" else st.number_input("fck", value=20.0, step=1.0, min_value=15.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("b (mm)"), unsafe_allow_html=True)
    b = st.number_input("b", value=230.0, step=5.0, min_value=100.0, format="%f", label_visibility="collapsed")
with c4:
    st.markdown(blue("Overall depth D (mm)"), unsafe_allow_html=True)
    D = st.number_input("D", value=450.0, step=5.0, min_value=150.0, format="%f", label_visibility="collapsed")
c5,c6,c7,c8 = st.columns(4)
with c5:
    st.markdown(blue("Clear cover (mm)"), unsafe_allow_html=True)
    clear_cover = st.number_input("cover", value=25.0, step=1.0, min_value=0.0, format="%f", label_visibility="collapsed")
with c6:
    st.markdown(blue("Stirrup dia (mm)"), unsafe_allow_html=True)
    phi_sv = st.selectbox("phi_sv", STIRRUP_OPTIONS, index=1, label_visibility="collapsed")
with c7:
    st.markdown(blue("Beam type for span/depth"), unsafe_allow_html=True)
    beam_type = st.selectbox("beam_type", ["Simply Supported","Continuous","Cantilever"], index=1, label_visibility="collapsed")
with c8:
    st.markdown(blue("Span modification factor"), unsafe_allow_html=True)
    mod_factor = st.number_input("mod", value=1.0, step=0.05, min_value=0.7, max_value=1.6, format="%f", label_visibility="collapsed")

st.markdown("---")

# ---------- Actions ----------
st.header("Actions (factored)")
c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(blue("Mu_support (kNm)"), unsafe_allow_html=True)
    Mu_support = st.number_input("Mu_spt", value=20.625, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")
with c2:
    st.markdown(blue("Mu_span (kNm)"), unsafe_allow_html=True)
    Mu_span = st.number_input("Mu_span", value=17.2, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")
with c3:
    st.markdown(blue("Vu (kN)"), unsafe_allow_html=True)
    Vu = st.number_input("Vu", value=64.60, step=0.5, min_value=0.0, format="%f", label_visibility="collapsed")

# Helper to form layers
def layer_elems(n1,d1,n2,d2, n3,d3):
    A1 = area_bars([d1, d3 if n3>0 else 0], [n1, n3 if n3>0 else 0])
    dia1 = max(d1, d3 if n3>0 else d1)
    A2 = area_bars([d2], [n2])
    info = []
    if A1 > 0:
        nos_equiv = max(1, int(round(4*A1/(math.pi*dia1**2))))
        info.append((nos_equiv, dia1))
    if A2 > 0:
        info.append((n2, d2))
    return info, A1+A2

st.markdown("---")

# ---------- Reinforcement – Support ----------
st.header("Reinforcement – Support (tension face)")
c = st.columns(6)
with c[0]: st.markdown(blue("Nos1"), unsafe_allow_html=True); ns1 = st.number_input("ns1", value=2, step=1, min_value=0, label_visibility="collapsed")
with c[1]: st.markdown(blue("dia1 (mm)"), unsafe_allow_html=True); ds1 = st.selectbox("ds1", BAR_OPTIONS, index=2, label_visibility="collapsed")
with c[2]: st.markdown(blue("Nos2"), unsafe_allow_html=True); ns2 = st.number_input("ns2", value=0, step=1, min_value=0, label_visibility="collapsed")
with c[3]: st.markdown(blue("dia2 (mm)"), unsafe_allow_html=True); ds2 = st.selectbox("ds2", BAR_OPTIONS, index=0, label_visibility="collapsed")
with c[4]: st.markdown(blue("Nos3 (same layer1)"), unsafe_allow_html=True); ns3 = st.number_input("ns3", value=0, step=1, min_value=0, label_visibility="collapsed")
with c[5]: st.markdown(blue("dia3 (same layer1)"), unsafe_allow_html=True); ds3 = st.selectbox("ds3", BAR_OPTIONS, index=0, label_visibility="collapsed")

layer_info_spt, Ast_spt_prov = layer_elems(ns1,ds1,ns2,ds2, ns3,ds3)
d_spt = d_effective_from_layers(D, clear_cover, phi_sv, layer_info_spt) or (D - (clear_cover + phi_sv + 0.5*ds1))
st.success(f"Computed d_prov,spt (mm) = {d_spt:.1f}")
Ast_spt, _, _, _ = ast_singly_for_Mu(Mu_support, fck, fy, b, d_spt)
pt_spt = 100.0 * Ast_spt / (b * d_spt) if b*d_spt>0 else float('nan')
label(f"{blue('Reinf.')} {blue('Mu_support')} kNm = {red(f'{Mu_support:.3f}')} &nbsp; "
      f"{blue('Ast req, spt')} mm² = {red(f'{Ast_spt:.2f}')} &nbsp; "
      f"{blue('pt req, spt')} % = {red(f'{pt_spt:.2f}')}" )
label(f"{blue('Ast req vs Prov (mm²)')} = {red(f'{Ast_spt:.2f}')} vs {red(f'{Ast_spt_prov:.2f}')} &nbsp; "
      f"{blue('Result')} = {OK if Ast_spt_prov >= Ast_spt else NOT_OK}" )

st.markdown("---")

# ---------- Reinforcement – Span ----------
st.header("Reinforcement – Span (tension face)")
c2 = st.columns(6)
with c2[0]: st.markdown(blue("Nos1"), unsafe_allow_html=True); nn1 = st.number_input("nn1", value=2, step=1, min_value=0, label_visibility="collapsed")
with c2[1]: st.markdown(blue("dia1 (mm)"), unsafe_allow_html=True); dn1 = st.selectbox("dn1", BAR_OPTIONS, index=2, label_visibility="collapsed")
with c2[2]: st.markdown(blue("Nos2"), unsafe_allow_html=True); nn2 = st.number_input("nn2", value=0, step=1, min_value=0, label_visibility="collapsed")
with c2[3]: st.markdown(blue("dia2 (mm)"), unsafe_allow_html=True); dn2 = st.selectbox("dn2", BAR_OPTIONS, index=0, label_visibility="collapsed")
with c2[4]: st.markdown(blue("Nos3 (same layer1)"), unsafe_allow_html=True); nn3 = st.number_input("nn3", value=0, step=1, min_value=0, label_visibility="collapsed")
with c2[5]: st.markdown(blue("dia3 (same layer1)"), unsafe_allow_html=True); dn3 = st.selectbox("dn3", BAR_OPTIONS, index=0, label_visibility="collapsed")

layer_info_span, Ast_span_prov = layer_elems(nn1,dn1,nn2,dn2, nn3,dn3)
d_span = d_effective_from_layers(D, clear_cover, phi_sv, layer_info_span) or (D - (clear_cover + phi_sv + 0.5*dn1))
st.success(f"Computed d_prov,span (mm) = {d_span:.1f}")
Ast_span, _, _, _ = ast_singly_for_Mu(Mu_span, fck, fy, b, d_span)
pt_span = 100.0 * Ast_span / (b * d_span) if b*d_span>0 else float('nan')
label(f"{blue('Reinf.')} {blue('Mu_span')} kNm = {red(f'{Mu_span:.3f}')} &nbsp; "
      f"{blue('Ast req, span')} mm² = {red(f'{Ast_span:.2f}')} &nbsp; "
      f"{blue('pt req, span')} % = {red(f'{pt_span:.2f}')}" )
label(f"{blue('Ast req vs Prov (mm²)')} = {red(f'{Ast_span:.2f}')} vs {red(f'{Ast_span_prov:.2f}')} &nbsp; "
      f"{blue('Result')} = {OK if Ast_span_prov >= Ast_span else NOT_OK}" )

st.markdown("---")

# ---------- Depth Check ----------
st.header("Depth Check")
xu = xu_max_ratio(fy)
C = 0.36 * fck * b * xu * (1 - 0.42 * xu)
d_req_support = math.sqrt((Mu_support*1e6)/C) if C>0 else float('nan')
d_req_span = math.sqrt((Mu_span*1e6)/C) if C>0 else float('nan')
d_req_ctrl = max(d_req_support, d_req_span)
st.info(f"d_req (mm) = {d_req_ctrl:.2f}")
st.write(f"d_prov,spt = {d_spt:.1f} mm; d_prov,span = {d_span:.1f} mm")
label(f"{blue('Result')} = {OK if (d_spt>=d_req_ctrl and d_span>=d_req_ctrl) else NOT_OK}" )

st.markdown("---")

# ---------- Shear & Stirrups ----------
st.header("Shear & Stirrups")
VuN = Vu * 1e3
d_use = min(d_spt, d_span)
tau_v = VuN / (b * d_use) if b*d_use>0 else float('nan')
p_t_avg = max(0.01, 0.5*((100*Ast_spt/(b*d_spt)) if b*d_spt>0 else 0 + (100*Ast_span/(b*d_span)) if b*d_span>0 else 0))
tau_c = tau_c_interp(p_t_avg, fck)
k = min(2.0, 1.0 + 200.0 / max(1.0, d_use))
tau_c_min = 0.035 * k * (fck ** 0.5)
tau_c_use = max(tau_c, tau_c_min)
tau_cmax = 0.62 * (fck ** 0.5)
Vc = tau_c_use * b * d_use
Vus_req = max(0.0, VuN - Vc)

c = st.columns(4)
with c[0]: st.markdown(blue("Stirrup dia (mm)"), unsafe_allow_html=True); phi_sv_local = st.selectbox("phi_sv_local", STIRRUP_OPTIONS, index=1, label_visibility="collapsed")
with c[1]: st.markdown(blue("Legs"), unsafe_allow_html=True); legs = st.selectbox("legs", [2,4], index=0, label_visibility="collapsed")
with c[2]: st.markdown(blue("fy_stirrup (MPa)"), unsafe_allow_html=True); fy_sv = st.selectbox("fy_sv", [250,415,500,"Use main fy"], index=1, label_visibility="collapsed")
with c[3]: st.markdown(blue("User spacing s (mm)"), unsafe_allow_html=True); s_user = st.number_input("s_user", value=225, min_value=25, step=5, label_visibility="collapsed")

fy_sv_val = float(fy) if fy_sv == "Use main fy" else float(fy_sv)
Asv = legs * (0.25 * math.pi * float(phi_sv_local)**2)
Vus_prov = (0.87 * fy_sv_val * Asv * d_use) / s_user  # N
s_demand = (0.87 * fy_sv_val * Asv * d_use) / Vus_req if Vus_req > 1e-9 else 1e9
s_min_shear = (0.87 * fy_sv_val * Asv) / (0.4 * b)
s_max = min(0.75*d_use, 300.0)
provide_ok = (s_user <= s_max) and (s_user <= s_demand) and (s_user <= s_min_shear)

label(f"{blue('τ_v')} = {red(f'{tau_v:.3f} MPa')} &nbsp; {blue('τ_c,design')} = {red(f'{tau_c_use:.3f} MPa')} &nbsp; {blue('τ_c,max')} = {red(f'{tau_cmax:.3f} MPa')}" )
label(f"{blue('Asv')} = {red(f'{Asv:.1f} mm²')} &nbsp; {blue('Vus req')} = {red(f'{Vus_req/1e3:.1f} kN')} &nbsp; {blue('Vus by s_user')} = {red(f'{Vus_prov/1e3:.1f} kN')}" )
label(f"{blue('s by demand (≤)')} = {red(f'{s_demand:.0f} mm')} &nbsp; {blue('s by min shear (≤)')} = {red(f'{s_min_shear:.0f} mm')} &nbsp; {blue('s by code max (≤)')} = {red(f'{s_max:.0f} mm')}" )
label(f"{blue('User s check')} = {OK if provide_ok else NOT_OK}" )

st.markdown("---")

# ---------- Side-Face ----------
st.header("Side-Face Reinforcement")
need_side = D > 750.0
if need_side:
    As_req_side = 0.001 * b * D  # 0.1% of web area (both faces together)
    label(f"{blue('Required (0.1% of web area)')} = {red(f'{As_req_side:.0f} mm²')} &nbsp; {OK}" )
else:
    label(f"Depth ≤ 750 mm → {blue('side-face steel not required')}" )

st.markdown("---")

# ---------- Span/Depth ----------
st.header("Span/Depth Ratio")
basic = 20 if beam_type == "Simply Supported" else (26 if beam_type == "Continuous" else 7)
permitted = basic * mod_factor
L = st.number_input("Clear span L (mm)", value=5000.0, step=50.0, label_visibility="collapsed")
L_over_d = L / d_span if d_span>0 else float('inf')
label(f"{blue('L/d actual')} = {red(f'{L_over_d:.2f}')} &nbsp; {blue('Permitted')} = {red(f'{permitted:.2f}')} &nbsp; "
      f"{blue('Result')} = {OK if L_over_d <= permitted else NOT_OK}" )

st.markdown("---")

# ---------- Summary ----------
st.header("Summary")
items = []
items.append(("Ast support", f"{Ast_spt:.1f}", f"{Ast_spt_prov:.1f}", Ast_spt_prov >= Ast_spt))
items.append(("Ast span", f"{Ast_span:.1f}", f"{Ast_span_prov:.1f}", Ast_span_prov >= Ast_span))
items.append(("Depth", f"{d_req_ctrl:.1f}", f"min(d_spt,d_span)={min(d_spt,d_span):.1f}", (d_spt>=d_req_ctrl and d_span>=d_req_ctrl)))
items.append(("Stirrups s_user", f"≤ {min(s_demand, s_min_shear, s_max):.0f}", f"{s_user:.0f}", (s_user <= s_max and s_user <= s_demand and s_user <= s_min_shear)))
df = pd.DataFrame([{"Check": n, "Required": r, "Provided": p, "OK": "Yes" if ok else "No"} for (n,r,p,ok) in items])
st.dataframe(df, hide_index=True, use_container_width=True)
overall = all(ok for (_,_,_,ok) in items)
st.success("Overall: PASS ✅" if overall else "Overall: CHECK ❌")
