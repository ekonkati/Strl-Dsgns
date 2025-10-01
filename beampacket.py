import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Head 1 – RCC Beam Check (ANNEX A logic)", layout="wide")

# ---------- Helpers ----------
def xu_max_ratio(fy):
    if fy <= 260: return 0.53
    if fy <= 450: return 0.48
    return 0.46

def mu_lim_kNm(fck, fy, b, d):
    xu = xu_max_ratio(fy) * d
    # Mu_lim = 0.36 fck b xu (d - 0.42 xu), units N*mm -> kNm
    return 0.36 * fck * b * xu * (d - 0.42 * xu) / 1e6

def ast_singly_for_Mu(Mu_kNm, fck, fy, b, d):
    """Compute Ast for a given Mu using an iterative jd. Returns Ast, xu/d, jd, Mu_lim."""
    Mu = Mu_kNm * 1e6
    xu_ratio = xu_max_ratio(fy)
    Mu_lim = mu_lim_kNm(fck, fy, b, d)
    # Start with jd ~ 0.9 d, update once via xu from equilibrium
    jd = 0.9 * d
    Ast = Mu / (0.87 * fy * jd) if jd > 0 else float('nan')
    if Ast <= 0 or math.isnan(Ast):
        return float('nan'), float('nan'), float('nan'), Mu_lim
    xu_try = (0.87 * fy * Ast) / (0.36 * fck * b)
    jd = d * (1 - 0.42 * (xu_try / d))
    if jd <= 0: jd = 0.9 * d
    Ast = Mu / (0.87 * fy * jd)
    return Ast, (xu_try/d), jd, Mu_lim

def d_required_for_Mu_lim(Mu_kNm, fck, fy, b):
    """Solve Mu_lim(d) >= Mu for minimum d. Closed-form from Mu_lim = C * d^2."""
    # Mu_lim = 0.36 * fck * b * (xu_max*d) * (d - 0.42*xu_max*d)
    #        = 0.36 * fck * b * d^2 * xu_max * (1 - 0.42*xu_max) = C * d^2
    xu_max = xu_max_ratio(fy)
    C = 0.36 * fck * b * xu_max * (1 - 0.42 * xu_max)  # N/mm^2 * mm -> N
    if C <= 0: return float('nan')
    Mu = Mu_kNm * 1e6  # Nmm
    d_req = math.sqrt(Mu / C)
    return d_req

def area_bars(diams, nos):
    A = 0.0
    for dia, n in zip(diams, nos):
        if n and dia:
            A += n * (math.pi * (dia ** 2) / 4.0)
    return A

def packet_card(title, rows):
    """rows: list of (label, unit, value_str)"""
    with st.container(border=True):
        st.markdown(f"**{title}**")
        df = pd.DataFrame([
            {"Item": lbl, "Unit": unit, "Value": val} for (lbl, unit, val) in rows
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)

# ---------- UI ----------
st.title("Head 1 – Design & Check (Support + Span + Depth + Rebar)")
with st.sidebar:
    st.header("Materials")
    fck = st.selectbox("Concrete fck (MPa)", [20, 25, 30, 35, 40], index=1)
    fy  = st.selectbox("Steel fy (MPa)", [250, 415, 500], index=1)
    fy_sv = st.selectbox("Stirrup fy (MPa)", [250, 415, 500], index=1)
    st.markdown("---")
    st.header("Section")
    b = st.number_input("Width b (mm)", min_value=100.0, value=230.0, step=10.0)
    d_prov = st.number_input("Effective depth d_prov (mm)", min_value=100.0, value=319.0, step=5.0)
    st.caption("Use your actual effective depth (overall depth minus cover & half bar dia).")


st.subheader("Actions (factored)")
c1, c2, c3 = st.columns(3)
with c1:
    Mu_support = st.number_input("Mu_support (kNm)", min_value=0.0, value=20.625, step=0.5, format="%f")
with c2:
    Mu_span = st.number_input("Mu_span (kNm)", min_value=0.0, value=17.2, step=0.5, format="%f")
with c3:
    Vu = st.number_input("Vu (kN) [optional for shear]", min_value=0.0, value=0.0, step=1.0)

st.markdown("---")
st.subheader("Flexure – Required steel")
Ast_spt, xu_spt_ratio, jd_spt, Mu_lim = ast_singly_for_Mu(Mu_support, fck, fy, b, d_prov)
Ast_span, xu_span_ratio, jd_span, _ = ast_singly_for_Mu(Mu_span, fck, fy, b, d_prov)

pt_spt = 100.0 * Ast_spt / (b * d_prov) if b*d_prov>0 else float('nan')
pt_span = 100.0 * Ast_span / (b * d_prov) if b*d_prov>0 else float('nan')
status_mu = "ok" if (Mu_support <= Mu_lim and Mu_span <= Mu_lim) else "exceeds"

# Packet: Required (support & span)
packet_card("Required at support", [
    ("Mu_support", "kNm", f"{Mu_support:.3f}"),
    ("Ast req, spt", "mm²", f"{Ast_spt:.2f}"),
    ("pt req, spt", "%", f"{pt_spt:.2f}"),
])
packet_card("Required at span", [
    ("Mu_span", "kNm", f"{Mu_span:.3f}"),
    ("Ast span", "mm²", f"{Ast_span:.2f}"),
    ("pt req, span", "%", f"{pt_span:.2f}"),
])

st.markdown("---")
st.subheader("Depth check (from Mu_lim)")
d_req_ctrl = max(d_required_for_Mu_lim(Mu_support, fck, fy, b),
                 d_required_for_Mu_lim(Mu_span, fck, fy, b))
res_depth = "okay" if d_prov >= d_req_ctrl else "not okay"
packet_card("check for depth", [
    ("d req mm", "mm", f"{d_req_ctrl:.2f}"),
    ("d prov mm", "mm", f"{d_prov:.0f}"),
    ("Result", "", res_depth),
])

st.markdown("---")
st.subheader("Reinforcement details provided – compare with required")
st.caption("Enter up to 3 bar groups for support and span (Nos & dia). Leave zeros if not used.")
# Support bars
cols = st.columns(6)
with cols[0]:
    ns1 = st.number_input("spt Nos1", min_value=0, value=2, step=1)
with cols[1]:
    ds1 = st.number_input("spt dia1 (mm)", min_value=0, value=12, step=1)
with cols[2]:
    ns2 = st.number_input("spt Nos2", min_value=0, value=0, step=1)
with cols[3]:
    ds2 = st.number_input("spt dia2 (mm)", min_value=0, value=0, step=1)
with cols[4]:
    ns3 = st.number_input("spt Nos3", min_value=0, value=0, step=1)
with cols[5]:
    ds3 = st.number_input("spt dia3 (mm)", min_value=0, value=0, step=1)

Ast_spt_prov = area_bars([ds1, ds2, ds3], [ns1, ns2, ns3])
pt_spt_prov = 100.0 * Ast_spt_prov / (b * d_prov) if b*d_prov>0 else float('nan')
res_spt = "okay" if Ast_spt_prov >= Ast_spt else "not okay"

packet_card("Reinf. details at support", [
    ("Nos.", "", f"{ns1}+{ns2}+{ns3}"),
    ("dia", "mm", f"{ds1}/{ds2}/{ds3}"),
    ("Ast support", "mm²", f"{Ast_spt_prov:.2f}"),
    ("pt support", "%", f"{pt_spt_prov:.2f}"),
    ("Result", "", res_spt),
])

# Span bars
cols2 = st.columns(6)
with cols2[0]:
    nn1 = st.number_input("span Nos1", min_value=0, value=2, step=1)
with cols2[1]:
    dn1 = st.number_input("span dia1 (mm)", min_value=0, value=12, step=1)
with cols2[2]:
    nn2 = st.number_input("span Nos2", min_value=0, value=0, step=1)
with cols2[3]:
    dn2 = st.number_input("span dia2 (mm)", min_value=0, value=0, step=1)
with cols2[4]:
    nn3 = st.number_input("span Nos3", min_value=0, value=0, step=1)
with cols2[5]:
    dn3 = st.number_input("span dia3 (mm)", min_value=0, value=0, step=1)

Ast_span_prov = area_bars([dn1, dn2, dn3], [nn1, nn2, nn3])
pt_span_prov = 100.0 * Ast_span_prov / (b * d_prov) if b*d_prov>0 else float('nan')
res_span = "okay" if Ast_span_prov >= Ast_span else "not okay"

packet_card("Reinf. details at span", [
    ("Nos.", "", f"{nn1}+{nn2}+{nn3}"),
    ("dia", "mm", f"{dn1}/{dn2}/{dn3}"),
    ("Ast span", "mm²", f"{Ast_span_prov:.2f}"),
    ("pt span", "%", f"{pt_span_prov:.2f}"),
    ("Result", "", res_span),
])

st.markdown("---")
st.subheader("Summary status")
status = all([res_depth=="okay", res_spt=="okay", res_span=="okay", status_mu=="ok"]) 
st.success("Head 1: PASS ✅" if status else "Head 1: CHECK ❌") 
st.caption(f"Mu limits: support/span within singly-reinforced limit → {status_mu}")

