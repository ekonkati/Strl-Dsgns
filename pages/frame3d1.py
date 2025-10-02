import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced Dynamic 3D Frame Analyzer")

# --- 1. Utility Functions for Input Parsing ---

def parse_spacings(spacing_str):
    """Parses a string like '2x3, 4, 3' into a list of float spacings."""
    spacings = []
    # Clean string and handle typical syntax
    spacing_str = spacing_str.lower().replace("m", "").replace(" ", "")
    parts = spacing_str.split(',')
    
    for part in parts:
        if 'x' in part:
            try:
                count, value = part.split('x')
                spacings.extend([float(value)] * int(count))
            except ValueError:
                # Handle malformed 'NxM' strings
                try:
                    spacings.append(float(part))
                except ValueError:
                    continue
        elif part:
            try:
                spacings.append(float(part))
            except ValueError:
                continue
    return spacings

def parse_z_levels(z_str):
    """Parses Z-levels string, accounting for 'below ground' and variable floor heights."""
    z_str = z_str.lower().replace(" ", "").replace("shallbe", "")
    z_heights = [0.0]
    
    # 1. Handle below ground
    basement_depth = 0.0
    below_ground_match = re.search(r'belowgroundis(\d+\.?\d*)m', z_str)
    if below_ground_match:
        basement_depth = float(below_ground_match.group(1))
        z_heights.insert(0, -basement_depth)
        # Remove the basement part from the string for floor parsing
        z_str = re.sub(r'belowgroundis(\d+\.?\d*)m', '', z_str)
        
    # 2. Handle floor heights
    # The remaining string might contain something like '3x4' or '4,3,2'
    floor_parts = parse_spacings(z_str.replace("m", ""))
    
    current_z = 0.0
    for height in floor_parts:
        if height > 0:
            current_z += height
            z_heights.append(current_z)

    # Ensure uniqueness and sort the resulting Z coordinates
    z_heights = sorted(list(set(z_heights)))
    return z_heights

# --- 2. Object-Oriented Model for the Structure ---

class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.restraints = [False] * 6 # [dx, dy, dz, rx, ry, rz]
        self.reactions = [0.0] * 6
        self.applied_loads = [0.0] * 6 # [Fx, Fy, Fz, Mx, My, Mz]
        self.is_support = False

class Element:
    """Represents a single beam/column element in the 3D structure."""
    def __init__(self, id, start_node, end_node, props):
        self.id = int(id)
        self.start_node = start_node
        self.end_node = end_node
        self.props = props 
        # UDL stored as total load (Dead + Live + Seismic/Wind components)
        self.udl = [0.0] * 3 # [wx, wy, wz] (uniform distributed load)
        # Mock results storage for visualization: [Axial, Shear2, Shear3, Torque, Moment2, Moment3]
        self.forces_start = [0.0] * 6
        self.forces_end = [0.0] * 6
        self.is_column = self.start_node.x == self.end_node.x and self.start_node.y == self.end_node.y

    def length(self):
        """Calculates the element length."""
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        dz = self.end_node.z - self.start_node.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

# --- 3. Structure Generation & Load Application ---

def generate_structure(x_spacings, y_spacings, z_heights):
    """Generates the nodes and elements based on grid definitions."""
    nodes = []
    elements = []
    
    x_coords = [0.0] + list(np.cumsum(x_spacings))
    y_coords = [0.0] + list(np.cumsum(y_spacings))
    z_coords = z_heights 

    node_id_counter = 1
    node_map = {}
    
    # 1. Create Nodes and Supports
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                node = Node(node_id_counter, x, y, z)
                # Apply Fixed Support at the lowest level
                if z == z_coords[0]:
                    node.restraints = [True] * 6 
                    node.is_support = True
                
                nodes.append(node)
                node_map[(x, y, z)] = node
                node_id_counter += 1

    # 2. Create Elements (Columns and Beams)
    element_id_counter = 1
    mock_props = {'E': 200e6, 'A': 0.01, 'I': 0.0001} 

    # Columns (Vertical elements)
    for x in x_coords:
        for y in y_coords:
            for i in range(len(z_coords) - 1):
                start_z, end_z = z_coords[i], z_coords[i+1]
                start_node = node_map[(x, y, start_z)]
                end_node = node_map[(x, y, end_z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                elements.append(element)
                element_id_counter += 1

    # Beams (Horizontal elements)
    for z in z_coords:
        if z == z_coords[0]: continue # No beams in the basement level for a simple frame model
        
        # X-direction beams
        for y in y_coords:
            for i in range(len(x_coords) - 1):
                start_x, end_x = x_coords[i], x_coords[i+1]
                start_node = node_map[(start_x, y, z)]
                end_node = node_map[(end_x, y, z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                elements.append(element)
                element_id_counter += 1

        # Y-direction beams
        for x in x_coords:
            for i in range(len(y_coords) - 1):
                start_y, end_y = y_coords[i], y_coords[i+1]
                start_node = node_map[(x, start_y, z)]
                end_node = node_map[(x, end_node, z)] # Error here fixed: use end_y
                end_node = node_map[(x, end_y, z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                elements.append(element)
                element_id_counter += 1
                
    return nodes, elements, x_coords, y_coords, z_coords

def apply_user_loads(nodes, elements, dead_load, live_load, wind_x, wind_y, seismic_x, seismic_y):
    """Applies UDLs to beams and lateral loads to nodes based on user input."""
    
    # 1. Apply Gravity Loads (Dead + Live) as UDL on Beams
    gravity_udl = -(dead_load + live_load) # Negative for Z-direction (downwards)
    for elem in elements:
        if not elem.is_column and elem.end_node.z > 0: # Apply only to above-ground beams
            elem.udl[2] += gravity_udl # Add to Fz component
    
    # 2. Apply Lateral Loads (Wind/Seismic) to all nodes at the top floor
    max_z = max(n.z for n in nodes) if nodes else 0

    for node in nodes:
        if node.z == max_z and max_z > 0:
            # Apply lateral forces at the top floor
            node.applied_loads[0] = wind_x + seismic_x # Fx
            node.applied_loads[1] = wind_y + seismic_y # Fy

    # 3. Apply a small example vertical nodal load at one corner for demonstration
    top_corner_nodes = [n for n in nodes if n.z == max_z and n.x == max(n.x for n in nodes) and n.y == max(n.y for n in nodes)]
    if top_corner_nodes:
        top_corner_nodes[0].applied_loads[2] += -20.0 # Additional downward point load Fz
        
    return nodes, elements

def perform_analysis(nodes, elements):
    """Mocks the structural analysis and generates sample forces/reactions."""
    
    # 1. Mock Reactions
    for node in nodes:
        if node.is_support:
            # Simple mock reaction generation proportional to load
            total_udl_load = sum(e.udl[2] * e.length() for e in elements if e.start_node.is_support or e.end_node.is_support)
            node.reactions[2] = -50.0 - total_udl_load * 0.1 # Mock Fz reaction (upward)

    # 2. Mock Internal Forces
    for element in elements:
        length = element.length()
        if length == 0: continue

        if element.is_column:
            # Column: Mostly Axial Compression
            element.forces_start[0] = -100.0  
            element.forces_end[0] = -100.0
            element.forces_start[5] = 10.0   
            element.forces_end[5] = -5.0     
        else:
            # Beam (with UDL): Shear and Bending Moment
            wz = element.udl[2]
            
            if abs(wz) > 0.001:
                L = length
                # Approximate fixed-fixed end moments/shears for UDL
                moment_mag = (wz * L**2) / 12.0
                shear_mag = (wz * L) / 2.0
                
                element.forces_start[1] = -shear_mag 
                element.forces_end[1] = shear_mag    
                element.forces_start[5] = moment_mag 
                element.forces_end[5] = -moment_mag 
            
            # Mock axial force (due to frame action)
            element.forces_start[0] = 5.0 
            element.forces_end[0] = 5.0

    return nodes, elements

# --- 4. Plotting Functions ---

def plot_3d_frame(nodes, elements, show_nodes, show_elements, show_loads, x_coords, y_coords, z_coords):
    """Creates a 3D Plotly visualization of the frame, nodes, and applied loads."""
    
    data = []

    # 1. Elements
    if show_elements:
        udl_x_elem, udl_y_elem, udl_z_elem = [], [], []
        regular_x_elem, regular_y_elem, regular_z_elem = [], [], []

        for elem in elements:
            x0, y0, z0 = elem.start_node.x, elem.start_node.y, elem.start_node.z
            x1, y1, z1 = elem.end_node.x, elem.end_node.y, elem.end_node.z
            
            has_udl = any(abs(l) > 0.001 for l in elem.udl)
            
            if has_udl:
                udl_x_elem.extend([x0, x1, None])
                udl_y_elem.extend([y0, y1, None])
                udl_z_elem.extend([z0, z1, None])
            else:
                regular_x_elem.extend([x0, x1, None])
                regular_y_elem.extend([y0, y1, None])
                regular_z_elem.extend([z0, z1, None])

        data.append(go.Scatter3d(
            x=udl_x_elem, y=udl_y_elem, z=udl_z_elem,
            mode='lines',
            line=dict(color='rgb(255, 0, 0)', width=5),
            name='Elements (w/ UDL)',
            hoverinfo='none'
        ))

        data.append(go.Scatter3d(
            x=regular_x_elem, y=regular_y_elem, z=regular_z_elem,
            mode='lines',
            line=dict(color='rgb(150, 150, 150)', width=3),
            name='Elements (No UDL)',
            hoverinfo='none'
        ))

    # 2. Nodes
    if show_nodes:
        x_nodes, y_nodes, z_nodes = [], [], []
        node_text = []
        for node in nodes:
            x_nodes.append(node.x)
            y_nodes.append(node.y)
            z_nodes.append(node.z)
            node_text.append(f"Node {node.id}<br>({node.x}, {node.y}, {node.z})")

        data.append(go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers',
            marker=dict(size=5, color='blue', symbol='circle'),
            name='Nodes',
            text=node_text,
            hoverinfo='text'
        ))

    # 3. Applied Loads 
    if show_loads:
        load_scale = 0.05 
        
        for node in nodes:
            fx, fy, fz = node.applied_loads[:3]
            
            if abs(fx) > 0.001 or abs(fy) > 0.001 or abs(fz) > 0.001:
                
                if abs(fx) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z], u=[fx], v=[0], w=[0],
                        sizemode='absolute', sizeref=abs(fx) * load_scale, showscale=False, anchor='tip',
                        colorscale=[[0, 'red'], [1, 'red']], name='Fx', 
                        hovertext=[f'Fx: {fx:.1f} kN @ N{node.id}'], hoverinfo='text'
                    ))
                
                if abs(fy) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z], u=[0], v=[fy], w=[0],
                        sizemode='absolute', sizeref=abs(fy) * load_scale, showscale=False, anchor='tip',
                        colorscale=[[0, 'green'], [1, 'green']], name='Fy',
                        hovertext=[f'Fy: {fy:.1f} kN @ N{node.id}'], hoverinfo='text'
                    ))

                if abs(fz) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z], u=[0], v=[0], w=[fz],
                        sizemode='absolute', sizeref=abs(fz) * load_scale, showscale=False, anchor='tip',
                        colorscale=[[0, 'purple'], [1, 'purple']], name='Fz',
                        hovertext=[f'Fz: {fz:.1f} kN @ N{node.id}'], hoverinfo='text'
                    ))

    # 4. Layout configuration
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X (m)', showgrid=True, zeroline=True, tickvals=x_coords),
            yaxis=dict(title='Y (m)', showgrid=True, zeroline=True, tickvals=y_coords),
            zaxis=dict(title='Z (m)', showgrid=True, zeroline=True, tickvals=z_coords),
            aspectmode='data', 
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title='3D Frame Model',
        height=700,
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode='closest'
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_2d_frame(nodes, elements, fixed_axis, fixed_coord):
    """
    Creates a 2D Plotly visualization of a selected grid plane, showing elements and loads.
    """
    
    plane_nodes = [n for n in nodes if getattr(n, fixed_axis.lower()) == fixed_coord]
    plane_node_ids = {n.id for n in plane_nodes}
    
    plane_elements = []
    for elem in elements:
        if elem.start_node.id in plane_node_ids and elem.end_node.id in plane_node_ids:
            is_valid_element = (getattr(elem.start_node, fixed_axis.lower()) == fixed_coord and 
                                getattr(elem.end_node, fixed_axis.lower()) == fixed_coord)
            if is_valid_element:
                plane_elements.append(elem)

    if fixed_axis == 'Y':
        p_axis, p_title = 'x', 'X-Position (m)'
        q_axis, q_title = 'z', 'Z-Position (m)'
    else: 
        p_axis, p_title = 'y', 'Y-Position (m)'
        q_axis, q_title = 'z', 'Z-Position (m)'

    data = []

    # 1. Elements
    for elem in plane_elements:
        p0, q0 = getattr(elem.start_node, p_axis), getattr(elem.start_node, q_axis)
        p1, q1 = getattr(elem.end_node, p_axis), getattr(elem.end_node, q_axis)
        line_color = 'red' if any(abs(l) > 0.001 for l in elem.udl) else 'gray'

        data.append(go.Scatter(
            x=[p0, p1], y=[q0, q1],
            mode='lines',
            line=dict(color=line_color, width=4),
            showlegend=False,
            hoverinfo='none'
        ))

    # 2. Nodes
    p_nodes = [getattr(n, p_axis) for n in plane_nodes]
    q_nodes = [getattr(n, q_axis) for n in plane_nodes]
    node_text = [f"Node {n.id}<br>({getattr(n, p_axis):.1f}, {getattr(n, q_axis):.1f})" for n in plane_nodes]

    data.append(go.Scatter(
        x=p_nodes, y=q_nodes,
        mode='markers+text',
        marker=dict(size=8, color='blue', symbol='circle'),
        text=[f"N{n.id}" for n in plane_nodes],
        textposition="top center",
        name='Nodes',
        hovertext=node_text,
        # FIX: Changed 'hovertext' to 'text' to resolve Plotly ValueError
        hoverinfo='text' 
    ))

    # 3. Applied Loads (Show 2D components)
    load_scale = 0.5
    for node in plane_nodes:
        p, q = getattr(node, p_axis), getattr(node, q_axis)
        
        f_p, f_q = 0, 0
        if p_axis == 'x': f_p = node.applied_loads[0] 
        elif p_axis == 'y': f_p = node.applied_loads[1] 
        if q_axis == 'z': f_q = node.applied_loads[2] 

        if abs(f_p) > 0.001 or abs(f_q) > 0.001:
            if abs(f_p) > 0.001:
                data.append(go.Scatter(
                    x=[p, p + np.sign(f_p) * load_scale], y=[q, q],
                    mode='lines+markers',
                    marker=dict(symbol='arrow', size=10, angleref='next', color='red'),
                    line=dict(width=2, color='red'),
                    name=f'F_{p_axis.upper()}',
                    showlegend=False,
                    hoverinfo='text',
                    text=[None, f"F_{p_axis.upper()}={f_p:.1f}kN"]
                ))
            
            if abs(f_q) > 0.001:
                 data.append(go.Scatter(
                    x=[p, p], y=[q, q + np.sign(f_q) * load_scale],
                    mode='lines+markers',
                    marker=dict(symbol='arrow', size=10, angleref='next', color='purple'),
                    line=dict(width=2, color='purple'),
                    name=f'F_{q_axis.upper()}',
                    showlegend=False,
                    hoverinfo='text',
                    text=[None, f"F_{q_axis.upper()}={f_q:.1f}kN"]
                ))


    layout = go.Layout(
        xaxis=dict(title=p_title, zeroline=True, showgrid=True),
        yaxis=dict(title=q_title, zeroline=True, showgrid=True, scaleanchor="x", scaleratio=1), 
        title=f"2D Elevation View: {fixed_axis}={fixed_coord:.1f} Plane (Loads Shown)",
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

def plot_element_forces(element):
    """Generates Plotly plots for Axial Force, Shear Force (V2), and Bending Moment (M3)."""
    L = element.length()
    if L == 0:
        return go.Figure(), go.Figure(), go.Figure()

    A1, V21, V31, T1, M21, M31 = element.forces_start
    A2, V22, V32, T2, M22, M32 = element.forces_end
    
    # Mock assumption: UDL in Z (wz) drives V2/M3
    wz = element.udl[2]
    w2 = -wz 
    
    s_norm = np.linspace(0, 1, 100)
    s = s_norm * L

    # Axial Force (A)
    axial_force = np.full_like(s, A1)

    # Shear Force V2 
    shear_force_v2 = V21 + w2 * s
    
    # Bending Moment M3 
    bending_moment_m3 = M31 - V21 * s + (w2 * s**2) / 2.0
    
    # Figure for Axial Force
    fig_a = go.Figure(
        data=[go.Scatter(x=s, y=axial_force, mode='lines', line=dict(color='blue', width=3), fill='tozeroy')],
        layout=go.Layout(
            title=f"Axial Force (A) - Span: {L:.2f}m",
            xaxis=dict(title="Span Position (m)"),
            yaxis=dict(title="Axial Force (kN)"),
            height=300
        )
    )
    
    # Figure for Shear Force V2
    fig_v = go.Figure(
        data=[go.Scatter(x=s, y=shear_force_v2, mode='lines', line=dict(color='orange', width=3), fill='tozeroy')],
        layout=go.Layout(
            title="Shear Force ($V_2$) (kN)",
            xaxis=dict(title="Span Position (m)"),
            yaxis=dict(title="Shear Force (kN)"),
            height=300
        )
    )
    
    # Figure for Bending Moment M3
    fig_m = go.Figure(
        data=[go.Scatter(x=s, y=bending_moment_m3, mode='lines', line=dict(color='green', width=3), fill='tozeroy')],
        layout=go.Layout(
            title="Bending Moment ($M_3$) (kNm)",
            xaxis=dict(title="Span Position (m)"),
            yaxis=dict(title="Moment (kNm)"),
            height=300
        )
    )
    
    return fig_a, fig_v, fig_m

# --- 5. Streamlit Application ---

# The cache function now takes inputs to recalculate when they change
@st.cache_resource
def load_data(x_str, y_str, z_str, dead_load, live_load, wind_x, wind_y, seismic_x, seismic_y):
    """Parses inputs, generates the structure, applies loads, and performs mock analysis."""
    
    X_SPACINGS = parse_spacings(x_str)
    Y_SPACINGS = parse_spacings(y_str)
    Z_HEIGHTS = parse_z_levels(z_str)
    
    if not X_SPACINGS or len(Z_HEIGHTS) < 2:
        st.error("Please provide valid geometry inputs.")
        return [], [], [], [], [], [] # Return empty lists to prevent crashing
        
    nodes, elements, x_coords, y_coords, z_coords = generate_structure(X_SPACINGS, Y_SPACINGS, Z_HEIGHTS)
    
    # Apply user-defined and standard loads
    nodes, elements = apply_user_loads(nodes, elements, dead_load, live_load, wind_x, wind_y, seismic_x, seismic_y)
    
    # Run the mock analysis to get internal forces and reactions
    nodes, elements = perform_analysis(nodes, elements)
    
    return nodes, elements, x_coords, y_coords, z_coords, X_SPACINGS, Y_SPACINGS, Z_HEIGHTS


# --- Sidebar Inputs for Geometry and Loads ---
st.sidebar.header("Structural Model Inputs")

with st.sidebar.expander("1. Geometry Inputs"):
    # Using the required input format as example defaults
    x_bays_str = st.text_input("X-Bays (e.g., 2x3, 4)", value="2x3", key='x_bays')
    y_bays_str = st.text_input("Y-Bays (e.g., 5)", value="1x5", key='y_bays')
    z_levels_str = st.text_input(
        "Z-Levels (e.g., 3x4, below ground is 2m)", 
        value="3x4, below ground is 2m", 
        key='z_levels'
    )
    
with st.sidebar.expander("2. Load Inputs (kN/m or kN)"):
    st.subheader("Gravity UDL on Beams")
    dead_load = st.number_input("Dead Load (kN/m)", value=5.0, min_value=0.0)
    live_load = st.number_input("Live Load (kN/m)", value=3.0, min_value=0.0)
    
    st.subheader("Lateral Load (Top Floor Nodes)")
    wind_x = st.number_input("Wind Load Fx (kN)", value=5.0)
    wind_y = st.number_input("Wind Load Fy (kN)", value=2.0)
    seismic_x = st.number_input("Seismic Load Fx (kN)", value=10.0)
    seismic_y = st.number_input("Seismic Load Fy (kN)", value=4.0)
    

# --- Load Data and Structure ---
nodes, elements, x_coords, y_coords, z_coords, X_SPACINGS, Y_SPACINGS, Z_HEIGHTS = load_data(
    x_bays_str, y_bays_str, z_levels_str, dead_load, live_load, wind_x, wind_y, seismic_x, seismic_y
)


# --- Main Content Layout ---

st.title("Advanced 3D Frame Analyzer")
st.markdown(f"""
    **Current Geometry:**
    - **X-Bays:** {X_SPACINGS} m (Total: {sum(X_SPACINGS):.1f}m)
    - **Y-Bays:** {Y_SPACINGS} m (Total: {sum(Y_SPACINGS):.1f}m)
    - **Z-Levels:** Lowest: {Z_HEIGHTS[0]}m, Floors: {Z_HEIGHTS[1:]} m
""")
st.info(f"Total Nodes: **{len(nodes)}** | Total Elements: **{len(elements)}**")
st.markdown("---")

# --- 3D Plot Controls (Relocated as requested) ---
with st.container():
    st.subheader("3D Visualization Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_nodes = st.toggle("Show Nodes", value=True)
    with col2:
        show_elements = st.toggle("Show Elements (Red = w/ UDL)", value=True)
    with col3:
        show_loads_3d = st.toggle("Show Applied Nodal Loads", value=True)
st.markdown("---")

# --- 3D Visualization ---
if nodes:
    st.plotly_chart(plot_3d_frame(nodes, elements, show_nodes, show_elements, show_loads_3d, x_coords, y_coords, z_coords), use_container_width=True)
else:
    st.warning("Structure could not be generated. Please check your geometry inputs.")
st.markdown("---")


# --- 6. Analysis Results Tabs ---
st.header("Analysis Results")

tab1, tab2, tab3 = st.tabs(["2D Elevation View (Grid Frames)", "Support Reactions", "Detailed Element Results"])

# --- Tab 1: 2D Elevation View (Grid Frames) ---
with tab1:
    if nodes:
        st.subheader("Selected Grid Frame View (with Applied Loads)")
        
        # Collect unique coordinates for selection boxes
        unique_x = sorted(list(set(n.x for n in nodes)))
        unique_y = sorted(list(set(n.y for n in nodes)))

        plane_axis = st.radio("Select Grid Plane Axis", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'))
        
        if plane_axis == 'X-Z (Y-Gridline)' and unique_y:
            selected_y = st.selectbox("Select Y-grid", options=unique_y, format_func=lambda x: f"Y = {x:.1f} m", key='y_coord')
            st.plotly_chart(plot_2d_frame(nodes, elements, 'Y', selected_y), use_container_width=True)
        elif plane_axis == 'Y-Z (X-Gridline)' and unique_x:
            selected_x = st.selectbox("Select X-grid", options=unique_x, format_func=lambda x: f"X = {x:.1f} m", key='x_coord')
            st.plotly_chart(plot_2d_frame(nodes, elements, 'X', selected_x), use_container_width=True)
        else:
            st.info("No grid lines available for the selected axis.")

# --- Tab 2: Support Reactions ---
with tab2:
    st.subheader("Support Reactions")
    support_nodes = {n.id: n for n in nodes if n.is_support}
    if support_nodes:
        node_id = st.selectbox("Select support node", options=list(support_nodes.keys()), key='support_node_id')
        selected_node = support_nodes[node_id]
        
        reaction_data = pd.DataFrame({
            "DOF": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
            "Value (kN, kNm)": [f"{r:.2f}" for r in selected_node.reactions],
            "Restraint": [("FIXED" if r else "FREE") for r in selected_node.restraints]
        })
        st.dataframe(reaction_data, use_container_width=True, hide_index=True)
    else:
        st.info("No fixed support nodes found in the structure.")

# --- Tab 3: Detailed Element Results (Element Forces) ---
with tab3:
    st.subheader("Element Force Diagrams (Axial, Shear $V_2$, Moment $M_3$)")
    
    element_options = {e.id: e for e in elements}
    selected_elem_id = st.selectbox(
        "Select Element ID", 
        options=list(element_options.keys()), 
        format_func=lambda x: f"Element {x} (Nodes {element_options[x].start_node.id}-{element_options[x].end_node.id})"
    )
    
    if selected_elem_id:
        selected_element = element_options[selected_elem_id]
        
        st.markdown(f"#### Forces for Element **{selected_elem_id}** (Length: {selected_element.length():.2f}m)")
        
        # Display End Forces Table
        forces_df = pd.DataFrame({
            "Force/Moment": ["Axial (A)", "Shear $V_2$", "Shear $V_3$", "Torque (T)", "Moment $M_2$", "Moment $M_3$"],
            "Start Node Force": [f"{f:.2f}" for f in selected_element.forces_start],
            "End Node Force": [f"{f:.2f}" for f in selected_element.forces_end],
        })
        st.dataframe(forces_df, use_container_width=True, hide_index=True)

        # Generate and display force diagrams
        fig_a, fig_v, fig_m = plot_element_forces(selected_element)
        
        st.plotly_chart(fig_a, use_container_width=True)
        st.plotly_chart(fig_v, use_container_width=True)
        st.plotly_chart(fig_m, use_container_width=True)

    else:
        st.info("Please select an element to view its detailed force diagrams.")
