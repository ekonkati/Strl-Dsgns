import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Advanced 3D Frame Analyzer")

# --- Constants for Geometry (HARDCODED as requested) ---
# X-Bays: 2x3m
X_SPACINGS = [3.0, 3.0]
# Y-Bays: Assuming 1x5m for a single bay frame in the Y direction (common practice when only X/Z specified)
Y_SPACINGS = [5.0]
# Z-Heights: Below ground 2m (Z=-2.0), then 3x4m floors (Z=0, 4.0, 8.0, 12.0)
Z_HEIGHTS = [-2.0, 0.0, 4.0, 8.0, 12.0]

# Mock Load Definitions
# Nodal loads applied at the top corner node (Node with highest Z, highest X, highest Y)
TOP_NODE_LOADS = [20.0, 15.0, -50.0, 0.0, 0.0, 0.0]  # [Fx, Fy, Fz, Mx, My, Mz]
# UDL applied to all beams (elements that are not columns)
BEAM_UDL = [0.0, 0.0, -10.0] # [wx, wy, wz] (Uniform load of -10 kN/m in the global Z direction)

# --- 1. Object-Oriented Model for the Structure ---

class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        # 6 degrees of freedom (DOF): [dx, dy, dz, rx, ry, rz]
        self.restraints = [False] * 6 # True means the DOF is restrained (fixed).
        self.reactions = [0.0] * 6
        self.applied_loads = [0.0] * 6 # [Fx, Fy, Fz, Mx, My, Mz]
        self.is_support = False

    def __repr__(self):
        return f"Node(id={self.id}, pos=({self.x}, {self.y}, {self.z}))"

class Element:
    """Represents a single beam/column element in the 3D structure."""
    def __init__(self, id, start_node, end_node, props):
        self.id = int(id)
        self.start_node = start_node
        self.end_node = end_node
        self.props = props # Placeholder for E, A, I, J, G
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

# --- 2. Structure Generation & Mock Analysis ---

def generate_structure(x_spacings, y_spacings, z_heights):
    """Generates the nodes and elements based on provided grid spacings and heights."""
    nodes = []
    elements = []
    
    # Generate Grid Coordinates
    x_coords = [0.0] + list(np.cumsum(x_spacings))
    y_coords = [0.0] + list(np.cumsum(y_spacings))
    
    # Use the defined Z_HEIGHTS directly
    z_coords = z_heights 

    node_id_counter = 1
    
    # 1. Create Nodes
    node_map = {}
    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                node = Node(node_id_counter, x, y, z)
                # Apply Fixed Support at the lowest level (Z_HEIGHTS[0])
                if z == z_coords[0]:
                    node.restraints = [True] * 6 # Fixed support
                    node.is_support = True
                
                nodes.append(node)
                node_map[(x, y, z)] = node
                node_id_counter += 1

    # 2. Create Elements (Beams and Columns)
    element_id_counter = 1
    mock_props = {'E': 200e6, 'A': 0.01, 'I': 0.0001} # Mock properties

    # Create Columns (Vertical elements)
    for x in x_coords:
        for y in y_coords:
            for i in range(len(z_coords) - 1):
                start_z, end_z = z_coords[i], z_coords[i+1]
                start_node = node_map[(x, y, start_z)]
                end_node = node_map[(x, y, end_z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                elements.append(element)
                element_id_counter += 1

    # Create Beams (Horizontal elements)
    for z in z_coords:
        # X-direction beams
        for y in y_coords:
            for i in range(len(x_coords) - 1):
                start_x, end_x = x_coords[i], x_coords[i+1]
                start_node = node_map[(start_x, y, z)]
                end_node = node_map[(end_x, y, z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                # Apply UDL to all non-ground beams
                if z > z_coords[0]: # Exclude basement level beams from typical floor UDL
                    element.udl = BEAM_UDL 
                elements.append(element)
                element_id_counter += 1

        # Y-direction beams
        for x in x_coords:
            for i in range(len(y_coords) - 1):
                start_y, end_y = y_coords[i], y_coords[i+1]
                start_node = node_map[(x, start_y, z)]
                end_node = node_map[(x, end_y, z)]
                element = Element(element_id_counter, start_node, end_node, mock_props)
                # Apply UDL to all non-ground beams
                if z > z_coords[0]: # Exclude basement level beams from typical floor UDL
                    element.udl = BEAM_UDL
                elements.append(element)
                element_id_counter += 1
                
    # Apply a specific nodal load at a corner (highest x, highest y, highest z)
    top_corner_node = node_map[(x_coords[-1], y_coords[-1], z_coords[-1])]
    top_corner_node.applied_loads = TOP_NODE_LOADS
    
    return nodes, elements

def perform_analysis(nodes, elements):
    """Mocks the structural analysis and generates sample forces/reactions."""
    # This function is heavily mocked to satisfy the visualization requirements
    
    # 1. Mock Reactions
    for node in nodes:
        if node.is_support:
            # Simple mock reaction generation
            node.reactions[2] = 100.0 + (node.id * 5) # Mock Fz reaction
            node.reactions[5] = 15.0 # Mock Mz reaction

    # 2. Mock Internal Forces for visualization (Crucial for the new feature)
    for element in elements:
        # Simple parabolic/linear force distribution for visualization
        length = element.length()
        if length == 0: continue

        # Simple mock forces based on element type
        if element.is_column:
            # Column: Mostly Axial Compression, some moment
            element.forces_start[0] = -50.0  # Axial (Compression)
            element.forces_end[0] = -50.0
            element.forces_start[5] = 10.0   # M3 start
            element.forces_end[5] = -5.0     # M3 end
        else:
            # Beam (with UDL): Mostly Shear and Bending Moment
            wz = element.udl[2]
            # Mocks simple fixed-fixed end moments/shears for UDL = wL^2/12 and wL/2
            if abs(wz) > 0.001:
                L = length
                moment_mag = (wz * L**2) / 12.0
                shear_mag = (wz * L) / 2.0
                
                element.forces_start[1] = -shear_mag # Shear 2 start
                element.forces_end[1] = shear_mag    # Shear 2 end
                element.forces_start[5] = moment_mag # M3 start (hogging, positive sign convention)
                element.forces_end[5] = -moment_mag  # M3 end (sagging/hogging)
            
            element.forces_start[0] = 5.0 # Small mock tension
            element.forces_end[0] = 5.0

    return nodes, elements

# --- 3. Plotting Functions ---

def plot_3d_frame(nodes, elements, show_nodes, show_elements, show_loads):
    """Creates a 3D Plotly visualization of the frame, nodes, and applied loads."""
    
    data = []

    # 1. Elements (Beams and Columns)
    if show_elements:
        x_elem, y_elem, z_elem = [], [], []
        
        # Color elements based on presence of UDL
        udl_x_elem, udl_y_elem, udl_z_elem = [], [], []
        regular_x_elem, regular_y_elem, regular_z_elem = [], [], []

        for elem in elements:
            x0, y0, z0 = elem.start_node.x, elem.start_node.y, elem.start_node.z
            x1, y1, z1 = elem.end_node.x, elem.end_node.y, elem.end_node.z
            
            # Check for UDL
            has_udl = any(abs(l) > 0.001 for l in elem.udl)
            
            if has_udl:
                udl_x_elem.extend([x0, x1, None])
                udl_y_elem.extend([y0, y1, None])
                udl_z_elem.extend([z0, z1, None])
            else:
                regular_x_elem.extend([x0, x1, None])
                regular_y_elem.extend([y0, y1, None])
                regular_z_elem.extend([z0, z1, None])

        # Plot elements with UDL (Thick, Red)
        data.append(go.Scatter3d(
            x=udl_x_elem, y=udl_y_elem, z=udl_z_elem,
            mode='lines',
            line=dict(color='rgb(255, 0, 0)', width=5),
            name='Elements (w/ UDL)',
            hoverinfo='none'
        ))

        # Plot regular elements (Thin, Gray)
        data.append(go.Scatter3d(
            x=regular_x_elem, y=regular_y_elem, z=regular_z_elem,
            mode='lines',
            line=dict(color='rgb(150, 150, 150)', width=3),
            name='Elements (No UDL)',
            hoverinfo='none'
        ))

    # 2. Nodes
    x_nodes, y_nodes, z_nodes = [], [], []
    node_text = []
    
    if show_nodes:
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

    # 3. Applied Loads (New Feature)
    if show_loads:
        for node in nodes:
            fx, fy, fz = node.applied_loads[:3]
            
            # Only plot if a force is non-zero
            if abs(fx) > 0.001 or abs(fy) > 0.001 or abs(fz) > 0.001:
                # Use small vectors (quiver plot approach)
                scale = 0.5 # Visualization scale factor
                
                # Plotting Fx load (red)
                if abs(fx) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z],
                        u=[fx], v=[0], w=[0],
                        sizemode='absolute', sizeref=abs(fx) * scale,
                        showscale=False, anchor='tip',
                        colorscale=[[0, 'red'], [1, 'red']],
                        name=f'Fx @ {node.id}: {fx:.1f} kN'
                    ))
                
                # Plotting Fy load (green)
                if abs(fy) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z],
                        u=[0], v=[fy], w=[0],
                        sizemode='absolute', sizeref=abs(fy) * scale,
                        showscale=False, anchor='tip',
                        colorscale=[[0, 'green'], [1, 'green']],
                        name=f'Fy @ {node.id}: {fy:.1f} kN'
                    ))

                # Plotting Fz load (purple)
                if abs(fz) > 0.001:
                    data.append(go.Cone(
                        x=[node.x], y=[node.y], z=[node.z],
                        u=[0], v=[0], w=[fz],
                        sizemode='absolute', sizeref=abs(fz) * scale,
                        showscale=False, anchor='tip',
                        colorscale=[[0, 'purple'], [1, 'purple']],
                        name=f'Fz @ {node.id}: {fz:.1f} kN'
                    ))


    # 4. Layout configuration
    max_range = max(max(x_coords), max(y_coords), max(z_coords)) - min(min(x_coords), min(y_coords), min(z_coords))
    
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X (m)', showgrid=True, zeroline=True, tickvals=x_coords),
            yaxis=dict(title='Y (m)', showgrid=True, zeroline=True, tickvals=y_coords),
            zaxis=dict(title='Z (m)', showgrid=True, zeroline=True, tickvals=z_coords),
            aspectmode='data', # Ensures true scale
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
    Fixed_axis must be 'X' or 'Y'.
    """
    
    # Filter nodes and elements that lie on the selected plane
    plane_nodes = [n for n in nodes if n[fixed_axis.lower()] == fixed_coord]
    plane_node_ids = {n.id for n in plane_nodes}
    
    plane_elements = []
    for elem in elements:
        if elem.start_node.id in plane_node_ids and elem.end_node.id in plane_node_ids:
            # Only include elements that are completely within the plane
            is_valid_element = (elem.start_node[fixed_axis.lower()] == fixed_coord and 
                                elem.end_node[fixed_axis.lower()] == fixed_coord)
            if is_valid_element:
                plane_elements.append(elem)

    # Determine plot axes
    if fixed_axis == 'Y':
        # X-Z plane view
        p_axis, p_title = 'x', 'X-Position (m)'
        q_axis, q_title = 'z', 'Z-Position (m)'
    else: # fixed_axis == 'X'
        # Y-Z plane view
        p_axis, p_title = 'y', 'Y-Position (m)'
        q_axis, q_title = 'z', 'Z-Position (m)'

    data = []

    # 1. Elements
    x_elem, y_elem = [], []
    for elem in plane_elements:
        p0, q0 = getattr(elem.start_node, p_axis), getattr(elem.start_node, q_axis)
        p1, q1 = getattr(elem.end_node, p_axis), getattr(elem.end_node, q_axis)
        
        # Color based on UDL
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
        hoverinfo='hovertext'
    ))

    # 3. Applied Loads (Show 2D components)
    load_scale = 0.5
    for node in plane_nodes:
        p, q = getattr(node, p_axis), getattr(node, q_axis)
        
        # Determine force components in the plane
        f_p, f_q = 0, 0
        
        # F_p (Force along the primary axis of the plot, which is not the fixed_axis)
        if p_axis == 'x': f_p = node.applied_loads[0] # Fx
        elif p_axis == 'y': f_p = node.applied_loads[1] # Fy

        # F_q (Force along the secondary axis of the plot, which is Z)
        if q_axis == 'z': f_q = node.applied_loads[2] # Fz

        if abs(f_p) > 0.001 or abs(f_q) > 0.001:
            # Plot F_p (Horizontal arrow)
            if abs(f_p) > 0.001:
                data.append(go.Scatter(
                    x=[p, p + np.sign(f_p) * load_scale], y=[q, q],
                    mode='lines+markers',
                    marker=dict(symbol='arrow', size=10, angleref='next', color='red'),
                    line=dict(width=2, color='red'),
                    name=f'F_p @ {node.id}: {f_p:.1f}',
                    showlegend=False,
                    hoverinfo='text',
                    text=[None, f"F_{p_axis.upper()}={f_p:.1f}kN"]
                ))
            
            # Plot F_q (Vertical arrow)
            if abs(f_q) > 0.001:
                 data.append(go.Scatter(
                    x=[p, p], y=[q, q + np.sign(f_q) * load_scale],
                    mode='lines+markers',
                    marker=dict(symbol='arrow', size=10, angleref='next', color='purple'),
                    line=dict(width=2, color='purple'),
                    name=f'F_q @ {node.id}: {f_q:.1f}',
                    showlegend=False,
                    hoverinfo='text',
                    text=[None, f"F_{q_axis.upper()}={f_q:.1f}kN"]
                ))


    layout = go.Layout(
        xaxis=dict(title=p_title, zeroline=True, showgrid=True),
        yaxis=dict(title=q_title, zeroline=True, showgrid=True, scaleanchor="x", scaleratio=1), # Maintain aspect ratio
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
        st.warning(f"Element {element.id} has zero length.")
        return go.Figure()

    # Get end forces
    # [Axial, Shear2, Shear3, Torque, Moment2, Moment3]
    A1, V21, V31, T1, M21, M31 = element.forces_start
    A2, V22, V32, T2, M22, M32 = element.forces_end
    
    # Get UDL in element's local coordinate system (MOCK: just use global Z for simplicity)
    wz = element.udl[2]
    # For a beam element in a 2D plane (local 1-2 plane), only w2 (local y) is relevant for V2/M3
    # MOCK: Let's assume the element is horizontal and wz acts as w2 (vertical load)
    w2 = -wz 
    
    # Span points (normalized from 0 to 1)
    s_norm = np.linspace(0, 1, 100)
    s = s_norm * L

    # Calculate force diagrams based on simple beam theory (MOCK)
    # 1. Axial Force (Assumed constant)
    axial_force = np.full_like(s, A1)

    # 2. Shear Force V2 (V(x) = V1 + w*x)
    shear_force_v2 = V21 + w2 * s
    
    # 3. Bending Moment M3 (M(x) = M1 + V1*x + w*x^2/2)
    bending_moment_m3 = M31 - V21 * s + (w2 * s**2) / 2.0
    
    
    # --- Create Subplots for the three diagrams ---
    
    fig = go.Figure()
    
    # 1. Axial Force Plot
    fig.add_trace(go.Scatter(x=s, y=axial_force, mode='lines', name='Axial Force (A)', line=dict(color='blue')),)

    # 2. Shear Force V2 Plot
    # Add a subplot below the axial plot. Plotly doesn't use subplots directly in this context, 
    # so we'll stack them using annotations and layout or rely on Streamlit's display area.
    # For simplicity, we'll generate three separate figures.
    
    fig_a = go.Figure(
        data=[go.Scatter(x=s, y=axial_force, mode='lines', line=dict(color='blue', width=3), fill='tozeroy')],
        layout=go.Layout(
            title="Axial Force (kN)",
            xaxis=dict(title=f"Span Position along Element {element.id} (m)"),
            yaxis=dict(title="Axial Force (kN)"),
            height=300
        )
    )
    
    fig_v = go.Figure(
        data=[go.Scatter(x=s, y=shear_force_v2, mode='lines', line=dict(color='orange', width=3), fill='tozeroy')],
        layout=go.Layout(
            title="Shear Force (V2) (kN)",
            xaxis=dict(title="Span Position along Element (m)"),
            yaxis=dict(title="Shear Force (kN)"),
            height=300
        )
    )
    
    # Note on Moment sign: Structural engineers often plot moment on the tension side. 
    # Plotly plots positive up. If we follow the sign convention of positive for compression on side 2 (top)
    # then M3 has the signs shown. I'll plot the magnitude for better visibility.
    fig_m = go.Figure(
        data=[go.Scatter(x=s, y=bending_moment_m3, mode='lines', line=dict(color='green', width=3), fill='tozeroy')],
        layout=go.Layout(
            title="Bending Moment (M3) (kNm)",
            xaxis=dict(title="Span Position along Element (m)"),
            yaxis=dict(title="Moment (kNm)"),
            height=300
        )
    )
    
    return fig_a, fig_v, fig_m


# --- 4. Streamlit App Layout ---

# Initialize Structure (using the new hardcoded geometry)
@st.cache_data
def load_data():
    nodes, elements = generate_structure(X_SPACINGS, Y_SPACINGS, Z_HEIGHTS)
    # Run mock analysis
    nodes, elements = perform_analysis(nodes, elements)
    return nodes, elements

nodes, elements = load_data()

# --- Title and Structure Info ---
st.title("Advanced 3D Frame Analyzer")
st.markdown(f"""
    **Generated Geometry:**
    - **X-Bays:** {X_SPACINGS} m (Total: {sum(X_SPACINGS)}m)
    - **Y-Bays:** {Y_SPACINGS} m (Total: {sum(Y_SPACINGS)}m)
    - **Z-Levels:** {Z_HEIGHTS} m (Basement: {Z_HEIGHTS[0]}m, Top Floor: {Z_HEIGHTS[-1]}m)
""")
st.info(f"Total Nodes: **{len(nodes)}** | Total Elements: **{len(elements)}**")
st.markdown("---")

# --- 3D Plot Controls (MOVED DOWNWARDS as requested) ---
with st.container():
    st.subheader("3D Visualization Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_nodes = st.toggle("Show Nodes", value=True)
    with col2:
        show_elements = st.toggle("Show Elements", value=True)
    with col3:
        # Load toggle for 3D plot
        show_loads_3d = st.toggle("Show Applied Loads", value=True)
st.markdown("---")

# --- 3D Visualization ---
st.plotly_chart(plot_3d_frame(nodes, elements, show_nodes, show_elements, show_loads_3d), use_container_width=True)
st.markdown("---")


# --- 5. Analysis Results Tabs ---
st.header("Analysis Results")

tab1, tab2, tab3 = st.tabs(["2D Elevation View (Grid Frames)", "Support Reactions", "Detailed Element Results"])

# --- Tab 1: 2D Elevation View (Grid Frames) ---
with tab1:
    st.subheader("Selected Grid Frame View (with Applied Loads)")
    plane_axis = st.radio("Select Grid Plane Axis", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'))
    
    if plane_axis == 'X-Z (Y-Gridline)':
        y_coords = sorted(list(set(n.y for n in nodes)))
        selected_y = st.selectbox("Select Y-grid", options=y_coords, format_func=lambda x: f"Y = {x:.1f} m", key='y_coord')
        st.plotly_chart(plot_2d_frame(nodes, elements, 'Y', selected_y), use_container_width=True)
    else:
        x_coords = sorted(list(set(n.x for n in nodes)))
        selected_x = st.selectbox("Select X-grid", options=x_coords, format_func=lambda x: f"X = {x:.1f} m", key='x_coord')
        st.plotly_chart(plot_2d_frame(nodes, elements, 'X', selected_x), use_container_width=True)

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

# --- Tab 3: Detailed Element Results (NEW FEATURE: Element Forces) ---
with tab3:
    st.subheader("Element Force Diagrams")
    
    element_options = {e.id: e for e in elements}
    selected_elem_id = st.selectbox(
        "Select Element ID", 
        options=list(element_options.keys()), 
        format_func=lambda x: f"Element {x} (Nodes {element_options[x].start_node.id}-{element_options[x].end_node.id})"
    )
    
    if selected_elem_id:
        selected_element = element_options[selected_elem_id]
        
        st.markdown(f"#### Forces for Element **{selected_elem_id}** (Length: {selected_element.length():.2f}m)")
        
        # Display End Forces
        forces_df = pd.DataFrame({
            "Force/Moment": ["Axial (A)", "Shear V2", "Shear V3", "Torque (T)", "Moment M2", "Moment M3"],
            "Start Node Force": [f"{f:.2f}" for f in selected_element.forces_start],
            "End Node Force": [f"{f:.2f}" for f in selected_element.forces_end],
        })
        st.dataframe(forces_df, use_container_width=True, hide_index=True)

        # Generate and display force diagrams
        fig_a, fig_v, fig_m = plot_element_forces(selected_element)
        
        # Display the plots for Axial, Shear, and Moment
        st.plotly_chart(fig_a, use_container_width=True)
        st.plotly_chart(fig_v, use_container_width=True)
        st.plotly_chart(fig_m, use_container_width=True)

    else:
        st.info("Please select an element to view its detailed force diagrams.")

# --- Footer ---
st.sidebar.caption("Structural Geometry Parameters")
st.sidebar.code(f"""
X Spacings: {X_SPACINGS}
Y Spacings: {Y_SPACINGS}
Z Heights: {Z_HEIGHTS}
""")
