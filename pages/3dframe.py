import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="3D Streamlit Frame Generator")

# --- 1. Geometry Parsing and Generation Functions ---

def parse_grid_input(input_string):
    """
    Parses flexible grid input strings (e.g., "3x5, 2x4.5, 5") into an array of lengths.
    """
    if not input_string:
        return []
    segments = [s.strip() for s in input_string.split(',') if s.strip()]
    lengths = []

    for segment in segments:
        match = re.match(r'^(\d+)x([\d\.]+)$', segment)
        if match:
            count = int(match.group(1))
            length = float(match.group(2))
            if count > 0 and length > 0:
                lengths.extend([length] * count)
        else:
            try:
                single_length = float(segment)
                if single_length > 0:
                    lengths.append(single_length)
            except ValueError:
                pass # Ignore invalid segments

    return lengths

def generate_grid_geometry(x_lengths, y_heights, z_lengths, foundation_depth):
    """
    Generates the structural geometry (nodes and elements) based on parsed grid lengths
    and foundation depth.
    """
    nodes = []
    elements = []
    node_id_counter = 1
    element_id_counter = 1

    cum_x = np.cumsum([0] + x_lengths)
    
    # Calculate cumulative Y: starts at -foundation_depth, then 0 (ground), then floor heights
    above_ground_heights = [0] + y_heights
    cum_y_above_ground = np.cumsum(above_ground_heights)
    
    # Insert foundation base level (-foundation_depth) at the start
    cum_y = np.insert(cum_y_above_ground, 0, -foundation_depth)
    
    cum_z = np.cumsum([0] + z_lengths)

    x_grid_count = len(cum_x)
    y_grid_count = len(cum_y)
    z_grid_count = len(cum_z)
    
    # Index for foundation base nodes (y=0)
    foundation_base_index = 0

    # Dictionary to quickly map (i, j, k) indices to node_id
    node_grid_map = {}

    # 1. Generate Nodes
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                node = {
                    'id': node_id_counter,
                    'x': cum_x[i],
                    'y': cum_y[j],
                    'z': cum_z[k],
                    # Assign Fixed support at the very bottom (j=0)
                    'support_type': 'Fixed' if j == foundation_base_index else 'None',
                    # Placeholder for analysis results (will be populated by analyze_structure)
                    'deflection_scale': 0, 
                    'deflection_magnitude': 0 
                }
                nodes.append(node)
                node_grid_map[(i, j, k)] = node_id_counter
                node_id_counter += 1
    
    nodes_df = pd.DataFrame(nodes)
    
    # Map node IDs to coordinates for element generation
    node_coords = {n['id']: (n['x'], n['y'], n['z']) for n in nodes}

    # 2. Generate Elements (Beams/Columns)
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                current_id = node_grid_map[(i, j, k)]

                # Connection in X direction (Beams) - Skip floor beams below ground (j=0, j=1)
                if i < x_grid_count - 1 and j >= 1: # Only connect beams from ground floor (j=1) upwards
                    neighbor_id = node_grid_map[(i + 1, j, k)]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-x', 
                        'BM': 0, 'SF': 0 # Placeholder for results
                    })
                    element_id_counter += 1

                # Connection in Y direction (Columns)
                if j < y_grid_count - 1:
                    neighbor_id = node_grid_map[(i, j + 1, k)]
                    column_type = 'foundation-column' if j == 0 else 'column'
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': column_type,
                        'BM': 0, 'SF': 0 # Placeholder for results
                    })
                    element_id_counter += 1

                # Connection in Z direction (Beams) - Skip floor beams below ground (j=0, j=1)
                if k < z_grid_count - 1 and j >= 1: # Only connect beams from ground floor (j=1) upwards
                    neighbor_id = node_grid_map[(i, j, k + 1)]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-z',
                        'BM': 0, 'SF': 0 # Placeholder for results
                    })
                    element_id_counter += 1

    return nodes_df, elements, node_coords

# --- 2. FEA Setup and Analysis Functions (Non-Mock Steps) ---

def map_degrees_of_freedom(nodes_df):
    """Assigns global degrees of freedom (DOF) indices to each node (6 DOF per node in 3D: Ux, Uy, Uz, Rx, Ry, Rz)."""
    dof_per_node = 6
    total_dofs = len(nodes_df) * dof_per_node
    
    # Assign global DOF index range to each node (starting at 0)
    nodes_df['dof_start_index'] = nodes_df.index * dof_per_node
    
    # Store total DOFs in the DataFrame metadata for easy access
    nodes_df.attrs['total_dofs'] = total_dofs
    
    st.markdown(f"**Total Global Degrees of Freedom (DoF):** `{total_dofs}`")
    return nodes_df

def calculate_element_stiffness(element, node_coords):
    """
    Placeholder for calculating the 12x12 element stiffness matrix [k] in global coordinates for a 3D element.
    This requires material (E, G) and section properties (A, I, J).
    """
    # In a real implementation: [k] = [T]^T * [k_local] * [T]
    # For now, return a zero matrix as a placeholder.
    return np.zeros((12, 12)) 

def assemble_global_stiffness(nodes_df, elements, node_coords):
    """
    Placeholder for assembling the Global Stiffness Matrix [K] by placing 
    each element's stiffness matrix into the correct global DoF locations.
    """
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    
    # Initialize the global stiffness matrix (usually a sparse matrix in large FEA)
    K_global = np.zeros((total_dofs, total_dofs))
    
    # In a real app, you would loop through elements, calculate [k], 
    # and use the DOF map to place the 12x12 matrix into K_global.
    
    return K_global

def perform_analysis(nodes_df, elements, load_params, node_coords):
    """
    Expands the structural analysis process to include FEA setup and mocked results.
    """
    st.info("Initiating 3D Finite Element Analysis (FEA) Setup...")
    
    # 1. Map Degrees of Freedom (DOF)
    nodes_df = map_degrees_of_freedom(nodes_df)
    
    # 2. Calculate Element Stiffness Matrices (k)
    # The actual calculation for a single element is mocked here for size reference.
    stiffness_matrix_placeholder = calculate_element_stiffness(elements[0], node_coords)
    st.markdown(f"**Element Stiffness Matrix Size:** `{stiffness_matrix_placeholder.shape}` (12x12 for 3D element)")

    # 3. Assemble Global Stiffness Matrix (K)
    K_global = assemble_global_stiffness(nodes_df, elements, node_coords)
    st.markdown(f"**Global Stiffness Matrix Size (K):** `{K_global.shape}` (Total DoF x Total DoF)")

    # --- FEA Solver Steps (Currently Mocked) ---
    st.warning("The solver steps (Load Vector [P], Applying Boundary Conditions, Solving for Displacements [U], and Calculating Internal Forces) are complex and still require implementation. Falling back to Mock Analysis results for visualization.")
    
    # Mocking deflection and moment results on elements
    for elem in elements:
        if 'beam' in elem['type']:
            # Mock parabolic moment for beams
            elem['BM'] = np.random.uniform(100, 300) 
            elem['SF'] = np.random.uniform(50, 150)
        elif 'column' in elem['type']:
            # Mock moment/shear for columns
            elem['BM'] = np.random.uniform(50, 100) 
            elem['SF'] = np.random.uniform(20, 50) 
            
    # Mocking node deflection (e.g., larger deflection at the top)
    max_y = nodes_df['y'].max()
    if max_y > 0:
        nodes_df['deflection_scale'] = nodes_df['y'] / max_y
        nodes_df['deflection_magnitude'] = nodes_df['deflection_scale'] * 0.5 
    
    return nodes_df, elements

# --- 3. Plotly Visualization Function ---

def plot_3d_frame(nodes_df, elements, node_coords, display_mode):
    """
    Creates an interactive 3D Plotly figure for the frame structure,
    with options to show Deflection, BM, or SF.
    """
    fig = go.Figure()

    # --- 1. Plot Nodes ---
    
    # Apply mocked deflection for visualization if requested
    if display_mode == 'Deflection':
        nodes_df['x_plot'] = nodes_df['x'] + nodes_df['deflection_magnitude'] * np.random.uniform(-1, 1, len(nodes_df))
        nodes_df['y_plot'] = nodes_df['y'] + nodes_df['deflection_magnitude'] * np.random.uniform(0, 0.5, len(nodes_df))
        nodes_df['z_plot'] = nodes_df['z'] + nodes_df['deflection_magnitude'] * np.random.uniform(-1, 1, len(nodes_df))
    else:
        nodes_df['x_plot'] = nodes_df['x']
        nodes_df['y_plot'] = nodes_df['y']
        nodes_df['z_plot'] = nodes_df['z']
        
    fig.add_trace(go.Scatter3d(
        x=nodes_df['x_plot'],
        y=nodes_df['y_plot'],
        z=nodes_df['z_plot'],
        mode='markers',
        marker=dict(
            size=5, 
            color=nodes_df['support_type'].apply(lambda x: 'red' if x == 'Fixed' else 'green'), 
            opacity=0.8
        ),
        hoverinfo='text',
        text=[f"Node ID: {row['id']}<br>Coords: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})<br>Support: {row['support_type']}" for index, row in nodes_df.iterrows()],
        name='Nodes / Supports'
    ))

    # --- 2. Plot Elements (Line Segments) ---
    
    # Group elements by type for trace separation and visualization
    elements_by_type = {'foundation-column': [], 'column': [], 'beam-x': [], 'beam-z': []}
    for elem in elements:
        elements_by_type[elem['type']].append(elem)

    # Visualization settings based on display mode
    if display_mode == 'Bending Moment':
        trace_name_suffix = ' (BM)'
        result_key = 'BM'
        line_color = 'red'
        line_width_scale = 0.05
    elif display_mode == 'Shear Force':
        trace_name_suffix = ' (SF)'
        result_key = 'SF'
        line_color = 'blue'
        line_width_scale = 0.05
    else: # Default or Deflection
        trace_name_suffix = ''
        result_key = None
        
    
    base_width = 5 # Base width for geometry lines
    type_colors = {'foundation-column': 'darkorange', 'column': 'darkorchid', 'beam-x': 'royalblue', 'beam-z': 'darkcyan'}
    
    for elem_type, elems in elements_by_type.items():
        x_coords, y_coords, z_coords = [], [], []
        calculated_widths = [] # List to hold individual calculated widths for averaging
        
        for elem in elems:
            p1 = node_coords[elem['start']]
            p2 = node_coords[elem['end']]
            
            # Use original coordinates for element lines
            x_coords.extend([p1[0], p2[0], None])
            y_coords.extend([p1[1], p2[1], None])
            z_coords.extend([p1[2], p2[2], None])
            
            # Calculate width based on result
            width = base_width
            if result_key and elem[result_key] > 0:
                width += elem[result_key] * line_width_scale
            calculated_widths.append(width) # Store the calculated width


        # Fix: For Scatter3D traces, line width must be a single number, not a list.
        # We calculate the average width for all elements of this type to represent the result magnitude.
        final_line_width = base_width
        if calculated_widths and result_key:
            # Use average width and cap it at 15 to avoid excessively thick lines
            final_line_width = min(sum(calculated_widths) / len(calculated_widths), 15)
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(
                color=line_color if result_key else type_colors[elem_type], 
                width=final_line_width # Pass the single calculated width value
            ),
            name=elem_type.replace('-', ' ').title() + trace_name_suffix,
            hoverinfo='none'
        ))

    # --- 3. Layout Configuration ---
    max_x = nodes_df['x'].max() if not nodes_df.empty else 10
    max_y = nodes_df['y'].max() if not nodes_df.empty else 10
    max_z = nodes_df['z'].max() if not nodes_df.empty else 10
    
    # Include foundation depth in calculation
    min_y = nodes_df['y'].min() if not nodes_df.empty else 0
    y_range_total = max_y - min_y
    
    # Calculate plot limits based on structure size, with buffer
    plot_limit_x = max_x * 1.1 
    plot_limit_z = max_z * 1.1 
    plot_limit_y_top = max_y * 1.1
    plot_limit_y_bottom = min_y * 1.1 # Ensures foundation is visible
    
    fig.update_layout(
        title=f'Interactive 3D Frame Geometry ({display_mode} View)',
        scene=dict(
            xaxis=dict(title='X (Length)', showgrid=True, zeroline=True, range=[0, plot_limit_x]),
            yaxis=dict(title='Y (Height)', showgrid=True, zeroline=True, range=[plot_limit_y_bottom, plot_limit_y_top]),
            zaxis=dict(title='Z (Width)', showgrid=True, zeroline=True, range=[0, plot_limit_z]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700
    )

    return fig

# --- 4. Streamlit App Layout and Logic ---

# Sidebar for User Inputs
st.sidebar.header("📐 Define Grid Geometry")
st.sidebar.markdown("Use format: `L1, N2xL2, ...` (e.g., `5, 2x4.5`). Units are assumed (e.g., meters).")

# Input fields
default_x = "3x5, 2x4"
x_input = st.sidebar.text_input("X Direction Bays (Length)", default_x, help="Example: 3x5, 4, 2x6")

default_y = "4.5, 3x3.5"
y_input = st.sidebar.text_input("Y Direction Heights (Floor)", default_y, help="Example: 4.5, 3x3.5 (Ground + 3 floors)")

default_z = "4x6"
z_input = st.sidebar.text_input("Z Direction Bays (Width)", default_z, help="Example: 4x6, 5")

default_foundation = 3.0
foundation_depth = st.sidebar.number_input(
    "Foundation Depth below Ground (m)", 
    min_value=0.0, 
    value=default_foundation, 
    step=0.5,
    help="Depth of column extension below Y=0 (ground level) for fixed support."
)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Load Parameters (Dead/Live)")

load_params = {}
load_params['slab_thickness'] = st.sidebar.number_input("Slab Thickness (m)", min_value=0.0, value=0.15, step=0.01)
load_params['slab_density'] = st.sidebar.number_input("Concrete Density (kN/m³)", min_value=1.0, value=25.0, step=1.0)
load_params['finish_load'] = st.sidebar.number_input("Floor Finish Load (kN/m²)", min_value=0.0, value=1.5, step=0.1)
load_params['live_load'] = st.sidebar.number_input("Live Load (kN/m²)", min_value=0.0, value=3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("📊 Visualization Mode")
display_mode = st.sidebar.selectbox(
    "Select Analysis Result to Display (Mocked)",
    ['Geometry/Supports', 'Deflection', 'Bending Moment', 'Shear Force']
)

# Button to trigger analysis (implicitly includes generation)
if st.sidebar.button("Run Analysis & Visualize 🚀"):
    st.session_state['run_generation'] = True

# Main Page Content
st.title("3D Frame Geometry & Mock Analysis App")
st.markdown("""
This application generates the 3D frame geometry, including foundations, and provides a framework for structural analysis.
Supports are **Fixed** at the foundation base (Y = `-Foundation Depth`).
""")

# --- 5. Main Logic Block ---

# Initialize session state flag if not present
if 'run_generation' not in st.session_state:
    st.session_state['run_generation'] = True # Run on initial load

if st.session_state.get('run_generation'):
    
    # 1. Parse Inputs
    x_lengths = parse_grid_input(x_input)
    y_heights = parse_grid_input(y_input)
    z_lengths = parse_grid_input(z_input)

    # 2. Validate
    if not x_lengths or not y_heights or not z_lengths:
        st.error("Please ensure all three dimensions (X, Y, Z) have valid, positive length inputs.")
        st.session_state['run_generation'] = False
    else:
        # 3. Generate Geometry
        with st.spinner('Generating 3D geometry and supports...'):
            nodes_df, elements, node_coords = generate_grid_geometry(x_lengths, y_heights, z_lengths, foundation_depth)
        
        # 4. Analysis Setup & Mock Results
        with st.spinner('Setting up FEA matrices and performing mock structural analysis...'):
            nodes_df, elements = perform_analysis(nodes_df, elements, load_params, node_coords)
        
        st.success(f"Generated successfully: {len(nodes_df)} nodes and {len(elements)} elements. Fixed supports are applied at Y={-foundation_depth:.2f}m.")

        # 5. Visualization
        frame_fig = plot_3d_frame(nodes_df, elements, node_coords, display_mode)
        st.plotly_chart(frame_fig, use_container_width=True)

        # 6. Display Summary Data
        st.subheader("Structure Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Total Nodes:** `{len(nodes_df)}`")
            st.markdown(f"**Total Elements:** `{len(elements)}`")
        with col2:
            st.markdown(f"**X Dimensions:** `{x_lengths}`")
            st.markdown(f"**Y Dimensions (Heights):** `{y_heights}`")
        with col3:
            st.markdown(f"**Z Dimensions:** `{z_lengths}`")
            st.markdown(f"**Foundation Depth:** `{foundation_depth} m`")

        st.subheader("Mock Load Combination Summary (Placeholder)")
        st.markdown(
            "| Load Case | Combination | Description (IS 456/875)| Status |"
            "|---|---|---|---|"
            "| **DL + LL** | 1.5 DL + 1.5 LL | Gravity Check | Ready for Analysis |"
            "| **DL + LL + WLx** | 1.2(DL + LL $\pm$ WLx) | Lateral Check | Ready for Analysis |"
            "| **DL + LL + ELx** | 1.5(DL $\pm$ ELx $\pm$ 0.3 ELy) | Seismic Check | Ready for Analysis |"
        )
        
        st.subheader("Node Coordinates (First 10)")
        st.dataframe(nodes_df[['id', 'x', 'y', 'z', 'support_type', 'dof_start_index']].head(10))

        # Reset flag
        st.session_state['run_generation'] = False
