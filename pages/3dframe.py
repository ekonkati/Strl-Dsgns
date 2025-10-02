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
                    # Placeholder for analysis results
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

# --- 2. FEA Setup and Analysis Functions ---

def map_degrees_of_freedom(nodes_df):
    """Assigns global degrees of freedom (DOF) indices to each node (6 DOF per node in 3D: Ux, Uy, Uz, Rx, Ry, Rz)."""
    dof_per_node = 6
    total_dofs = len(nodes_df) * dof_per_node
    
    # Assign global DOF index range to each node (starting at 0)
    # The index is the starting DOF for Ux. The six DOFs are [Ux, Uy, Uz, Rx, Ry, Rz]
    nodes_df['dof_start_index'] = nodes_df.index * dof_per_node
    
    nodes_df.attrs['total_dofs'] = total_dofs
    
    st.markdown(f"**Total Global Degrees of Freedom (DoF):** `{total_dofs}`")
    return nodes_df

def calculate_element_stiffness(element, node_coords, props):
    """
    Calculates the 12x12 element stiffness matrix [k] in global coordinates for a 3D element.
    k_global = T^T * k_local * T
    """
    E = props['E']
    G = props['G']
    A = props['A']
    Izz = props['Izz'] # Moment of Inertia about local z (major bending)
    Iyy = props['Iyy'] # Moment of Inertia about local y (minor bending)
    J = props['J']     # Torsional constant

    # 1. Geometry and Direction Cosines
    xi, yi, zi = node_coords[element['start']]
    xj, yj, zj = node_coords[element['end']]
    
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
    if L == 0:
        return np.zeros((12, 12))

    # Direction cosines
    l = (xj - xi) / L
    m = (yj - yi) / L
    n = (zj - zi) / L

    # Assuming a vertical z'-axis (parallel to global Y) for the transformation matrix setup
    # unless the element is vertical (column)
    
    # Define vector V for local x' (axis of the member)
    x_prime_vec = np.array([l, m, n])
    
    # Define vector V_ref (reference vector for y' axis projection)
    # Use global Y-axis (0, 1, 0) unless the element is vertical (then use global Z)
    if np.isclose(l**2 + n**2, 0): # Element is vertical (parallel to global Y)
        V_ref = np.array([0, 0, 1]) # Use Z-axis as reference
    else:
        V_ref = np.array([0, 1, 0]) # Use Y-axis as reference

    # Calculate local y' (y_prime_vec) perpendicular to x'
    # y' = cross(z', x') -> z' = cross(x', V_ref) / |cross(x', V_ref)|
    # We choose z' = cross(x', V_ref)
    z_prime_vec = np.cross(x_prime_vec, V_ref)
    z_prime_vec = z_prime_vec / np.linalg.norm(z_prime_vec)
    
    # Calculate local y' (y_prime_vec)
    y_prime_vec = np.cross(z_prime_vec, x_prime_vec)

    # 2. Local Stiffness Matrix (k_local) - 12x12
    
    # Axial and Torsional stiffness constants
    C1 = E * A / L
    C4 = G * J / L
    
    # Shear and Bending stiffness constants (Bending about local z')
    C2 = 12 * E * Izz / (L**3)
    C3 = 6 * E * Izz / (L**2)
    C5 = 4 * E * Izz / L
    C6 = 2 * E * Izz / L
    
    # Shear and Bending stiffness constants (Bending about local y')
    C7 = 12 * E * Iyy / (L**3)
    C8 = 6 * E * Iyy / (L**2)
    C9 = 4 * E * Iyy / L
    C10 = 2 * E * Iyy / L
    
    k_local = np.zeros((12, 12))
    
    # Local DOF order: [u1, v1, w1, rx1, ry1, rz1, u2, v2, w2, rx2, ry2, rz2]
    
    # Axial (1-7)
    k_local[0, 0], k_local[0, 6] = C1, -C1
    k_local[6, 0], k_local[6, 6] = -C1, C1
    
    # Shear/Bending about z' (2-8 and 6-12)
    k_local[1, 1], k_local[1, 7] = C2, -C2
    k_local[7, 1], k_local[7, 7] = -C2, C2
    
    k_local[1, 5], k_local[1, 11] = C3, C3
    k_local[7, 5], k_local[7, 11] = -C3, -C3
    
    k_local[5, 1], k_local[11, 1] = C3, -C3
    k_local[5, 7], k_local[11, 7] = -C3, C3
    
    k_local[5, 5], k_local[11, 11] = C5, C5
    k_local[5, 11], k_local[11, 5] = C6, C6
    
    # Shear/Bending about y' (3-9 and 5-11) - Note the negative sign on shear terms due to right-hand rule
    k_local[2, 2], k_local[2, 8] = C7, -C7
    k_local[8, 2], k_local[8, 8] = -C7, C7
    
    k_local[2, 4], k_local[2, 10] = -C8, -C8 # Shear-Moment coupling terms
    k_local[8, 4], k_local[8, 10] = C8, C8
    
    k_local[4, 2], k_local[10, 2] = -C8, C8
    k_local[4, 8], k_local[10, 8] = C8, -C8
    
    k_local[4, 4], k_local[10, 10] = C9, C9
    k_local[4, 10], k_local[10, 4] = C10, C10
    
    # Torsion (4-10)
    k_local[3, 3], k_local[3, 9] = C4, -C4
    k_local[9, 3], k_local[9, 9] = -C4, C4
    
    # 3. Transformation Matrix (T) - 12x12
    T_3x3 = np.array([x_prime_vec, y_prime_vec, z_prime_vec])
    
    T = np.zeros((12, 12))
    # Fill T with T_3x3 along the diagonal (T is block diagonal)
    for i in range(4):
        T[i*3:i*3+3, i*3:i*3+3] = T_3x3
        
    # 4. Global Stiffness Matrix (k_global)
    k_global = T.T @ k_local @ T
    
    return k_global

def assemble_global_stiffness(nodes_df, elements, node_coords, props):
    """
    Assembles the Global Stiffness Matrix [K] by placing each element's 
    stiffness matrix into the correct global DoF locations.
    """
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    
    # Initialize the global stiffness matrix (using numpy array for simplicity)
    K_global = np.zeros((total_dofs, total_dofs))
    
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()

    for elem in elements:
        # Calculate element's stiffness matrix in global coordinates
        k_global_elem = calculate_element_stiffness(elem, node_coords, props)
        
        # Get global DoF indices for the element's start and end nodes
        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        
        # Element DoF indices in the global matrix
        dofs_i = np.arange(dof_start_i, dof_start_i + 6)
        dofs_j = np.arange(dof_start_j, dof_start_j + 6)
        
        # Map the 12 element DOFs to the global DOFs
        global_dofs = np.concatenate((dofs_i, dofs_j))
        
        # Assemble the element stiffness into the global matrix K
        for row_elem in range(12):
            for col_elem in range(12):
                global_row = global_dofs[row_elem]
                global_col = global_dofs[col_elem]
                
                K_global[global_row, global_col] += k_global_elem[row_elem, col_elem]
    
    return K_global

def perform_analysis(nodes_df, elements, load_params, node_coords, props):
    """
    Expands the structural analysis process to include FEA setup and mocked results.
    """
    st.info("Initiating 3D Finite Element Analysis (FEA) Setup...")
    
    # 1. Map Degrees of Freedom (DOF)
    nodes_df = map_degrees_of_freedom(nodes_df)
    
    # 2. Element Stiffness Matrices
    stiffness_matrix_placeholder = calculate_element_stiffness(elements[0], node_coords, props)
    st.markdown(f"**Element Stiffness Matrix Size:** `{stiffness_matrix_placeholder.shape}` (12x12 for 3D element)")
    
    # 3. Assemble Global Stiffness Matrix (K)
    K_global = assemble_global_stiffness(nodes_df, elements, node_coords, props)
    st.markdown(f"**Global Stiffness Matrix Size (K):** `{K_global.shape}` (Total DoF x Total DoF)")
    
    # Store K_global in session state for potential future solving
    st.session_state['K_global'] = K_global

    # --- Next FEA Solver Steps (Currently Placeholder) ---
    st.warning("FEA Setup Complete. The following steps are required next:")
    st.markdown("""
    1.  **Define Load Vector [P]:** Convert area loads (slab) and linear loads (self-weight) into equivalent nodal forces.
    2.  **Apply Boundary Conditions:** Reduce the global system [K] by removing rows/columns corresponding to fixed DOFs (e.g., Ux, Uy, Uz, Rx, Ry, Rz at fixed supports).
    3.  **Solve for Displacements [U]:** Solve the reduced system: $\mathbf{U}_{free} = \mathbf{K}_{reduced}^{-1} \mathbf{P}_{reduced}$.
    4.  **Calculate Internal Forces:** Use $\mathbf{F}_{element} = \mathbf{k}_{global} \cdot \mathbf{U}_{element}$ to find Bending Moments, Shear Forces, etc.
    """)
    st.info("Falling back to Mock Analysis results for visualization.")
    
    # Mocking deflection and moment results on elements (to keep visualization working)
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

# --- 3. Plotly Visualization Function (Unchanged) ---
# ... (plot_3d_frame function remains the same) ...
def plot_3d_frame(nodes_df, elements, node_coords, display_mode):
    """
    Creates an interactive 3D Plotly figure for the frame structure,
    with options to show Deflection, BM, or SF.
    """
    fig = go.Figure()

    # --- 1. Plot Nodes ---
    
    # Apply mocked deflection for visualization if requested
    if display_mode == 'Deflection':
        # Apply slight random deflection for visualization realism
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
st.sidebar.header("🔩 Material & Section Properties")

# Assuming concrete for default values (E and G in kN/m2, I and A in m4/m2)
props = {}
props['E'] = st.sidebar.number_input("Young's Modulus E (kN/m²)", value=2.5e7, format="%e", help="E for M25 concrete is ~25000 MPa or 2.5e7 kN/m²")
props['G'] = st.sidebar.number_input("Shear Modulus G (kN/m²)", value=1.0e7, format="%e", help="G is typically E / (2*(1+v)).")

st.sidebar.subheader("Section Properties (Uniform)")
props['A'] = st.sidebar.number_input("Area A (m²)", value=0.12, format="%.4f", help="e.g., 300x400 section: 0.3 * 0.4 = 0.12 m²")
props['Izz'] = st.sidebar.number_input("$I_{zz}$ (m⁴) - Major Axis", value=0.0016, format="%.6f", help="I = b*h³/12 (Major Bending)")
props['Iyy'] = st.sidebar.number_input("$I_{yy}$ (m⁴) - Minor Axis", value=0.0009, format="%.6f", help="I = h*b³/12 (Minor Bending)")
props['J'] = st.sidebar.number_input("Torsional Constant J (m⁴)", value=0.0015, format="%.6f", help="Approximated J for rectangular section")

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
st.title("3D Frame FEA Setup & Mock Analysis App")
st.markdown("""
This application generates the 3D frame geometry and successfully computes the **Global Stiffness Matrix $\mathbf{K}$** using the 3D Stiffness Method (FEA).
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
        
        # 4. Perform Full FEA Setup
        with st.spinner('Calculating element stiffnesses and assembling Global Stiffness Matrix K...'):
            nodes_df, elements = perform_analysis(nodes_df, elements, load_params, node_coords, props)
        
        st.success(f"FEA Setup Complete: {len(nodes_df)} nodes and {len(elements)} elements. Global Stiffness Matrix K is assembled.")

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
            st.markdown(f"**Global Matrix Size:** `{st.session_state['K_global'].shape}`")
            st.markdown(f"**Total DoF:** `{nodes_df.attrs.get('total_dofs', 0)}`")
        with col3:
            st.markdown(f"**Foundation Depth:** `{foundation_depth} m`")

        st.subheader("Mock Load Combination Summary (Placeholder)")
        st.markdown(
            "| Load Case | Combination | Description (IS 456/875)| Status |"
            "|---|---|---|---|"
            "| **DL + LL** | 1.5 DL + 1.5 LL | Gravity Check | Ready for Analysis |"
            "| **DL + LL + WLx** | 1.2(DL + LL $\pm$ WLx) | Lateral Check | Ready for Analysis |"
            "| **DL + LL + ELx** | 1.5(DL $\pm$ ELx $\pm$ 0.3 ELy) | Seismic Check | Ready for Analysis |"
        )
        
        st.subheader("Node Coordinates and DoF Start Indices (First 10)")
        st.dataframe(nodes_df[['id', 'x', 'y', 'z', 'support_type', 'dof_start_index']].head(10))

        # Reset flag
        st.session_state['run_generation'] = False
