import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="3D Streamlit Frame Generator")

# --- 1. Utility Functions ---

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

def calculate_rc_properties(b, h, E, nu=0.2):
    """
    Calculates 3D structural properties (A, Izz, Iyy, J, G) for a rectangular section b x h.
    The local coordinate system is typically: x' (along element), y' (minor axis), z' (major axis).
    For a rectangular section b x h:
    - Area A = b * h
    - Izz (bending about major axis z') = b * h^3 / 12
    - Iyy (bending about minor axis y') = h * b^3 / 12
    - J (torsional constant) is approximated for a rectangular section.
    """
    b_val = b / 1000.0  # Convert mm to m
    h_val = h / 1000.0  # Convert mm to m
    
    A = b_val * h_val
    
    # Izz is Moment of Inertia about the axis parallel to 'b' (major bending axis for columns/beams in their strong direction)
    Izz = b_val * h_val**3 / 12.0
    
    # Iyy is Moment of Inertia about the axis parallel to 'h' (minor bending axis)
    Iyy = h_val * b_val**3 / 12.0
    
    # Torsional constant J approximation for rectangular section (Timoshenko & Goodier)
    # J = k * b * h^3 where k depends on h/b. 
    # For concrete, we often use a simplified approach or look-up tables.
    # Using the more general approximate formula for closed rectangular section, simplified for solid:
    if b_val > h_val: # b is the longer side (b_val, h_val)
        b_long = b_val
        h_short = h_val
    else:
        b_long = h_val
        h_short = b_val
    
    # A common approximation for solid rectangles (k2 from theory)
    k2 = 1/3 * (1 - 0.63 * (h_short/b_long) * (1 - h_short**4 / (12*b_long**4)))
    J = k2 * b_long * h_short**3
    
    # Shear Modulus G
    G = E / (2 * (1 + nu))

    return {
        'A': A, 'Izz': Izz, 'Iyy': Iyy, 'J': J, 'E': E, 'G': G, 
        'b_mm': b, 'h_mm': h
    }

# --- 2. Geometry Parsing and Generation Functions ---

def generate_grid_geometry(x_lengths, y_heights, z_lengths, foundation_depth, prop_column, prop_beam):
    """
    Generates the structural geometry (nodes and elements) based on parsed grid lengths
    and foundation depth, assigning element-specific properties.
    """
    nodes = []
    elements = []
    node_id_counter = 1
    element_id_counter = 1

    cum_x = np.cumsum([0] + x_lengths)
    above_ground_heights = [0] + y_heights
    cum_y_above_ground = np.cumsum(above_ground_heights)
    cum_y = np.insert(cum_y_above_ground, 0, -foundation_depth)
    cum_z = np.cumsum([0] + z_lengths)

    x_grid_count = len(cum_x)
    y_grid_count = len(cum_y)
    z_grid_count = len(cum_z)
    
    foundation_base_index = 0
    node_grid_map = {}

    # 1. Generate Nodes (Unchanged)
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                node = {
                    'id': node_id_counter,
                    'x': cum_x[i],
                    'y': cum_y[j],
                    'z': cum_z[k],
                    'support_type': 'Fixed' if j == foundation_base_index else 'None',
                    'deflection_scale': 0, 
                    'deflection_magnitude': 0 
                }
                nodes.append(node)
                node_grid_map[(i, j, k)] = node_id_counter
                node_id_counter += 1
    
    nodes_df = pd.DataFrame(nodes)
    node_coords = {n['id']: (n['x'], n['y'], n['z']) for n in nodes}

    # 2. Generate Elements (Beams/Columns) and assign properties
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                current_id = node_grid_map[(i, j, k)]

                # Connection in X direction (Beams) - Only connect beams from ground floor (j=1) upwards
                if i < x_grid_count - 1 and j >= 1: 
                    neighbor_id = node_grid_map[(i + 1, j, k)]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-x', 
                        'BM': 0, 'SF': 0, **prop_beam # Assign beam properties
                    })
                    element_id_counter += 1

                # Connection in Y direction (Columns)
                if j < y_grid_count - 1:
                    neighbor_id = node_grid_map[(i, j + 1, k)]
                    column_type = 'foundation-column' if j == 0 else 'column'
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': column_type,
                        'BM': 0, 'SF': 0, **prop_column # Assign column properties
                    })
                    element_id_counter += 1

                # Connection in Z direction (Beams) - Only connect beams from ground floor (j=1) upwards
                if k < z_grid_count - 1 and j >= 1: 
                    neighbor_id = node_grid_map[(i, j, k + 1)]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-z',
                        'BM': 0, 'SF': 0, **prop_beam # Assign beam properties
                    })
                    element_id_counter += 1

    return nodes_df, elements, node_coords

# --- 3. FEA Setup and Analysis Functions ---

def map_degrees_of_freedom(nodes_df):
    """Assigns global degrees of freedom (DOF) indices to each node."""
    dof_per_node = 6
    total_dofs = len(nodes_df) * dof_per_node
    
    nodes_df['dof_start_index'] = nodes_df.index * dof_per_node
    nodes_df.attrs['total_dofs'] = total_dofs
    
    st.markdown(f"**Total Global Degrees of Freedom (DoF):** `{total_dofs}`")
    return nodes_df

def calculate_element_stiffness(element, node_coords):
    """
    Calculates the 12x12 element stiffness matrix [k] in global coordinates 
    using element-specific properties (E, G, A, Izz, Iyy, J).
    """
    E = element['E']
    G = element['G']
    A = element['A']
    Izz = element['Izz'] # Moment of Inertia about local z (major bending)
    Iyy = element['Iyy'] # Moment of Inertia about local y (minor bending)
    J = element['J']     # Torsional constant

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

    # Define vector V for local x' (axis of the member)
    x_prime_vec = np.array([l, m, n])
    
    # Define vector V_ref (reference vector for y' axis projection)
    if np.isclose(l**2 + n**2, 0): # Element is vertical (parallel to global Y)
        V_ref = np.array([0, 0, 1]) # Use Z-axis as reference
    else:
        V_ref = np.array([0, 1, 0]) # Use Y-axis as reference

    # Calculate local z' (z_prime_vec)
    z_prime_vec = np.cross(x_prime_vec, V_ref)
    norm_z = np.linalg.norm(z_prime_vec)
    z_prime_vec = z_prime_vec / norm_z if norm_z > 1e-6 else np.array([0, 0, 0])
    
    # Calculate local y' (y_prime_vec)
    y_prime_vec = np.cross(z_prime_vec, x_prime_vec)

    # 2. Local Stiffness Matrix (k_local) - 12x12
    k_local = np.zeros((12, 12))
    
    # Stiffness constants
    C1 = E * A / L             # Axial
    C4 = G * J / L             # Torsion
    C2 = 12 * E * Izz / (L**3) # Shear/Bending z' (Major)
    C3 = 6 * E * Izz / (L**2)
    C5 = 4 * E * Izz / L
    C6 = 2 * E * Izz / L
    C7 = 12 * E * Iyy / (L**3) # Shear/Bending y' (Minor)
    C8 = 6 * E * Iyy / (L**2)
    C9 = 4 * E * Iyy / L
    C10 = 2 * E * Iyy / L
    
    # Populate k_local (standard 3D beam element stiffness matrix)
    
    # [u, v, w, rx, ry, rz] @ node 1 and 2
    
    # u/u, v/v, w/w, rx/rx, ry/ry, rz/rz
    k_local[0, 0], k_local[6, 6] = C1, C1  # Axial
    k_local[1, 1], k_local[7, 7] = C2, C2  # Shear V
    k_local[2, 2], k_local[8, 8] = C7, C7  # Shear W
    k_local[3, 3], k_local[9, 9] = C4, C4  # Torsion
    k_local[4, 4], k_local[10, 10] = C9, C9 # Moment Ry
    k_local[5, 5], k_local[11, 11] = C5, C5 # Moment Rz
    
    # Cross coupling & remote terms (i-j)
    k_local[0, 6], k_local[6, 0] = -C1, -C1
    k_local[3, 9], k_local[9, 3] = -C4, -C4

    k_local[1, 7], k_local[7, 1] = -C2, -C2
    k_local[2, 8], k_local[8, 2] = -C7, -C7
    
    # Shear-Moment (V-Rz, W-Ry)
    k_local[1, 5], k_local[5, 1] = C3, C3
    k_local[1, 11], k_local[11, 1] = C3, C3
    k_local[7, 5], k_local[5, 7] = -C3, -C3
    k_local[7, 11], k_local[11, 7] = -C3, -C3
    
    k_local[2, 4], k_local[4, 2] = -C8, -C8
    k_local[2, 10], k_local[10, 2] = -C8, -C8
    k_local[8, 4], k_local[4, 8] = C8, C8
    k_local[8, 10], k_local[10, 8] = C8, C8
    
    # Moment-Moment (Rz-Rz, Ry-Ry)
    k_local[5, 11], k_local[11, 5] = C6, C6
    k_local[4, 10], k_local[10, 4] = C10, C10
    
    # 3. Transformation Matrix (T) - 12x12
    T_3x3 = np.array([x_prime_vec, y_prime_vec, z_prime_vec])
    
    T = np.zeros((12, 12))
    for i in range(4):
        T[i*3:i*3+3, i*3:i*3+3] = T_3x3
        
    # 4. Global Stiffness Matrix (k_global)
    k_global = T.T @ k_local @ T
    
    return k_global

def assemble_global_stiffness(nodes_df, elements, node_coords):
    """
    Assembles the Global Stiffness Matrix [K] by summing the contributions 
    of each element's global stiffness matrix.
    """
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    
    K_global = np.zeros((total_dofs, total_dofs))
    
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()

    for elem in elements:
        # Calculate element's stiffness matrix in global coordinates using its unique properties
        k_global_elem = calculate_element_stiffness(elem, node_coords)
        
        # Get global DoF indices for the element's start and end nodes
        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        
        # Map the 12 element DOFs to the global DOFs
        global_dofs = np.concatenate((np.arange(dof_start_i, dof_start_i + 6), 
                                      np.arange(dof_start_j, dof_start_j + 6)))
        
        # Assemble the element stiffness into the global matrix K
        for row_elem in range(12):
            for col_elem in range(12):
                global_row = global_dofs[row_elem]
                global_col = global_dofs[col_elem]
                
                K_global[global_row, global_col] += k_global_elem[row_elem, col_elem]
    
    return K_global

def perform_analysis(nodes_df, elements, load_params, node_coords):
    """
    Performs FEA setup (DOF mapping, stiffness assembly) and mocks the results.
    """
    st.info("Initiating 3D Finite Element Analysis (FEA) Setup...")
    
    # 1. Map Degrees of Freedom (DOF)
    nodes_df = map_degrees_of_freedom(nodes_df)
    
    # 2. Element Stiffness Matrices
    # Use a typical element (e.g., the first beam) for size check
    stiffness_matrix_placeholder = calculate_element_stiffness(elements[0], node_coords)
    st.markdown(f"**Element Stiffness Matrix Size:** `{stiffness_matrix_placeholder.shape}` (12x12 for 3D element)")
    
    # 3. Assemble Global Stiffness Matrix (K)
    K_global = assemble_global_stiffness(nodes_df, elements, node_coords)
    st.markdown(f"**Global Stiffness Matrix Size (K):** `{K_global.shape}` (Total DoF x Total DoF)")
    
    st.session_state['K_global'] = K_global

    # --- Next FEA Solver Steps (Currently Placeholder) ---
    st.warning("FEA Setup Complete. The next steps are Load Vector [P] and Boundary Conditions.")
    st.info("Falling back to Mock Analysis results for visualization.")
    
    # Mocking results (same as before to keep visualization working)
    for elem in elements:
        if 'beam' in elem['type']:
            elem['BM'] = np.random.uniform(100, 300) 
            elem['SF'] = np.random.uniform(50, 150)
        elif 'column' in elem['type']:
            elem['BM'] = np.random.uniform(50, 100) 
            elem['SF'] = np.random.uniform(20, 50) 
            
    max_y = nodes_df['y'].max()
    if max_y > 0:
        nodes_df['deflection_scale'] = nodes_df['y'] / max_y
        nodes_df['deflection_magnitude'] = nodes_df['deflection_scale'] * 0.5 
    
    return nodes_df, elements

# --- 4. Plotly Visualization Function (Unchanged) ---
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
        calculated_widths = [] 
        
        for elem in elems:
            p1 = node_coords[elem['start']]
            p2 = node_coords[elem['end']]
            
            x_coords.extend([p1[0], p2[0], None])
            y_coords.extend([p1[1], p2[1], None])
            z_coords.extend([p1[2], p2[2], None])
            
            width = base_width
            if result_key and elem[result_key] > 0:
                width += elem[result_key] * line_width_scale
            calculated_widths.append(width) 


        final_line_width = base_width
        if calculated_widths and result_key:
            final_line_width = min(sum(calculated_widths) / len(calculated_widths), 15)
        
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(
                color=line_color if result_key else type_colors[elem_type], 
                width=final_line_width 
            ),
            name=elem_type.replace('-', ' ').title() + trace_name_suffix,
            hoverinfo='none'
        ))

    # --- 3. Layout Configuration ---
    max_x = nodes_df['x'].max() if not nodes_df.empty else 10
    max_y = nodes_df['y'].max() if not nodes_df.empty else 10
    max_z = nodes_df['z'].max() if not nodes_df.empty else 10
    
    min_y = nodes_df['y'].min() if not nodes_df.empty else 0
    
    plot_limit_x = max_x * 1.1 
    plot_limit_z = max_z * 1.1 
    plot_limit_y_top = max_y * 1.1
    plot_limit_y_bottom = min_y * 1.1 
    
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

# --- 5. Streamlit App Layout and Logic ---

# Sidebar for User Inputs
st.sidebar.header("📐 Define Grid Geometry")
st.sidebar.markdown("Use format: `L1, N2xL2, ...` (e.g., `5, 2x4.5`). Units are assumed (e.g., meters).")

# Grid Inputs (Unchanged)
default_x = "3x5, 2x4"
x_input = st.sidebar.text_input("X Direction Bays (Length)", default_x)
default_y = "4.5, 3x3.5"
y_input = st.sidebar.text_input("Y Direction Heights (Floor)", default_y)
default_z = "4x6"
z_input = st.sidebar.text_input("Z Direction Bays (Width)", default_z)
default_foundation = 3.0
foundation_depth = st.sidebar.number_input(
    "Foundation Depth below Ground (m)", 
    min_value=0.0, 
    value=default_foundation, 
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.header("🔩 Material & Section Properties (RC)")

# Material Property
E_val = st.sidebar.number_input("Young's Modulus E (kN/m²)", value=2.5e7, format="%e", help="E for M25 concrete is ~2.5e7 kN/m²")

st.sidebar.subheader("Column Dimensions (b x h, mm)")
col_b = st.sidebar.number_input("Column Width b (mm)", value=300, min_value=100)
col_h = st.sidebar.number_input("Column Depth h (mm)", value=600, min_value=100)

st.sidebar.subheader("Beam Dimensions (b x h, mm)")
beam_b = st.sidebar.number_input("Beam Width b (mm)", value=230, min_value=100)
beam_h = st.sidebar.number_input("Beam Depth h (mm)", value=450, min_value=100)

# Calculate derived properties
prop_column = calculate_rc_properties(col_b, col_h, E_val)
prop_beam = calculate_rc_properties(beam_b, beam_h, E_val)
G_val = prop_column['G'] # G is the same for both if E is the same

st.sidebar.markdown(f"**Derived Shear Modulus G:** `{G_val:.2e}` kN/m²")

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
The application now handles **separate cross-sections for beams and columns**, automatically calculating $A, I_{zz}, I_{yy}, J$ 
and assembling the Global Stiffness Matrix $\mathbf{K}$ using element-specific properties.
""")

# --- 6. Main Logic Block ---

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
        # 3. Generate Geometry (Pass calculated properties)
        with st.spinner('Generating 3D geometry and supports...'):
            nodes_df, elements, node_coords = generate_grid_geometry(
                x_lengths, y_heights, z_lengths, foundation_depth, prop_column, prop_beam
            )
        
        # 4. Perform Full FEA Setup
        with st.spinner('Calculating element stiffnesses and assembling Global Stiffness Matrix K...'):
            nodes_df, elements = perform_analysis(nodes_df, elements, load_params, node_coords)
        
        st.success(f"FEA Setup Complete: {len(nodes_df)} nodes and {len(elements)} elements. Global Stiffness Matrix K is assembled.")

        # 5. Visualization
        frame_fig = plot_3d_frame(nodes_df, elements, node_coords, display_mode)
        st.plotly_chart(frame_fig, use_container_width=True)

        # 6. Display Summary Data
        st.subheader("FEA Setup Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Total Nodes:** `{len(nodes_df)}`")
            st.markdown(f"**Total Elements:** `{len(elements)}`")
        with col2:
            st.markdown(f"**Global Matrix Size:** `{st.session_state['K_global'].shape}`")
            st.markdown(f"**Total DoF:** `{nodes_df.attrs.get('total_dofs', 0)}`")
        with col3:
            st.markdown(f"**E:** `{E_val:.2e}` kN/m²")
            st.markdown(f"**G:** `{G_val:.2e}` kN/m²")

        st.subheader("Section Property Details (Used in Stiffness Matrix)")
        
        st.markdown("**Column Properties (b: $300mm \times$ h: $600mm$)**")
        st.dataframe(pd.DataFrame([prop_column]).drop(columns=['E', 'G', 'b_mm', 'h_mm']).T.rename(columns={0: 'Value'}))

        st.markdown("**Beam Properties (b: $230mm \times$ h: $450mm$)**")
        st.dataframe(pd.DataFrame([prop_beam]).drop(columns=['E', 'G', 'b_mm', 'h_mm']).T.rename(columns={0: 'Value'}))

        # Reset flag
        st.session_state['run_generation'] = False
