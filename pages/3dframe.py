import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="3D Streamlit Frame Generator")

# --- 1. Utility Functions ---

def parse_grid_input(input_string):
    """Parses flexible grid input strings (e.g., "3x5, 2x4.5, 5") into an array of lengths."""
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
                pass 
    return lengths

def calculate_rc_properties(b, h, E, nu=0.2):
    """Calculates 3D structural properties (A, Izz, Iyy, J, G) for a rectangular section b x h."""
    b_val = b / 1000.0  # Convert mm to m
    h_val = h / 1000.0  # Convert mm to m
    
    A = b_val * h_val
    Izz = b_val * h_val**3 / 12.0 # Major bending (about axis parallel to b)
    Iyy = h_val * b_val**3 / 12.0 # Minor bending (about axis parallel to h)
    
    # Torsional constant J approximation for rectangular section
    b_long = max(b_val, h_val)
    h_short = min(b_val, h_val)
    k2 = 1/3 * (1 - 0.63 * (h_short/b_long) * (1 - h_short**4 / (12*b_long**4)))
    J = k2 * b_long * h_short**3
    
    G = E / (2 * (1 + nu))

    return {
        'A': A, 'Izz': Izz, 'Iyy': Iyy, 'J': J, 'E': E, 'G': G, 
        'b_m': b_val, 'h_m': h_val
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

    # 1. Generate Nodes 
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                node = {
                    'id': node_id_counter, 'x': cum_x[i], 'y': cum_y[j], 'z': cum_z[k],
                    'support_type': 'Fixed' if j == foundation_base_index else 'None',
                    'u': np.zeros(6) # Displacement vector [u, v, w, rx, ry, rz]
                }
                nodes.append(node)
                node_grid_map[(i, j, k)] = node_id_counter
                node_id_counter += 1
    
    nodes_df = pd.DataFrame(nodes)
    node_coords = {n['id']: (n['x'], n['y'], n['z']) for n in nodes}
    nodes_df.attrs['grid_map'] = node_grid_map
    nodes_df.attrs['x_lengths'] = x_lengths
    nodes_df.attrs['z_lengths'] = z_lengths

    # 2. Generate Elements (Beams/Columns) and assign properties
    element_id_counter = 1
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                current_id = node_grid_map[(i, j, k)]

                # Connection in X direction (Beams) - Only connect beams from ground floor (j=1) upwards
                if i < x_grid_count - 1 and j >= 1: 
                    neighbor_id = node_grid_map[(i + 1, j, k)]
                    L = x_lengths[i]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-x', 'L': L,
                        'i': i, 'j': j, 'k': k, **prop_beam
                    })
                    element_id_counter += 1

                # Connection in Y direction (Columns)
                if j < y_grid_count - 1:
                    neighbor_id = node_grid_map[(i, j + 1, k)]
                    L = abs(cum_y[j+1] - cum_y[j])
                    column_type = 'foundation-column' if j == 0 else 'column'
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': column_type, 'L': L,
                        'i': i, 'j': j, 'k': k, **prop_column
                    })
                    element_id_counter += 1

                # Connection in Z direction (Beams) - Only connect beams from ground floor (j=1) upwards
                if k < z_grid_count - 1 and j >= 1: 
                    neighbor_id = node_grid_map[(i, j, k + 1)]
                    L = z_lengths[k]
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-z', 'L': L,
                        'i': i, 'j': j, 'k': k, **prop_beam
                    })
                    element_id_counter += 1

    return nodes_df, elements, node_coords

# --- 3. FEA Setup Functions (Stiffness Matrix) ---

def map_degrees_of_freedom(nodes_df):
    """Assigns global degrees of freedom (DOF) indices to each node."""
    dof_per_node = 6
    total_dofs = len(nodes_df) * dof_per_node
    
    nodes_df['dof_start_index'] = nodes_df.index * dof_per_node
    nodes_df.attrs['total_dofs'] = total_dofs
    
    st.sidebar.markdown(f"**Total Global DoF:** `{total_dofs}`") # Moved to sidebar for compact info
    return nodes_df

def calculate_element_stiffness(element, node_coords):
    """
    Calculates the 12x12 element stiffness matrix [k] in global coordinates 
    using element-specific properties (E, G, A, Izz, Iyy, J).
    """
    E = element['E']
    G = element['G']
    A = element['A']
    Izz = element['Izz'] 
    Iyy = element['Iyy'] 
    J = element['J']     

    # 1. Geometry and Direction Cosines
    xi, yi, zi = node_coords[element['start']]
    xj, yj, zj = node_coords[element['end']]
    
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
    if L == 0:
        return np.zeros((12, 12))

    l = (xj - xi) / L
    m = (yj - yi) / L
    n = (zj - zi) / L

    x_prime_vec = np.array([l, m, n])
    
    if np.isclose(l**2 + n**2, 0): # Element is vertical (parallel to global Y)
        V_ref = np.array([0, 0, 1]) 
    else:
        V_ref = np.array([0, 1, 0]) 

    z_prime_vec = np.cross(x_prime_vec, V_ref)
    norm_z = np.linalg.norm(z_prime_vec)
    z_prime_vec = z_prime_vec / norm_z if norm_z > 1e-6 else np.array([0, 0, 0])
    
    y_prime_vec = np.cross(z_prime_vec, x_prime_vec)

    # 2. Local Stiffness Matrix (k_local) - 12x12
    k_local = np.zeros((12, 12))
    
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
    k_local[0, 0], k_local[6, 6] = C1, C1  
    k_local[1, 1], k_local[7, 7] = C2, C2  
    k_local[2, 2], k_local[8, 8] = C7, C7  
    k_local[3, 3], k_local[9, 9] = C4, C4  
    k_local[4, 4], k_local[10, 10] = C9, C9 
    k_local[5, 5], k_local[11, 11] = C5, C5 
    
    k_local[0, 6], k_local[6, 0] = -C1, -C1
    k_local[3, 9], k_local[9, 3] = -C4, -C4

    k_local[1, 7], k_local[7, 1] = -C2, -C2
    k_local[2, 8], k_local[8, 2] = -C7, -C7
    
    k_local[1, 5], k_local[5, 1] = C3, C3
    k_local[1, 11], k_local[11, 1] = C3, C3
    k_local[7, 5], k_local[5, 7] = -C3, -C3
    k_local[7, 11], k_local[11, 7] = -C3, -C3
    
    k_local[2, 4], k_local[4, 2] = -C8, -C8
    k_local[2, 10], k_local[10, 2] = -C8, -C8
    k_local[8, 4], k_local[4, 8] = C8, C8
    k_local[8, 10], k_local[10, 8] = C8, C8
    
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
    """Assembles the Global Stiffness Matrix [K]."""
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    K_global = np.zeros((total_dofs, total_dofs))
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()

    for elem in elements:
        k_global_elem = calculate_element_stiffness(elem, node_coords)
        
        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        
        global_dofs = np.concatenate((np.arange(dof_start_i, dof_start_i + 6), 
                                      np.arange(dof_start_j, dof_start_j + 6)))
        
        for row_elem in range(12):
            for col_elem in range(12):
                global_row = global_dofs[row_elem]
                global_col = global_dofs[col_elem]
                K_global[global_row, global_col] += k_global_elem[row_elem, col_elem]
    
    return K_global

# --- 4. Load Calculation & Vector Assembly ---

def calculate_gravity_loads(nodes_df, elements, load_params, node_coords):
    """
    Calculates element fixed-end forces (FEFs) from gravity loads (Self-weight + Slab Load).
    Assumes gravity acts in the negative Y direction.
    """
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    if total_dofs == 0:
        return np.zeros(0)
        
    P_global = np.zeros(total_dofs)
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()
    x_lengths = nodes_df.attrs['x_lengths']
    z_lengths = nodes_df.attrs['z_lengths']
    
    # 1. Total Floor Area Load (q_floor)
    concrete_density = load_params['slab_density']
    slab_thickness = load_params['slab_thickness']
    finish_load = load_params['finish_load']
    live_load = load_params['live_load']
    
    q_slab_dl = concrete_density * slab_thickness + finish_load
    q_total_ll = live_load
    q_total_gravity = q_slab_dl + q_total_ll # kN/m^2 (Unfactored)

    # Corrected LaTeX unit formatting:
    st.markdown(f"**Calculated Gravity Floor Pressure (Unfactored):** $ {q_total_gravity:.2f} \\frac{{kN}}{{m^2}} $")

    for elem in elements:
        P_elem = np.zeros(12)
        L = elem['L']
        
        if 'column' in elem['type']:
            # Column Self-Weight (Axial load in Y direction)
            w_sw = concrete_density * elem['A']
            # Only 50% of the UDL load is applied to the end nodes as point load in the local axial direction
            P_elem[0] = -w_sw * L / 2.0  # Force at start node (Local u)
            P_elem[6] = -w_sw * L / 2.0  # Force at end node (Local u)
            
        elif 'beam' in elem['type']:
            # Beam Self-Weight (UDL in global Y, which is local Z' for a beam lying in the XZ plane)
            w_beam_sw = concrete_density * elem['A']
            
            # Tributary Area Load from Slab (UDL in global Y, local Z')
            if 'x' in elem['type'] and elem['k'] < len(z_lengths):
                # Beam in X direction (tributary width is Lz / 2)
                tributary_width = z_lengths[elem['k']] / 2.0
            elif 'z' in elem['type'] and elem['i'] < len(x_lengths):
                # Beam in Z direction (tributary width is Lx / 2)
                tributary_width = x_lengths[elem['i']] / 2.0
            else:
                tributary_width = 0 

            w_slab = q_total_gravity * tributary_width
            w_total = w_beam_sw + w_slab # Total UDL on beam, acts in -Global Y direction

            # Convert UDL w_total into Fixed-End Actions (FEFs) for bending about local Y' (Minor Axis)
            # UDL (w) acts in the -Global Y direction. For horizontal beams, this is along the local Z' axis.
            F_local_v = w_total * L / 2.0  # Forces
            M_local_rz = w_total * L**2 / 12.0 # Moments

            # Apply FEF components in local coordinates:
            P_elem[2] = -F_local_v  # Shear Force at start (P2)
            P_elem[8] = -F_local_v  # Shear Force at end (P8)
            P_elem[4] = -M_local_rz # Moment at start (P4)
            P_elem[10] = M_local_rz # Moment at end (P10)


        # Transform P_elem (local) to P_global_elem (global) using the transformation matrix T
        
        # 1. Geometry and Direction Cosines (repeated from stiffness calc to get T)
        xi, yi, zi = node_coords[elem['start']]
        xj, yj, zj = node_coords[elem['end']]
        
        l = (xj - xi) / L
        m = (yj - yi) / L
        n = (zj - zi) / L

        x_prime_vec = np.array([l, m, n])
        if np.isclose(l**2 + n**2, 0): V_ref = np.array([0, 0, 1]) 
        else: V_ref = np.array([0, 1, 0]) 

        z_prime_vec = np.cross(x_prime_vec, V_ref)
        norm_z = np.linalg.norm(z_prime_vec)
        z_prime_vec = z_prime_vec / norm_z if norm_z > 1e-6 else np.array([0, 0, 0])
        y_prime_vec = np.cross(z_prime_vec, x_prime_vec)
        T_3x3 = np.array([x_prime_vec, y_prime_vec, z_prime_vec])
        
        T = np.zeros((12, 12))
        for i in range(4): T[i*3:i*3+3, i*3:i*3+3] = T_3x3

        P_global_elem = T.T @ P_elem # Transform fixed-end force vector (reaction)

        # Apply to global P vector (P_nodal = -P_fixed_end_actions)
        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        global_dofs = np.concatenate((np.arange(dof_start_i, dof_start_i + 6), 
                                      np.arange(dof_start_j, dof_start_j + 6)))
        
        # We are adding the negative of the FEF into P_global to represent the equivalent nodal load
        for i, global_dof in enumerate(global_dofs):
            P_global[global_dof] += P_global_elem[i]

    return P_global

# --- 5. FEA Solver and Displacement Calculation (Updated) ---

def solve_fea_system(nodes_df, K_global, P_global):
    """
    Applies boundary conditions (Fixed Support) and solves the system K*U = P for displacements U.
    """
    total_dofs = K_global.shape[0]
    
    # 1. Identify Constrained (Known) DoFs
    known_dofs = [] 
    
    for _, row in nodes_df.iterrows():
        if row['support_type'] == 'Fixed':
            start_index = row['dof_start_index']
            known_dofs.extend(range(start_index, start_index + 6))

    # 2. Identify Unknown (Free) DoFs
    all_dofs = np.arange(total_dofs)
    unknown_dofs = np.setdiff1d(all_dofs, known_dofs)
    
    if len(unknown_dofs) == 0:
        st.error("No free degrees of freedom found. Structure is overly constrained.")
        return nodes_df.copy(), np.zeros(total_dofs)

    # 3. Partition Matrices (Reduction)
    K_uu = K_global[np.ix_(unknown_dofs, unknown_dofs)]
    P_u = P_global[unknown_dofs]

    # 4. Solve for Unknown Displacements (U_u)
    try:
        # Check for singularity before inversion
        if np.linalg.det(K_uu) < 1e-9:
             st.error("Reduced Stiffness Matrix is singular. Check structure stability (mechanisms).")
             return nodes_df.copy(), np.zeros(total_dofs)
             
        U_u = np.linalg.solve(K_uu, P_u)
    except np.linalg.LinAlgError as e:
        st.error(f"Linear algebra error during solution: {e}")
        return nodes_df.copy(), np.zeros(total_dofs)

    # 5. Assemble Full Displacement Vector (U)
    U_full = np.zeros(total_dofs)
    U_full[unknown_dofs] = U_u
    
    # 6. Update Nodes DataFrame with Results
    temp_nodes_df = nodes_df.copy() # Work on a copy to apply results back
    
    # FIX: Explicitly ensure the 'u' column is of object dtype to reliably store numpy arrays.
    # This addresses the ValueError when assigning an array to a single cell.
    temp_nodes_df['u'] = temp_nodes_df['u'].astype(object) 
    
    for index, row in temp_nodes_df.iterrows():
        start_index = row['dof_start_index']
        # Use .at for robust assignment of a complex type (numpy array) to a single cell.
        temp_nodes_df.at[index, 'u'] = U_full[start_index : start_index + 6]

    return temp_nodes_df, U_full

def perform_analysis(nodes_df, elements, load_params, node_coords):
    """
    Performs full FEA setup, solves for displacement under gravity, and computes mock internal forces.
    """
    
    # 1. Map Degrees of Freedom (DOF)
    nodes_df = map_degrees_of_freedom(nodes_df)
    
    # 2. Assemble Global Stiffness Matrix (K)
    K_global = assemble_global_stiffness(nodes_df, elements, node_coords)
    st.session_state['K_global'] = K_global
    
    # 3. Generate Global Load Vector (P)
    P_global = calculate_gravity_loads(nodes_df, elements, load_params, node_coords)
    
    # 4. Solve System for Displacement (U)
    with st.spinner('Solving for displacement vector U...'):
        # nodes_df_solved is the node dataframe with the 'u' vector updated
        nodes_df_solved, U_full = solve_fea_system(nodes_df, K_global, P_global) 
    
    # 5. Calculate Internal Forces (Simplified Mocking for Visualization)
    max_deflection = np.max(np.abs(U_full[np.arange(1, len(U_full), 6)])) # Max V displacement
    
    if max_deflection == 0:
        st.warning("Calculated displacement is zero. Check loads/boundary conditions.")
        
    for index, row in nodes_df_solved.iterrows():
        # Ensure 'u' is treated as an array before using np.linalg.norm
        u_vector = row['u']
        nodes_df_solved.loc[index, 'deflection_magnitude'] = np.linalg.norm(u_vector[0:3]) 
    
    # Mock internal forces (Aesthetic only - full back-transformation required for real forces)
    concrete_density = load_params['slab_density']
    q_total_gravity = (concrete_density * load_params['slab_thickness'] + load_params['finish_load'] + load_params['live_load'])

    for elem in elements:
        if 'beam' in elem['type']:
            L = elem['L']
            mock_force_base = 0.5 * q_total_gravity * L**2 * 0.1 
            elem['BM'] = np.random.uniform(mock_force_base * 0.8, mock_force_base * 1.2)
            elem['SF'] = np.random.uniform(mock_force_base * 0.5, mock_force_base * 0.8)
        elif 'column' in elem['type']:
            L = elem['L']
            w_col_sw = concrete_density * elem['A']
            mock_force_base = w_col_sw * L * 0.5 
            elem['BM'] = np.random.uniform(mock_force_base * 0.1, mock_force_base * 0.2) 
            elem['SF'] = np.random.uniform(mock_force_base * 0.05, mock_force_base * 0.1) 
            
    st.success(f"FEA Solution Complete. Maximum Y Displacement (Deflection) is **${max_deflection * 1000:.2f} \\text{ mm}$** (Under unfactored gravity load).")
    st.info("Bending Moment (BM) and Shear Force (SF) displayed are conceptual approximations for visualization only.")

    return nodes_df_solved, elements

# --- 6. Plotly Visualization Function ---
def plot_3d_frame(nodes_df, elements, node_coords, display_mode):
    """
    Creates an interactive 3D Plotly figure for the frame structure,
    with options to show Deflection, BM, or SF. Line width is proportional to member depth (h_m).
    """
    fig = go.Figure()

    # Determine plot properties based on display mode
    if display_mode == 'Bending Moment':
        trace_name_suffix = ' (BM)'
        result_key = 'BM'
        line_color_mode = 'red'
        force_width_scale = 0.05
    elif display_mode == 'Shear Force':
        trace_name_suffix = ' (SF)'
        result_key = 'SF'
        line_color_mode = 'blue'
        force_width_scale = 0.05
    else: # Default: Deflection or Geometry
        trace_name_suffix = ''
        result_key = None
        line_color_mode = None
        
    # Scale factor for displacement visualization
    scale_factor = 200 
    
    nodes_df['x_plot'] = nodes_df.apply(lambda row: row['x'] + row['u'][0] * scale_factor if 'u' in row else row['x'], axis=1)
    nodes_df['y_plot'] = nodes_df.apply(lambda row: row['y'] + row['u'][1] * scale_factor if 'u' in row else row['y'], axis=1)
    nodes_df['z_plot'] = nodes_df.apply(lambda row: row['z'] + row['u'][2] * scale_factor if 'u' in row else row['z'], axis=1)

    if display_mode in ['Geometry/Supports', 'Bending Moment', 'Shear Force']:
        # Show undeformed geometry for force diagrams
        nodes_df['x_plot'] = nodes_df['x']
        nodes_df['y_plot'] = nodes_df['y']
        nodes_df['z_plot'] = nodes_df['z']
        
    # 1. Plot Nodes 
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
        text=[
            f"Node ID: {row['id']}<br>Coords: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})<br>"
            f"Disp (mm): U={row['u'][0]*1000:.2f}, V={row['u'][1]*1000:.2f}, W={row['u'][2]*1000:.2f}<br>"
            f"Support: {row['support_type']}"
            for index, row in nodes_df.iterrows()
        ],
        name='Nodes / Supports'
    ))

    # 2. Plot Elements (Line Segments)
    type_colors = {'foundation-column': 'darkorange', 'column': 'darkorchid', 'beam-x': 'royalblue', 'beam-z': 'darkcyan'}
    
    # Calculate global max h for normalization (optional, but good for consistent visual scale)
    max_h = max(elem.get('h_m', 0.5) for elem in elements) * 1000 # Max depth in mm
    
    for elem_type, elems in pd.DataFrame(elements).groupby('type'):
        x_coords, y_coords, z_coords = [], [], []
        line_widths = []
        
        for index, elem in elems.iterrows():
            p1_row = nodes_df[nodes_df['id'] == elem['start']].iloc[0]
            p2_row = nodes_df[nodes_df['id'] == elem['end']].iloc[0]

            p1_plot = (p1_row['x_plot'], p1_row['y_plot'], p1_row['z_plot'])
            p2_plot = (p2_row['x_plot'], p2_row['y_plot'], p2_row['z_plot'])
            
            x_coords.extend([p1_plot[0], p2_plot[0], None])
            y_coords.extend([p1_plot[1], p2_plot[1], None])
            z_coords.extend([p1_plot[2], p2_plot[2], None])
            
            # --- Realistic Visualization based on member size ---
            h_mm = elem.get('h_m', 0.5) * 1000 # Depth in mm
            
            # Base width proportional to h (e.g., scale 1mm depth to 0.02 Plotly width)
            # Clamped between a min (2) and a max (15) width
            base_width = np.clip(h_mm * 0.02, 2, 15) 
            
            width = base_width
            if result_key:
                # Add force magnitude to the width
                width += np.abs(elem[result_key]) * force_width_scale
            
            line_widths.extend([width, width, None]) 

        # Create a trace for this element type
        # Plotly doesn't handle varying widths in a single Scatter3d trace easily.
        # We need to create a trace per element to get force visualization, 
        # but for clean sizing, we'll plot all of one type together with an average/representative width.
        # For simplicity in this display, we will use the calculated average width for the trace.
        
        representative_width = np.mean([w for w in line_widths if w is not None]) if line_widths else 5
        
        # Plot the members: iterate over lines instead of collecting all points
        for index, elem in elems.iterrows():
            p1_row = nodes_df[nodes_df['id'] == elem['start']].iloc[0]
            p2_row = nodes_df[nodes_df['id'] == elem['end']].iloc[0]
            
            p1_plot = (p1_row['x_plot'], p1_row['y_plot'], p1_row['z_plot'])
            p2_plot = (p2_row['x_plot'], p2_row['y_plot'], p2_row['z_plot'])
            
            h_mm = elem.get('h_m', 0.5) * 1000
            line_width = np.clip(h_mm * 0.02, 2, 15) # Base width
            
            current_color = line_color_mode if line_color_mode else type_colors[elem_type]
            
            if result_key:
                 line_width += np.abs(elem[result_key]) * force_width_scale
                 
            fig.add_trace(go.Scatter3d(
                x=[p1_plot[0], p2_plot[0]],
                y=[p1_plot[1], p2_plot[1]],
                z=[p1_plot[2], p2_plot[2]],
                mode='lines',
                line=dict(
                    color=current_color, 
                    width=line_width 
                ),
                name=None, # Hide individual names
                hoverinfo='text',
                showlegend=(index == 0), # Only show legend entry once per type
                text=f"Element ID: {elem['id']}<br>Type: {elem_type}<br>L: {elem['L']:.2f} m<br>H: {h_mm:.0f} mm<br>" + 
                     (f"BM: {elem['BM']:.2f} kNm" if 'BM' in elem else "") +
                     (f"SF: {elem['SF']:.2f} kN" if 'SF' in elem else "")
            ))
            
    # 3. Layout Configuration 
    max_x = nodes_df['x'].max() if not nodes_df.empty else 10
    max_y = nodes_df['y'].max() if not nodes_df.empty else 10
    max_z = nodes_df['z'].max() if not nodes_df.empty else 10
    min_y = nodes_df['y'].min() if not nodes_df.empty else 0
    
    plot_limit_x = max_x * 1.1 
    plot_limit_z = max_z * 1.1 
    plot_limit_y_top = max_y * 1.1
    plot_limit_y_bottom = min_y * 1.1 
    
    fig.update_layout(
        title=f'Interactive 3D Frame Geometry (FEA Displacement $\\times{scale_factor}$)',
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


# --- 7. Streamlit App Layout and Logic ---

# Sidebar for User Inputs
st.sidebar.header("📐 Define Grid Geometry")
st.sidebar.markdown("Use format: `L1, N2xL2, ...` (e.g., `5, 2x4.5`). Units are assumed (e.g., meters).")

# Grid Inputs 
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
concrete_density = 25.0 # Fixed for calculation
E_val = st.sidebar.number_input("Young's Modulus E (kN/m²)", value=2.5e7, format="%e", help="E for M25 concrete is ~2.5e7 kN/m²")

st.sidebar.subheader("Column Dimensions ($b \times h$, mm)")
col_b = st.sidebar.number_input("Column Width b (mm)", value=300, min_value=100)
col_h = st.sidebar.number_input("Column Depth h (mm)", value=600, min_value=100)

st.sidebar.subheader("Beam Dimensions ($b \times h$, mm)")
beam_b = st.sidebar.number_input("Beam Width b (mm)", value=230, min_value=100)
beam_h = st.sidebar.number_input("Beam Depth h (mm)", value=450, min_value=100)

# Calculate derived properties
prop_column = calculate_rc_properties(col_b, col_h, E_val)
prop_beam = calculate_rc_properties(beam_b, beam_h, E_val)
G_val = prop_column['G'] 

# Corrected LaTeX unit formatting:
st.sidebar.markdown(f"**Derived Shear Modulus G:** $ {G_val:.2e} \\frac{{kN}}{{m^2}} $")

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Gravity Load Inputs")

load_params = {}
load_params['slab_thickness'] = st.sidebar.number_input("Slab Thickness $t$ (m)", min_value=0.0, value=0.15, step=0.01)
load_params['slab_density'] = st.sidebar.number_input("Concrete Density $\gamma$ (kN/m³)", min_value=1.0, value=concrete_density, step=1.0)
load_params['finish_load'] = st.sidebar.number_input("Floor Finish Load (kN/m²)", min_value=0.0, value=1.5, step=0.1)
load_params['live_load'] = st.sidebar.number_input("Live Load (kN/m²)", min_value=0.0, value=3.0, step=0.5)

st.sidebar.markdown("---")

# Mandatory button to trigger analysis (Moved up to ensure visibility)
if st.sidebar.button("Run Analysis & Visualize 🚀"):
    st.session_state['run_generation'] = True

st.sidebar.header("📊 Visualization Mode")
display_mode = st.sidebar.selectbox(
    "Select Result to Display",
    ['Deflection', 'Geometry/Supports', 'Bending Moment', 'Shear Force']
)

# Initialize state on first run
if 'run_generation' not in st.session_state:
    st.session_state['run_generation'] = False 
if 'nodes_df' not in st.session_state:
    st.session_state['nodes_df'] = None
if 'elements' not in st.session_state:
    st.session_state['elements'] = None
if 'node_coords' not in st.session_state:
    st.session_state['node_coords'] = None

# Main Page Content
st.title("3D Frame Analysis: Gravity Load and Realistic Sizing")
st.markdown(r"""
The structural geometry is defined by the grid, and **member thickness in the visualization is now proportional to the input $\boldsymbol{h}$ (depth) of the beam/column** for a more realistic display.
The **Gravity Load Vector** $\mathbf{P}$ is calculated based on:
1.  **Slab Dead Load:** (Slab Thickness $\times$ Concrete Density) + Floor Finish Load (Distributed to beams via tributary area).
2.  **Live Load:** Distributed to beams via tributary area.
3.  **Self-Weight:** Column self-weight applied axially; Beam self-weight applied as UDL.

The system $\mathbf{K}\mathbf{U} = \mathbf{P}$ is solved for the displacement vector $\mathbf{U}$.
""")

# --- 8. Main Logic Block ---
if st.session_state.get('run_generation'):
    
    # 1. Parse Inputs
    x_lengths = parse_grid_input(x_input)
    y_heights = parse_grid_input(y_input)
    z_lengths = parse_grid_input(z_input)
    
    # 2. Validate & Generate Geometry
    if not x_lengths or not y_heights or not z_lengths:
        st.error("Please ensure all three dimensions (X, Y, Z) have valid, positive length inputs.")
        st.session_state['run_generation'] = False
    else:
        with st.spinner('Generating 3D geometry and assigning properties...'):
            nodes_df, elements, node_coords = generate_grid_geometry(
                x_lengths, y_heights, z_lengths, foundation_depth, prop_column, prop_beam
            )
            st.session_state['nodes_df'] = nodes_df
            st.session_state['elements'] = elements
            st.session_state['node_coords'] = node_coords
        
        # 3. Perform Full FEA Setup and Solve
        with st.spinner('Calculating K and P matrices, and solving for U...'):
            nodes_df_solved, elements_solved = perform_analysis(nodes_df, elements, load_params, node_coords)
            st.session_state['nodes_df'] = nodes_df_solved
            st.session_state['elements'] = elements_solved
        
    # Reset run trigger
    st.session_state['run_generation'] = False

# --- 9. Display Visualization if data exists ---
if st.session_state['nodes_df'] is not None:
    frame_fig = plot_3d_frame(
        st.session_state['nodes_df'], 
        st.session_state['elements'], 
        st.session_state['node_coords'], 
        display_mode
    )
    st.plotly_chart(frame_fig, use_container_width=True)

    # Display Summary Data
    st.subheader("FEA System Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Total Nodes:** `{len(st.session_state['nodes_df'])}`")
        st.markdown(f"**Total Elements:** `{len(st.session_state['elements'])}`")
    with col2:
        st.markdown(f"**Global Matrix Size (K):** `{st.session_state['K_global'].shape if 'K_global' in st.session_state else 'N/A'}`")
    with col3:
        # Corrected LaTeX unit formatting:
        st.markdown(f"**E:** $ {E_val:.2e} \\frac{{kN}}{{m^2}} $")
        
    st.subheader("Section Property Details")
    
    col_prop = pd.DataFrame([prop_column]).drop(columns=['E', 'G']).T.rename(index={'b_m': 'Width $b$ (m)', 'h_m': 'Depth $h$ (m)'}).rename(columns={0: 'Value'})
    beam_prop = pd.DataFrame([prop_beam]).drop(columns=['E', 'G']).T.rename(index={'b_m': 'Width $b$ (m)', 'h_m': 'Depth $h$ (m)'}).rename(columns={0: 'Value'})

    col1_prop, col2_prop = st.columns(2)
    with col1_prop:
        st.markdown("**Column Properties**")
        st.dataframe(col_prop, use_container_width=True)
    with col2_prop:
        st.markdown("**Beam Properties**")
        st.dataframe(beam_prop, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Maximum Deflection Summary")
    max_y_deflection = st.session_state['nodes_df']['u'].apply(lambda x: x[1]).abs().max() * 1000
    st.metric("Max Vertical Displacement (V) in mm", f"{max_y_deflection:.4f}")
