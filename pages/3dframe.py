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
    Izz = b_val * h_val**3 / 12.0 # Major bending inertia (about axis parallel to b)
    Iyy = h_val * b_val**3 / 12.0 # Minor bending inertia (about axis parallel to h)
    
    # Torsional constant J approximation for rectangular section
    b_long = max(b_val, h_val)
    h_short = min(b_val, h_val)
    J = b_long * h_short**3 * (1/3 - 0.21 * (h_short/b_long) * (1 - h_short**4 / (12*b_long**4))) 
    if J <= 0: J = 0.001 * (b_val * h_val)**2 # Fallback

    G = E / (2 * (1 + nu))

    return {
        'A': A, 'Izz': Izz, 'Iyy': Iyy, 'J': J, 'E': E, 'G': G, 
        'b_m': b_val, 'h_m': h_val
    }

# --- 2. Geometry and FEA Matrix Helper Functions ---

def map_degrees_of_freedom(nodes_df):
    """Assigns global degrees of freedom (DOF) indices to each node."""
    dof_per_node = 6
    total_dofs = len(nodes_df) * dof_per_node
    
    nodes_df['dof_start_index'] = nodes_df.index * dof_per_node
    nodes_df.attrs['total_dofs'] = total_dofs
    
    st.sidebar.markdown(f"**Total Global DoF:** `{total_dofs}`") 
    return nodes_df

def _get_element_matrices(element, node_coords, load_params, x_lengths, z_lengths):
    """
    Calculates element's local stiffness matrix (k_local), transformation matrix (T), 
    and local fixed-end force vector (P_fixed_local) from gravity.
    """
    E = element['E']; G = element['G']; A = element['A']; Izz = element['Izz']; Iyy = element['Iyy']; J = element['J']     
    xi, yi, zi = node_coords[element['start']]; xj, yj, zj = node_coords[element['end']]
    
    L = np.sqrt((xj - xi)**2 + (yj - yi)**2 + (zj - zi)**2)
    if L < 1e-6: # Check for near zero length
        return np.zeros((12, 12)), np.zeros((12, 12)), np.zeros(12), 0.0, L

    l = (xj - xi) / L; m = (yj - yi) / L; n = (zj - zi) / L

    x_prime_vec = np.array([l, m, n])
    if np.isclose(l**2 + n**2, 0): V_ref = np.array([0, 0, 1]) # Vertical element
    else: V_ref = np.array([0, 1, 0]) # Horizontal element reference

    z_prime_vec = np.cross(x_prime_vec, V_ref)
    norm_z = np.linalg.norm(z_prime_vec)
    z_prime_vec = z_prime_vec / norm_z if norm_z > 1e-6 else np.array([0, 0, 0])
    y_prime_vec = np.cross(z_prime_vec, x_prime_vec)
    
    # 1. Transformation Matrix (T) - 12x12
    T_3x3 = np.array([x_prime_vec, y_prime_vec, z_prime_vec], dtype=float)
    T = np.zeros((12, 12), dtype=float)
    for i in range(4): T[i*3:i*3+3, i*3:i*3+3] = T_3x3

    # 2. Local Stiffness Matrix (k_local) - 12x12
    k_local = np.zeros((12, 12), dtype=float)
    C1 = E * A / L; C4 = G * J / L; C2 = 12 * E * Izz / (L**3); C3 = 6 * E * Izz / (L**2)
    C5 = 4 * E * Izz / L; C6 = 2 * E * Izz / L; C7 = 12 * E * Iyy / (L**3); C8 = 6 * E * Iyy / (L**2)
    C9 = 4 * E * Iyy / L; C10 = 2 * E * Iyy / L
    
    # Populate k_local (standard 3D beam element stiffness matrix)
    k_local[0, 0], k_local[6, 6] = C1, C1  ; k_local[0, 6], k_local[6, 0] = -C1, -C1
    k_local[3, 3], k_local[9, 9] = C4, C4  ; k_local[3, 9], k_local[9, 3] = -C4, -C4
    
    k_local[1, 1], k_local[7, 7] = C2, C2  ; k_local[1, 7], k_local[7, 1] = -C2, -C2
    k_local[5, 5], k_local[11, 11] = C5, C5; k_local[5, 11], k_local[11, 5] = C6, C6
    k_local[1, 5], k_local[5, 1] = C3, C3  ; k_local[1, 11], k_local[11, 1] = C3, C3
    k_local[7, 5], k_local[5, 7] = -C3, -C3; k_local[7, 11], k_local[11, 7] = -C3, -C3

    k_local[2, 2], k_local[8, 8] = C7, C7  ; k_local[2, 8], k_local[8, 2] = -C7, -C7
    k_local[4, 4], k_local[10, 10] = C9, C9; k_local[4, 10], k_local[10, 4] = C10, C10
    k_local[2, 4], k_local[4, 2] = -C8, -C8; k_local[2, 10], k_local[10, 2] = -C8, -C8
    k_local[8, 4], k_local[4, 8] = C8, C8  ; k_local[8, 10], k_local[10, 8] = C8, C8
    
    # 3. Local Fixed-End Force Vector (P_fixed_local)
    P_fixed_local = np.zeros(12, dtype=float)
    q_total_gravity = (load_params['slab_density'] * load_params['slab_thickness'] + 
                       load_params['finish_load'] + load_params['live_load'])
    w_udl = 0.0 # Total UDL magnitude (will be negative, acting downwards)
    
    if 'column' in element['type']:
        w_sw = load_params['slab_density'] * element['A']
        # Axial load in local u (DOF 0 and 6). Local x' is aligned with global Y for a vertical column.
        P_fixed_local[0] = -w_sw * L / 2.0  
        P_fixed_local[6] = -w_sw * L / 2.0
        w_udl = w_sw # Used for load visualization
        
    elif 'beam' in element['type']:
        w_beam_sw = load_params['slab_density'] * element['A']
        
        tributary_width = 0.0
        if 'x' in element['type'] and element['k'] < len(z_lengths):
            tributary_width = z_lengths[element['k']] / 2.0
        elif 'z' in element['type'] and element['i'] < len(x_lengths):
            tributary_width = x_lengths[element['i']] / 2.0

        w_slab_and_live = q_total_gravity * tributary_width
        w_total = w_beam_sw + w_slab_and_live # Total UDL magnitude
        
        # Assume gravity acts along local z' (DOF 2, 8 for shear; 5, 11 for moment) for major axis bending
        F_local_v = w_total * L / 2.0      
        M_local_rz = w_total * L**2 / 12.0 

        # Shear Forces (P2, P8)
        P_fixed_local[2] = F_local_v      
        P_fixed_local[8] = F_local_v
        # Bending Moments (P5, P11) - Note the signs for equilibrium
        P_fixed_local[5] = -M_local_rz 
        P_fixed_local[11] = M_local_rz 
        w_udl = w_total # Used for load visualization

    return k_local, T, P_fixed_local, w_udl, L

# --- 3. Geometry Generation and FEA Setup Functions (Assembly, Solve, Force) ---

def generate_grid_geometry(x_lengths, y_heights, z_lengths, foundation_depth, prop_column, prop_beam):
    """Generates the structural geometry (nodes and elements)."""
    nodes = []
    elements = []
    node_id_counter = 1

    cum_x = np.cumsum([0] + x_lengths); above_ground_heights = [0] + y_heights
    cum_y_above_ground = np.cumsum(above_ground_heights)
    cum_y = np.insert(cum_y_above_ground, 0, -foundation_depth)
    cum_z = np.cumsum([0] + z_lengths)

    x_grid_count = len(cum_x); y_grid_count = len(cum_y); z_grid_count = len(cum_z)
    ground_level_index = 1
    node_grid_map = {}

    # 1. Generate Nodes 
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                node = {'id': node_id_counter, 'x': cum_x[i], 'y': cum_y[j], 'z': cum_z[k],
                        'support_type': 'Fixed' if j == 0 else 'None',
                        'u': np.zeros(6, dtype=float)}
                nodes.append(node)
                node_grid_map[(i, j, k)] = node_id_counter
                node_id_counter += 1
    
    nodes_df = pd.DataFrame(nodes)
    node_coords = {n['id']: (n['x'], n['y'], n['z']) for n in nodes}
    nodes_df.attrs['grid_map'] = node_grid_map
    nodes_df.attrs['x_lengths'] = x_lengths
    nodes_df.attrs['z_lengths'] = z_lengths

    # 2. Generate Elements
    element_id_counter = 1
    for i in range(x_grid_count):
        for j in range(y_grid_count):
            for k in range(z_grid_count):
                current_id = node_grid_map[(i, j, k)]

                if i < x_grid_count - 1 and j >= ground_level_index: # Beam-X
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': node_grid_map[(i + 1, j, k)], 
                        'type': 'beam-x', 'i': i, 'j': j, 'k': k, **prop_beam
                    })
                    element_id_counter += 1
                if j < y_grid_count - 1: # Column-Y
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': node_grid_map[(i, j + 1, k)], 
                        'type': 'column', 'i': i, 'j': j, 'k': k, **prop_column
                    })
                    element_id_counter += 1
                if k < z_grid_count - 1 and j >= ground_level_index: # Beam-Z
                    elements.append({
                        'id': element_id_counter, 'start': current_id, 'end': node_grid_map[(i, j, k + 1)], 
                        'type': 'beam-z', 'i': i, 'j': j, 'k': k, **prop_beam
                    })
                    element_id_counter += 1

    return nodes_df, elements, node_coords

def assemble_global_stiffness(nodes_df, elements, node_coords, load_params):
    """Assembles the Global Stiffness Matrix [K]."""
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    K_global = np.zeros((total_dofs, total_dofs), dtype=float)
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()

    for elem in elements:
        k_local, T, _, _, _ = _get_element_matrices(elem, node_coords, load_params, nodes_df.attrs['x_lengths'], nodes_df.attrs['z_lengths'])
        
        k_global_elem = T.T @ k_local @ T
        
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

def calculate_gravity_loads(nodes_df, elements, load_params, node_coords):
    """Calculates the Global Nodal Load Vector P_global."""
    total_dofs = nodes_df.attrs.get('total_dofs', 0)
    if total_dofs == 0: return np.zeros(0, dtype=float)
        
    P_global = np.zeros(total_dofs, dtype=float)
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()
    x_lengths = nodes_df.attrs['x_lengths']
    z_lengths = nodes_df.attrs['z_lengths']

    for elem in elements:
        _, T, P_fixed_local, _, _ = _get_element_matrices(elem, node_coords, load_params, x_lengths, z_lengths)
        
        # P_global_elem is the global equivalent nodal force (P_nodal = -P_fixed_global)
        P_fixed_global = T.T @ P_fixed_local 

        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        global_dofs = np.concatenate((np.arange(dof_start_i, dof_start_i + 6), 
                                      np.arange(dof_start_j, dof_start_j + 6)))
        
        # P_global is the negative of the fixed-end forces (equivalent nodal loads)
        for i, global_dof in enumerate(global_dofs):
            P_global[global_dof] -= P_fixed_global[i]

    return P_global

def solve_fea_system(nodes_df, K_global, P_global):
    """Applies boundary conditions and solves the system K*U = P for displacements U."""
    total_dofs = K_global.shape[0]
    if total_dofs == 0: return nodes_df.copy(), np.zeros(0, dtype=float)
        
    known_dofs = [] 
    for _, row in nodes_df.iterrows():
        if row['support_type'] == 'Fixed':
            known_dofs.extend(range(row['dof_start_index'], row['dof_start_index'] + 6))

    all_dofs = np.arange(total_dofs)
    unknown_dofs = np.setdiff1d(all_dofs, known_dofs)
    
    if len(unknown_dofs) == 0:
        st.error("No free degrees of freedom found. Structure is overly constrained.")
        return nodes_df.copy(), np.zeros(total_dofs, dtype=float)

    K_uu = K_global[np.ix_(unknown_dofs, unknown_dofs)]
    P_u = P_global[unknown_dofs]

    try:
        if np.linalg.det(K_uu) < 1e-9:
             st.error("Reduced Stiffness Matrix is singular. Check structure stability (mechanisms).")
             return nodes_df.copy(), np.zeros(total_dofs, dtype=float) 
             
        U_u = np.linalg.solve(K_uu, P_u)
    except np.linalg.LinAlgError as e:
        st.error(f"Linear algebra error during solution: {e}. Cannot solve system.")
        return nodes_df.copy(), np.zeros(total_dofs, dtype=float)

    U_full = np.zeros(total_dofs, dtype=float)
    U_full[unknown_dofs] = U_u
    
    temp_nodes_df = nodes_df.copy() 
    temp_nodes_df['u'] = temp_nodes_df['u'].astype(object) 
    
    for index, row in temp_nodes_df.iterrows():
        start_index = row['dof_start_index']
        temp_nodes_df.at[index, 'u'] = U_full[start_index : start_index + 6]

    return temp_nodes_df, U_full

def calculate_element_end_forces(elements, U_full, node_coords, nodes_df, load_params):
    """Calculates internal forces for each element F_local = k_local * T * U_global - P_fixed_local."""
    x_lengths = nodes_df.attrs['x_lengths']
    z_lengths = nodes_df.attrs['z_lengths']
    node_id_to_dof_start = nodes_df.set_index('id')['dof_start_index'].to_dict()
    
    elements_with_forces = []
    
    for elem in elements:
        k_local, T, P_fixed_local, w_udl, L = _get_element_matrices(elem, node_coords, load_params, x_lengths, z_lengths)
        
        # 1. Get U_global_elem (12x1)
        dof_start_i = node_id_to_dof_start[elem['start']]
        dof_start_j = node_id_to_dof_start[elem['end']]
        global_dofs = np.concatenate((np.arange(dof_start_i, dof_start_i + 6), 
                                      np.arange(dof_start_j, dof_start_j + 6)))
        
        # Guard against zero-length U_full (solver failure)
        if len(U_full) == 0:
            U_global_elem = np.zeros(12)
        else:
            U_global_elem = U_full[global_dofs]

        # 2. Calculate F_local = k_local * T * U_global - P_fixed_local
        F_local = k_local @ (T @ U_global_elem) - P_fixed_local
        
        # 3. Extract key moments and forces
        # Bending Moment Mz' (DOFs 5 and 11) is the major axis moment (gravity bending for horizontal beams)
        elem['M_start'] = F_local[5] 
        elem['M_end'] = F_local[11]
        elem['Axial_Force'] = F_local[0]
        elem['UDL_Total'] = w_udl
        
        elements_with_forces.append(elem)
        
    return elements_with_forces

def perform_analysis(nodes_df, elements, load_params, node_coords):
    """Performs full FEA setup, solves for displacement and internal forces."""
    
    nodes_df = map_degrees_of_freedom(nodes_df)
    
    # 1. Assemble Global Stiffness Matrix (K)
    K_global = assemble_global_stiffness(nodes_df, elements, node_coords, load_params)
    st.session_state['K_global'] = K_global
    
    # 2. Generate Global Load Vector (P)
    P_global = calculate_gravity_loads(nodes_df, elements, load_params, node_coords)
    
    # 3. Solve System for Displacement (U)
    with st.spinner('Solving for displacement vector U...'):
        nodes_df_solved, U_full = solve_fea_system(nodes_df, K_global, P_global) 
    
    # 4. Calculate Internal Element Forces (F)
    elements_solved = calculate_element_end_forces(elements, U_full, node_coords, nodes_df_solved, load_params)
    
    # 5. Calculate Deflection Magnitude and Max Deflection
    max_deflection = 0.0
    
    if isinstance(U_full, np.ndarray) and len(U_full) > 0:
        y_indices = np.arange(1, len(U_full), 6)
        
        if y_indices.size > 0:
            abs_displacements = np.abs(U_full[y_indices])
            if np.all(np.isfinite(abs_displacements)):
                max_deflection = np.max(abs_displacements)
            
        nodes_df_solved['deflection_magnitude'] = nodes_df_solved['u'].apply(
            lambda x: np.linalg.norm(x[0:3]) if isinstance(x, np.ndarray) and x.shape == (6,) else 0.0
        )
    
    if max_deflection == 0:
        st.warning("Calculated displacement is zero. Check loads/boundary conditions or structural stability.")
    
    if max_deflection > 0:
        st.success(f"FEA Solution Complete. Maximum Y Displacement (Deflection) is **${max_deflection * 1000:.2f} \\text{{ mm}}$**.")

    return nodes_df_solved, elements_solved

# --- 4. Plotly Visualization Function (Updated) ---
def plot_3d_frame(nodes_df, elements, display_mode):
    """
    Creates an interactive 3D Plotly figure with visualizations for 
    Deflection, Load Distribution, or Bending Moment.
    """
    fig = go.Figure()
    
    scale_factor = 200 # For displacement visualization
    
    nodes_df['x_plot'] = nodes_df.apply(lambda row: row['x'] + row['u'][0] * scale_factor if isinstance(row['u'], np.ndarray) and display_mode == 'Deflection' else row['x'], axis=1)
    nodes_df['y_plot'] = nodes_df.apply(lambda row: row['y'] + row['u'][1] * scale_factor if isinstance(row['u'], np.ndarray) and display_mode == 'Deflection' else row['y'], axis=1)
    nodes_df['z_plot'] = nodes_df.apply(lambda row: row['z'] + row['u'][2] * scale_factor if isinstance(row['u'], np.ndarray) and display_mode == 'Deflection' else row['z'], axis=1)

    node_coords_plot = nodes_df.set_index('id')[['x_plot', 'y_plot', 'z_plot']].to_dict('index')

    # 1. Plot Nodes 
    fig.add_trace(go.Scatter3d(
        x=nodes_df['x_plot'], y=nodes_df['y_plot'], z=nodes_df['z_plot'],
        mode='markers',
        marker=dict(size=5, color=nodes_df['support_type'].apply(lambda x: 'red' if x == 'Fixed' else 'green'), opacity=0.8),
        hoverinfo='text',
        text=[
            f"Node ID: {row['id']}<br>Coords: ({row['x']:.2f}, {row['y']:.2f}, {row['z']:.2f})<br>"
            f"Disp (mm): U={row['u'][0]*1000:.2f}, V={row['u'][1]*1000:.2f}, W={row['u'][2]*1000:.2f}<br>"
            f"Support: {row['support_type']}"
            for index, row in nodes_df.iterrows() if isinstance(row['u'], np.ndarray)
        ],
        name='Nodes / Supports'
    ))

    # 2. Plot Elements (Geometry)
    type_colors = {'column': 'darkorchid', 'beam-x': 'royalblue', 'beam-z': 'darkcyan'}
    max_moment = max(abs(elem.get('M_start', 0)) for elem in elements) or 1.0
    
    for elem_type, elems in pd.DataFrame(elements).groupby('type'):
        
        for index, elem in elems.iterrows():
            p1 = node_coords_plot[elem['start']]; p2 = node_coords_plot[elem['end']]
            p1_xyz = (p1['x_plot'], p1['y_plot'], p1['z_plot'])
            p2_xyz = (p2['x_plot'], p2['y_plot'], p2['z_plot'])
            
            # Line width proportional to member depth (h_m)
            h_mm = elem.get('h_m', 0.5) * 1000 
            base_width = np.clip(h_mm * 0.025, 2, 15) 
            line_width = base_width
            current_color = type_colors.get(elem_type, 'gray')
            
            element_hover_text = (
                f"Element ID: {elem['id']}<br>Type: {elem_type}<br>"
                f"Section: {elem['b_m']*1000:.0f}x{elem['h_m']*1000:.0f} mm<br>"
                f"L: {elem.get('L', 0):.2f} m"
            )
            
            # --- Bending Moment Coloring/Sizing ---
            moment_text = ""
            if display_mode == 'Bending Moment':
                moment_text = f"<br>M_start: {elem.get('M_start', 0):.2f} kNm<br>M_end: {elem.get('M_end', 0):.2f} kNm"
                
                # Color code by magnitude (Red=High, Blue=Low)
                M_max_abs = max(abs(elem.get('M_start', 0)), abs(elem.get('M_end', 0)))
                # Normalize moment for color scale 0 to 1
                norm_moment = M_max_abs / max_moment 
                
                # Line width increase based on max moment in element
                line_width = base_width + norm_moment * 5 
                
                # Interpolate color (e.g., from blue to red)
                r = int(255 * norm_moment)
                b = int(255 * (1 - norm_moment))
                current_color = f'rgb({r}, 0, {b})'
            
            fig.add_trace(go.Scatter3d(
                x=[p1_xyz[0], p2_xyz[0]], y=[p1_xyz[1], p2_xyz[1]], z=[p1_xyz[2], p2_xyz[2]],
                mode='lines', line=dict(color=current_color, width=line_width),
                name=None, hoverinfo='text', showlegend=(index == 0),
                text=element_hover_text + moment_text
            ))
            
            # --- Bending Moment Diagram Markers ---
            if display_mode == 'Bending Moment':
                # Plot markers at start and end nodes, sized by moment magnitude
                for node_id, M_val in [(elem['start'], elem['M_start']), (elem['end'], elem['M_end'])]:
                    if abs(M_val) > 1e-3:
                        coord = node_coords_plot[node_id]
                        size = 5 + abs(M_val) / max_moment * 15 # Scale marker size
                        fig.add_trace(go.Scatter3d(
                            x=[coord['x_plot']], y=[coord['y_plot']], z=[coord['z_plot']],
                            mode='markers',
                            marker=dict(size=size, color='yellow' if M_val > 0 else 'cyan', symbol='square', opacity=1.0),
                            name=None,
                            showlegend=False,
                            hoverinfo='text',
                            text=f"Node {node_id}<br>Moment Mz': {M_val:.2f} kNm"
                        ))

            # --- Load Distribution Visualization (UDL Arrows) ---
            if display_mode == 'Load Distribution' and 'beam' in elem_type and elem['UDL_Total'] > 0:
                mid_x = (p1_xyz[0] + p2_xyz[0]) / 2.0
                mid_y = (p1_xyz[1] + p2_xyz[1]) / 2.0
                mid_z = (p1_xyz[2] + p2_xyz[2]) / 2.0
                
                # UDL is gravity (in -Y direction)
                load_magnitude = elem['UDL_Total'] * elem.get('L', 1.0) # Total UDL force
                arrow_size = np.clip(load_magnitude * 0.05, 0.5, 3) # Scale arrow size
                
                # Plot an arrow (cone) or simply a vector/marker at midspan
                fig.add_trace(go.Cone(
                    x=[mid_x], y=[mid_y + arrow_size], z=[mid_z],
                    u=[0], v=[-1], w=[0], # Vector points straight down
                    sizemode="absolute", sizeref=arrow_size * 2, anchor="tip",
                    colorscale=[[0, 'purple'], [1, 'purple']],
                    name=None, showlegend=False,
                    hoverinfo='text',
                    text=f"UDL: {elem['UDL_Total']:.2f} kN/m"
                ))

    # 3. Layout Configuration 
    max_x = nodes_df['x'].max() if not nodes_df.empty else 10
    max_y = nodes_df['y'].max() if not nodes_df.empty else 10
    max_z = nodes_df['z'].max() if not nodes_df.empty else 10
    min_y = nodes_df['y'].min() if not nodes_df.empty else 0
    
    plot_limit_x = max_x * 1.1; plot_limit_z = max_z * 1.1 
    plot_limit_y_top = max_y * 1.1; plot_limit_y_bottom = min_y * 1.1 
    
    fig.update_layout(
        title=f'Interactive 3D Frame - {display_mode}',
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
st.sidebar.markdown("Use format: `L1, N2xL2, ...` (e.g., `5, 2x4.5`). Units are meters.")

# Grid Inputs 
default_x = "3x5, 2x4"
x_input = st.sidebar.text_input("X Direction Bays (Length)", default_x)
default_y = "4.5, 3x3.5"
y_input = st.sidebar.text_input("Y Direction Heights (Floor)", default_y)
default_z = "4x6"
z_input = st.sidebar.text_input("Z Direction Bays (Width)", default_z)
default_foundation = 3.0
foundation_depth = st.sidebar.number_input("Foundation Depth below Ground (m)", min_value=0.0, value=default_foundation, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔩 Material & Section Properties (RC)")

E_val = st.sidebar.number_input("Young's Modulus E (kN/m²)", value=2.5e7, format="%e", help="E for M25 concrete is ~2.5e7 kN/m²")
st.sidebar.subheader("Column Dimensions ($b \\times h$, mm)")
col_b = st.sidebar.number_input("Column Width b (mm)", value=300, min_value=100); col_h = st.sidebar.number_input("Column Depth h (mm)", value=600, min_value=100)
st.sidebar.subheader("Beam Dimensions ($b \\times h$, mm)")
beam_b = st.sidebar.number_input("Beam Width b (mm)", value=230, min_value=100); beam_h = st.sidebar.number_input("Beam Depth h (mm)", value=450, min_value=100)

prop_column = calculate_rc_properties(col_b, col_h, E_val)
prop_beam = calculate_rc_properties(beam_b, beam_h, E_val)

st.sidebar.markdown("---")
st.sidebar.header("⚖️ Gravity Load Inputs")

load_params = {}
load_params['slab_thickness'] = st.sidebar.number_input("Slab Thickness $t$ (m)", min_value=0.0, value=0.15, step=0.01)
load_params['slab_density'] = st.sidebar.number_input("Concrete Density $\\gamma$ (kN/m³)", min_value=1.0, value=25.0, step=1.0)
load_params['finish_load'] = st.sidebar.number_input("Floor Finish Load (kN/m²)", min_value=0.0, value=1.5, step=0.1)
load_params['live_load'] = st.sidebar.number_input("Live Load (kN/m²)", min_value=0.0, value=3.0, step=0.5)

st.sidebar.markdown("---")

# Mandatory button to trigger analysis 
if st.sidebar.button("Run Analysis & Visualize 🚀"):
    st.session_state['run_generation'] = True

st.sidebar.header("📊 Visualization Mode")
display_mode = st.sidebar.selectbox(
    "Select Result to Display",
    ['Deflection', 'Geometry/Supports', 'Load Distribution', 'Bending Moment']
)

# Initialize state
if 'run_generation' not in st.session_state: st.session_state['run_generation'] = False 
if 'nodes_df' not in st.session_state: st.session_state['nodes_df'] = None
if 'elements' not in st.session_state: st.session_state['elements'] = None
if 'node_coords' not in st.session_state: st.session_state['node_coords'] = None

# Main Page Content
st.title("3D Frame Analysis: Load and Moment Visualization")
st.markdown(r"""
The structural solver now calculates **Element End Moments** based on solved displacements ($\mathbf{U}$) and applied fixed-end forces (gravity load). 

* **Deflection:** Shows the deformed structure (magnified).
* **Load Distribution:** Shows the UDL (Uniformly Distributed Load) arrows applied to the beams.
* **Bending Moment:** Shows the element lines colored/sized by the maximum moment, with yellow/cyan markers indicating moments at the nodes.
""")

# --- 6. Main Logic Block ---
if st.session_state.get('run_generation'):
    
    x_lengths = parse_grid_input(x_input)
    y_heights = parse_grid_input(y_input)
    z_lengths = parse_grid_input(z_input)
    
    if not x_lengths or not y_heights or not z_lengths:
        st.error("Please ensure all three dimensions (X, Y, Z) have valid, positive length inputs.")
    else:
        with st.spinner('Generating 3D geometry...'):
            nodes_df, elements, node_coords = generate_grid_geometry(
                x_lengths, y_heights, z_lengths, foundation_depth, prop_column, prop_beam
            )
        
        with st.spinner('Performing FEA (Stiffness, Load, Solve, Forces)...'):
            nodes_df_solved, elements_solved = perform_analysis(nodes_df, elements, load_params, node_coords)
            st.session_state['nodes_df'] = nodes_df_solved
            st.session_state['elements'] = elements_solved
            st.session_state['node_coords'] = node_coords
        
    st.session_state['run_generation'] = False

# --- 7. Display Visualization ---
if st.session_state['nodes_df'] is not None:
    frame_fig = plot_3d_frame(
        st.session_state['nodes_df'], 
        st.session_state['elements'], 
        display_mode
    )
    st.plotly_chart(frame_fig, use_container_width=True)

    st.subheader("FEA Results Summary")
    
    # Calculate overall gravity load for display
    q_total_gravity = (load_params['slab_density'] * load_params['slab_thickness'] + 
                       load_params['finish_load'] + load_params['live_load'])
    
    max_abs_moment = max(abs(elem.get('M_start', 0)) for elem in st.session_state['elements'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Total Nodes:** `{len(st.session_state['nodes_df'])}`")
        st.markdown(f"**Total Elements:** `{len(st.session_state['elements'])}`")
    with col2:
        st.markdown(f"**Global Floor Pressure:** $ {q_total_gravity:.2f} \\frac{{kN}}{{m^2}} $")
        if 'K_global' in st.session_state:
            st.markdown(f"**Global K Size:** `{st.session_state['K_global'].shape}`")
    with col3:
        st.metric("Max Absolute Moment", f"{max_abs_moment:.2f} kNm")
        max_y_deflection = st.session_state['nodes_df']['u'].apply(
            lambda x: x[1] if isinstance(x, np.ndarray) and len(x) > 1 and np.isfinite(x[1]) else 0
        ).abs().max() * 1000
        st.metric("Max Vertical Displacement", f"{max_y_deflection:.2f} mm")
