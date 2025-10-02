import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Full 3D FEA Building Analyzer (IS Code)")

# --- 1. Object-Oriented Model for the Structure (Unchanged FEA Core) ---

class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        # 6 degrees of freedom (DOF): [dx, dy, dz, rx, ry, rz]
        # True means the DOF is restrained (fixed).
        self.restraints = [False] * 6
        self.reactions = np.zeros(6) # Fx, Fy, Fz, Mx, My, Mz
        self.load = np.zeros(6)      # Nodal loads
        self.displacements = np.zeros(6) # Calculated displacements

    def __repr__(self):
        return f"Node(id={self.id}, pos=({self.x}, {self.y}, {self.z}))"

class Element:
    """Represents a single beam/column element in the 3D structure."""
    def __init__(self, id, start_node, end_node, props):
        self.id = int(id)
        self.start_node = start_node # Node object
        self.end_node = end_node     # Node object
        self.props = props           # {'E', 'G', 'A', 'Iy', 'Iz', 'J', 'rho'}
        self.length = self._calculate_length()

    def _calculate_length(self):
        """Calculates the element length."""
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        dz = self.end_node.z - self.start_node.z
        return np.sqrt(dx**2 + dy**2 + dz**2)

    def __repr__(self):
        return f"Element(id={self.id}, nodes={self.start_node.id}-{self.end_node.id})"

# --- 2. 3D Beam Element Stiffness & Transformation Functions (Unchanged) ---

def get_element_stiffness_matrix(element: Element) -> np.ndarray:
    """Generates the 12x12 local stiffness matrix for a 3D beam element."""
    L = element.length
    E, G = element.props['E'], element.props['G']
    A = element.props['A']
    Iy, Iz = element.props['Iy'], element.props['Iz']
    J = element.props['J']

    if L < 1e-9:
        return np.zeros((12, 12))

    # Pre-calculated terms
    EA_L = E * A / L
    EIy_L = E * Iy / L
    EIz_L = E * Iz / L
    GA_J_L = G * J / L

    K_local = np.zeros((12, 12))

    # Axial (DOF 1, 7)
    K_local[0, 0] = K_local[6, 6] = EA_L
    K_local[0, 6] = K_local[6, 0] = -EA_L

    # Torsion (DOF 4, 10)
    K_local[3, 3] = K_local[9, 9] = GA_J_L
    K_local[3, 9] = K_local[9, 3] = -GA_J_L

    # Bending about Z-axis (Shear in Y-dir: DOF 2, 6, 8, 12)
    K_local[1, 1] = K_local[7, 7] = 12 * EIz_L / L**2
    K_local[1, 7] = K_local[7, 1] = -12 * EIz_L / L**2
    K_local[1, 5] = K_local[5, 1] = 6 * EIz_L / L
    K_local[1, 11] = K_local[11, 1] = 6 * EIz_L / L
    K_local[7, 5] = K_local[5, 7] = -6 * EIz_L / L
    K_local[7, 11] = K_local[11, 7] = -6 * EIz_L / L
    K_local[5, 5] = K_local[11, 11] = 4 * EIz_L
    K_local[5, 11] = K_local[11, 5] = 2 * EIz_L

    # Bending about Y-axis (Shear in Z-dir: DOF 3, 5, 9, 11)
    K_local[2, 2] = K_local[8, 8] = 12 * EIy_L / L**2
    K_local[2, 8] = K_local[8, 2] = -12 * EIy_L / L**2
    K_local[2, 4] = K_local[4, 2] = -6 * EIy_L / L 
    K_local[2, 10] = K_local[10, 2] = -6 * EIy_L / L
    K_local[8, 4] = K_local[4, 8] = 6 * EIy_L / L
    K_local[8, 10] = K_local[10, 8] = 6 * EIy_L / L
    K_local[4, 4] = K_local[10, 10] = 4 * EIy_L
    K_local[4, 10] = K_local[10, 4] = 2 * EIy_L

    return K_local

def get_transformation_matrix(element: Element) -> np.ndarray:
    """Generates the 12x12 transformation matrix (T) from local to global coordinates."""
    n1, n2 = element.start_node, element.end_node
    L = element.length
    
    if L < 1e-9: 
        return np.eye(12)

    # Direction cosines
    lx = (n2.x - n1.x) / L
    mx = (n2.y - n1.y) / L
    nx = (n2.z - n1.z) / L
    
    # Standard 3D frame assumption for local y and z orientation
    
    if abs(lx) < 1e-6 and abs(mx) < 1e-6: # Vertical element (parallel to global Z-axis)
        ly, my, ny = 1, 0, 0
        lz, mz, nz = 0, nx, -mx
    else: # General case
        D = np.sqrt(lx**2 + mx**2)
        ly = -mx / D
        my = lx / D
        ny = 0
        
        # Local z-axis (V3) is cross product of local x (V1) and local y (V2)
        lz = mx * ny - nx * my
        mz = nx * ly - lx * ny
        nz = lx * my - mx * ly
        
    R = np.array([
        [lx, mx, nx],
        [ly, my, ny],
        [lz, mz, nz]
    ])
    
    # Transformation matrix T is block diagonal
    T = np.zeros((12, 12))
    for i in range(4):
        T[i*3:i*3+3, i*3:i*3+3] = R

    return T

# --- 3. Structure Class and Solver (Unchanged) ---

class Structure:
    """Manages the full finite element analysis model."""
    def __init__(self, nodes: List[Node], elements: List[Element]):
        self.nodes = nodes
        self.elements = elements
        self.dof_map = self._create_dof_map()
        self.num_dof = len(self.dof_map)
        self.K_global = np.zeros((self.num_dof, self.num_dof))
        self.F_global = np.zeros(self.num_dof)
        
    def _create_dof_map(self) -> Dict[Tuple[int, int], int]:
        """Maps (Node ID, DOF Index) to Global DOF Number."""
        dof_map = {}
        global_dof_counter = 0
        for node in self.nodes:
            for i in range(6): # dx, dy, dz, rx, ry, rz
                dof_map[(node.id, i)] = global_dof_counter
                global_dof_counter += 1
        return dof_map

    def assemble_matrices(self):
        """Assembles the global stiffness matrix (K) and load vector (F)."""
        self.K_global.fill(0.0)
        self.F_global.fill(0.0)
        
        for element in self.elements:
            T = get_transformation_matrix(element)
            K_local = get_element_stiffness_matrix(element)
            
            K_global_e = T.T @ K_local @ T
            
            # Map element DOFs (12) to global DOFs
            node_ids = [element.start_node.id, element.end_node.id]
            global_indices = []
            for node_id in node_ids:
                for i in range(6):
                    global_indices.append(self.dof_map[(node_id, i)])

            # Assembly
            for i in range(12):
                for j in range(12):
                    self.K_global[global_indices[i], global_indices[j]] += K_global_e[i, j]

        # Assemble global load vector F
        for node in self.nodes:
            for i in range(6):
                global_dof = self.dof_map[(node.id, i)]
                self.F_global[global_dof] += node.load[i]

    def solve(self) -> np.ndarray:
        """Applies boundary conditions and solves for displacements."""
        
        free_dof, support_dof = [], []
        
        for node in self.nodes:
            for i in range(6):
                global_dof = self.dof_map[(node.id, i)]
                if node.restraints[i]:
                    support_dof.append(global_dof)
                else:
                    free_dof.append(global_dof)
        
        # Partition K and F matrices
        Kff = self.K_global[np.ix_(free_dof, free_dof)]
        Ksf = self.K_global[np.ix_(support_dof, free_dof)]
        Ff = self.F_global[free_dof]

        # Solve for free displacements (Df)
        Df = np.linalg.solve(Kff, Ff)

        # Reconstruct full displacement vector D
        D_full = np.zeros(self.num_dof)
        D_full[free_dof] = Df
        
        # Calculate support reactions (Rs): Rs = Ksf * Df - Fs (where Fs is load at support DOF, which is F_global[support_dof])
        Fs = self.F_global[support_dof]
        Rs = Ksf @ Df - Fs

        # Distribute results back to nodes
        for node in self.nodes:
            for i in range(6):
                global_dof = self.dof_map[(node.id, i)]
                node.displacements[i] = D_full[global_dof]
                
                if node.restraints[i]:
                    idx = support_dof.index(global_dof)
                    node.reactions[i] = Rs[idx]
                else:
                    node.reactions[i] = 0.0

        return D_full

# --- 4. Geometry Generation (Multi-Bay, Multi-Story) ---

def generate_default_structure(params) -> Tuple[List[Node], List[Element]]:
    """
    Generates a multi-bay, multi-storey 3D grid structure.
    
    New Parameters:
    - Num Bays X, Num Bays Y, Num Stories
    - Bay Length X, Bay Length Y, Storey Height
    """
    
    Lx = params['Bay Length X']
    Ly = params['Bay Length Y']
    H = params['Storey Height']
    
    n_bays_x = params['Num Bays X']
    n_bays_y = params['Num Bays Y']
    n_stories = params['Num Stories']
    
    # 1. Generate Nodal Coordinates
    
    # Grid points needed: (n_bays + 1) in each direction
    x_coords = np.arange(n_bays_x + 1) * Lx
    y_coords = np.arange(n_bays_y + 1) * Ly
    z_coords = np.arange(n_stories + 1) * H # Base is z=0
    
    nodes = []
    node_map = {}
    node_id_counter = 1
    
    # Create all nodes
    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                node = Node(node_id_counter, x, y, z)
                
                # Apply Fixed restraints at the base (z=0)
                if z == 0.0:
                    node.restraints = [True] * 6 # Fully fixed
                    
                nodes.append(node)
                # Store node map using a tuple (x_index, y_index, z_index)
                node_map[(i, j, k)] = node
                node_id_counter += 1
                
    # 2. Setup Material and Section Properties
    E_c = 5000 * np.sqrt(params['fck']) * 1e6 # N/m^2
    E = E_c / (1e6) # kN/m2 (Converted to kN/m^2)
    G = E / (2 * (1 + 0.2)) # Shear Modulus (v=0.2)
    rho = params['rho'] # Density kN/m3 (RC)
    
    b, d = params['Section Size'], params['Section Size']
    A = b * d 
    Iy = (b * d**3) / 12.0 # Moment of inertia about Y
    Iz = (d * b**3) / 12.0 # Moment of inertia about Z
    J = Iy + Iz # Torsional constant (approx for square)
    
    element_props = {'E': E, 'G': G, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'rho': rho}
    
    # 3. Generate Elements (Beams and Columns)
    elements = []
    elem_counter = 1
    
    # Loop over all internal grid indices
    for k in range(n_stories + 1): # Stories (Z)
        for j in range(n_bays_y + 1): # Y-Gridlines
            for i in range(n_bays_x + 1): # X-Gridlines
                
                current_node = node_map[(i, j, k)]
                
                # --- A. Columns (Vertical, Z-direction) ---
                if k < n_stories:
                    # Connect to the node above
                    node_above = node_map[(i, j, k + 1)]
                    elements.append(Element(elem_counter, current_node, node_above, element_props.copy()))
                    elem_counter += 1
                
                # --- B. Beams in X-direction ---
                if i < n_bays_x and k > 0: # Exclude base (k=0) for floor beams
                    # Connect to the node on the right (i+1)
                    node_right = node_map[(i + 1, j, k)]
                    elements.append(Element(elem_counter, current_node, node_right, element_props.copy()))
                    elem_counter += 1

                # --- C. Beams in Y-direction ---
                if j < n_bays_y and k > 0: # Exclude base (k=0) for floor beams
                    # Connect to the node in front (j+1)
                    node_front = node_map[(i, j + 1, k)]
                    elements.append(Element(elem_counter, current_node, node_front, element_props.copy()))
                    elem_counter += 1

    return nodes, elements

# --- 5. Load Application Functions (Updated for Full Building) ---

def apply_dead_load(nodes, elements, params):
    """Calculates self-weight (dead load) and applies it to nodes across all floors."""
    
    Lx = params['Bay Length X']
    Ly = params['Bay Length Y']
    
    # Distributed self-weight (q) = density * area (kN/m)
    q_elem = params['rho'] * params['Section Size']**2
    
    # Apply self-weight to elements
    for element in elements:
        L = element.length
        P_z = -q_elem * L / 2.0 # P_z is load in global Z direction
        
        element.start_node.load[2] += P_z
        element.end_node.load[2] += P_z
        
    # Apply floor/slab load (treated as uniform area load distributed to nodes)
    floor_dl = -params['Slab DL (kN/m2)'] * Lx * Ly / 4.0 # Load per node for an interior bay
    
    # Only apply to nodes above the base (k > 0)
    for n in nodes:
        if n.z > 1e-6:
            # Simple approximation: apply the tributary area load to all non-base nodes
            n.load[2] += floor_dl

def apply_live_load(nodes, elements, params):
    """Applies simplified live load to all floor nodes."""
    
    Lx = params['Bay Length X']
    Ly = params['Bay Length Y']
    
    floor_ll = -params['Live Load (kN/m2)'] * Lx * Ly / 4.0
        
    # Apply to all nodes above the base (k > 0)
    for n in nodes:
        if n.z > 1e-6:
            n.load[2] += floor_ll

def apply_wind_load(nodes, elements, params, direction='X'):
    """Applies simplified lateral (wind) load to all nodes at each floor level."""
    
    H = params['Storey Height']
    
    # Wind Pressure (Pz) = 0.6 * Vz^2 (Simplified IS 875 Part 3 approximation)
    Vb = params['Wind Speed (m/s)']
    P_wind = 0.6 * Vb**2 * 1e-3 # kN/m2 (Assuming constants=1)
    
    # Tributary height for a floor node: H/2 below and H/2 above = H
    # Total tributary area for an internal column line node: Ly * H (approx)
    
    # Calculate total base shear for one elevation (e.g., in X direction)
    # Area exposed to wind = (Total Height) * (Total Building Width)
    total_height = params['Num Stories'] * H
    total_width = params['Num Bays Y'] * params['Bay Length Y'] # Wind on X face (width is Y length)
    
    total_shear = P_wind * total_height * total_width
    
    # Distribute total shear to each floor (assuming uniform distribution)
    shear_per_story = total_shear / params['Num Stories']
    
    # Distribute shear per story to all nodes at that level
    nodes_per_story = (params['Num Bays X'] + 1) * (params['Num Bays Y'] + 1)
    # Only nodes on the exposed face carry wind load, but for simplicity in FEA,
    # we distribute the shear equally to all nodes at that floor level (assuming a rigid diaphragm)
    # Lateral force per node (P_node)
    P_node_per_floor = shear_per_story / nodes_per_story 
    
    # Apply the load to all nodes above the base
    for n in nodes:
        if n.z > 1e-6: # Skip base nodes
            if direction == 'X':
                # Apply load in +X direction (index 0)
                n.load[0] += P_node_per_floor
            elif direction == 'Y':
                # Apply load in +Y direction (index 1)
                n.load[1] += P_node_per_floor

# --- 6. Main Analysis & Visualization Logic (Unchanged) ---

REACTION_SCALE = 2.0 

def plot_3d_frame_with_reactions(nodes: List[Node], elements: List[Element], scale: float) -> go.Figure:
    """Generates an interactive 3D Plotly figure of the frame structure and support reactions."""

    fig = go.Figure()

    # --- 1. Plot Elements (Lines) ---
    line_x, line_y, line_z = [], [], []
    for element in elements:
        n1 = element.start_node
        n2 = element.end_node
        line_x.extend([n1.x, n2.x, None])
        line_y.extend([n1.y, n2.y, None])
        line_z.extend([n1.z, n2.z, None])

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines',
        line=dict(color='rgba(40, 40, 40, 0.8)', width=2.5),
        name='Structural Elements',
        hoverinfo='none'
    ))

    # --- 2. Plot Nodes (Markers) ---
    node_x = [n.x for n in nodes]
    node_y = [n.y for n in nodes]
    node_z = [n.z for n in nodes]
    
    node_text = [
        f"Node {n.id}<br>Coords: ({n.x:.1f}, {n.y:.1f}, {n.z:.1f})m<br>Load Fz: {n.load[2]:.2f} kN" 
        for n in nodes
    ]
    node_colors = ['#c0392b' if all(n.restraints) else '#2ecc71' for n in nodes] 

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(size=4, color=node_colors, symbol='circle', line=dict(width=0.5, color='Black')),
        hovertext=node_text,
        hoverinfo='text',
        name='Nodes'
    ))

    # --- 3. Plot Support Reactions (Vectors) ---
    reaction_traces = []
    
    reaction_props = [
        (0, 'red', 'Reaction Fx (kN)'),
        (1, 'green', 'Reaction Fy (kN)'),
        (2, 'blue', 'Reaction Fz (kN)'),
    ]

    for index, color, name in reaction_props:
        rx_x, rx_y, rx_z = [], [], []
        arrow_hover_text = []

        for n in nodes:
            if n.restraints[index] and abs(n.reactions[index]) > 1e-3: 
                start_point = [n.x, n.y, n.z]
                reaction_magnitude = n.reactions[index] 
                
                end_point = list(start_point)
                
                # Arrow direction (opposite of reaction for visual clarity)
                end_point[index] += -reaction_magnitude * scale / 25 

                rx_x.extend([start_point[0], end_point[0], None])
                rx_y.extend([start_point[1], end_point[1], None])
                rx_z.extend([start_point[2], end_point[2], None])
                
                arrow_hover_text.append(f"Node {n.id}<br>{name}: {reaction_magnitude:.2f} kN")

        if rx_x:
            # Line trace for the reaction vector
            reaction_traces.append(go.Scatter3d(
                x=rx_x, y=rx_y, z=rx_z,
                mode='lines',
                line=dict(color=color, width=3),
                name=name,
                showlegend=True,
                hoverinfo='none'
            ))

    fig.add_traces(reaction_traces)

    # --- 4. Layout Configuration ---
    
    max_range = max(max(node_x)-min(node_x), max(node_y)-min(node_y), max(node_z)-min(node_z))
    center_x = np.mean(node_x)
    center_y = np.mean(node_y)
    center_z = np.mean(node_z)
    
    # Adjust range slightly to provide padding
    padding = max_range * 0.1 
    
    fig.update_layout(
        title=f'3D Building View: Load Case **{st.session_state["current_load_case"]}**',
        height=700,
        scene=dict(
            xaxis=dict(title='X Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True),
            yaxis=dict(title='Y Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True),
            zaxis=dict(title='Z Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True),
            aspectmode='manual',
            # Set cubic aspect ratio based on the largest dimension for true scaling
            aspectratio=dict(x=1, y=1, z=1), 
            camera=dict(
                up=dict(x=0, y=0, z=1), 
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5) 
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# --- 7. Main Analysis & Data Flow ---

def run_analysis(nodes: List[Node], elements: List[Element], case: str, params: dict):
    """Calculates loads, runs the FEA solver, and stores results."""
    
    # 1. Reset loads on all nodes before applying the new case load
    for n in nodes:
        n.load.fill(0.0)

    # Helper function to apply and retrieve loads for a specific case (used for combinations)
    def get_case_loads(case_type, current_params):
        # Generate a temporary, identical structure state for isolated load application
        temp_nodes, temp_elements = generate_default_structure(current_params)
        
        # Apply appropriate loading function to temporary nodes
        if case_type == 'DL':
            apply_dead_load(temp_nodes, temp_elements, current_params)
        elif case_type == 'LL':
            apply_live_load(temp_nodes, temp_elements, current_params)
        elif case_type == 'WLX':
            apply_wind_load(temp_nodes, temp_elements, current_params, 'X')
        elif case_type == 'WLY':
            apply_wind_load(temp_nodes, temp_elements, current_params, 'Y')
        
        # Return loads indexed by node ID
        return {n.id: n.load for n in temp_nodes}


    # 2. Apply load for the selected case or combination
    
    if case == 'DL (Dead Load)':
        apply_dead_load(nodes, elements, params)
    elif case == 'LL (Live Load)':
        apply_live_load(nodes, elements, params)
    elif case == 'WLX (Wind Load +X)':
        apply_wind_load(nodes, elements, params, 'X')
    elif case == 'WLY (Wind Load +Y)':
        apply_wind_load(nodes, elements, params, 'Y')
    
    # --- Handle Load Combinations ---
    elif case == '1.5(DL+LL)':
        dl_loads = get_case_loads('DL', params)
        ll_loads = get_case_loads('LL', params)
        
        for n in nodes:
            n.load = 1.5 * (dl_loads[n.id] + ll_loads[n.id])
        
    elif case == '1.5(DL+WLX)':
        dl_loads = get_case_loads('DL', params)
        wlx_loads = get_case_loads('WLX', params)
        
        for n in nodes:
            n.load = 1.5 * (dl_loads[n.id] + wlx_loads[n.id])
    # Note: More combinations can be added here.


    # 3. Run Solver
    try:
        structure = Structure(nodes, elements)
        structure.assemble_matrices()
        structure.solve()
        st.session_state['analysis_success'] = True
    except np.linalg.LinAlgError:
        st.session_state['analysis_success'] = False
        st.error("Analysis Failed: Singular matrix detected. Check geometry/restraints.")
    except Exception as e:
        st.session_state['analysis_success'] = False
        st.error(f"Analysis Failed: {e}")

    # 4. Store the resulting structure state
    st.session_state['current_nodes'] = nodes
    st.session_state['current_elements'] = elements
    st.session_state['current_load_case'] = case


# --- Streamlit Initial Setup ---
if 'analysis_success' not in st.session_state:
    st.session_state['analysis_success'] = False
if 'current_load_case' not in st.session_state:
    st.session_state['current_load_case'] = 'DL (Dead Load)'
if 'current_nodes' not in st.session_state:
    st.session_state['current_nodes'] = []
if 'current_elements' not in st.session_state:
    st.session_state['current_elements'] = []

st.title("Full 3D Building Analysis and Visualization (FEA & IS Code Demo)")
st.markdown("Analyzes a **multi-bay, multi-storey** regular reinforced concrete (RC) frame structure.")
st.markdown("---")

# --- Sidebar for Parameters and Controls ---
with st.sidebar:
    st.header("1. Building Geometry")
    
    # New Building Dimensions
    NUM_STORIES = st.number_input("Number of Stories (k)", min_value=1, value=3, step=1, key='num_stories')
    NUM_BAYS_X = st.number_input("Number of Bays in X (i)", min_value=1, value=2, step=1, key='num_bays_x')
    NUM_BAYS_Y = st.number_input("Number of Bays in Y (j)", min_value=1, value=2, step=1, key='num_bays_y')
    
    st.subheader("Bay/Storey Dimensions")
    BAY_LENGTH_X = st.number_input("Bay Length X (Lx) (m)", min_value=3.0, value=8.0, step=0.5, key='Lx')
    BAY_LENGTH_Y = st.number_input("Bay Length Y (Ly) (m)", min_value=3.0, value=6.0, step=0.5, key='Ly')
    STOREY_HEIGHT = st.number_input("Storey Height (H) (m)", min_value=2.0, value=3.5, step=0.1, key='H')
    
    st.header("2. Material & Section")
    SECTION_SIZE = st.number_input("Square Section Size (m)", min_value=0.2, value=0.45, step=0.05, format="%.2f", key='b')
    FCK = st.number_input("Concrete Grade (fck) (MPa)", min_value=15, value=25, step=5, key='fck')
    DENSITY_RC = st.number_input("RC Density (rho) (kN/m³)", min_value=10.0, value=25.0, step=1.0, key='rho_input')
    
    st.header("3. Loads (Simplified)")
    SLAB_DL = st.number_input("Slab Dead Load (kN/m²)", min_value=0.5, value=2.5, step=0.5, key='slab_dl')
    LIVE_LOAD = st.number_input("Live Load (kN/m²)", min_value=0.5, value=3.0, step=0.5, key='ll')
    WIND_SPEED = st.number_input("Basic Wind Speed (Vb) (m/s)", min_value=10, value=39, step=1, key='wind_speed')
    
    st.header("4. Analysis Control")
    LOAD_CASES = [
        'DL (Dead Load)', 'LL (Live Load)', 
        'WLX (Wind Load +X)', 'WLY (Wind Load +Y)', 
        '1.5(DL+LL)', '1.5(DL+WLX)'
    ]
    
    selected_case = st.selectbox("Select Load Case to Analyze", options=LOAD_CASES, key='case_select')
    
    if st.button("Run Full Analysis", use_container_width=True):
        # 1. Collect all parameters
        analysis_params = {
            'Num Stories': NUM_STORIES,
            'Num Bays X': NUM_BAYS_X,
            'Num Bays Y': NUM_BAYS_Y,
            'Bay Length X': BAY_LENGTH_X,
            'Bay Length Y': BAY_LENGTH_Y,
            'Storey Height': STOREY_HEIGHT,
            'Section Size': SECTION_SIZE,
            'fck': FCK,
            'rho': DENSITY_RC,
            'Slab DL (kN/m2)': SLAB_DL,
            'Live Load (kN/m2)': LIVE_LOAD,
            'Wind Speed (m/s)': WIND_SPEED,
        }
        
        # 2. Generate new, full building structure
        nodes, elements = generate_default_structure(analysis_params)
        
        # 3. Run the solver for the selected case
        run_analysis(nodes, elements, selected_case, analysis_params)
        
    st.subheader("Visualization Scale")
    REACTION_SCALE = st.slider(
        "Reaction Vector Scale",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        key='reaction_scale',
        help="Adjusts the visual length of the reaction force arrows."
    )


# --- Main Content Area ---
if not st.session_state['analysis_success'] and not st.session_state['current_nodes']:
    st.info("Set the building geometry parameters and click **'Run Full Analysis'** in the sidebar.")
elif st.session_state['analysis_success']:
    
    # 3D View
    st.header(f"3D Model View: {st.session_state['current_load_case']}")
    st.caption(f"Total Nodes: {len(st.session_state['current_nodes'])}, Total Elements: {len(st.session_state['current_elements'])}")
    
    fig = plot_3d_frame_with_reactions(
        st.session_state['current_nodes'], 
        st.session_state['current_elements'], 
        st.session_state['reaction_scale']
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Tabs for detailed results
    tab_reactions, tab_displacements, tab_info = st.tabs([
        "Support Reactions", 
        "Nodal Displacements", 
        "Model Information"
    ])

    with tab_reactions:
        st.header("Calculated Support Reactions")
        st.caption("Only nodes at the base (Z=0) with fixed restraints are shown.")
        
        support_nodes_data = [
            {'Node ID': n.id, 
             'Coords': f"({n.x:.1f}, {n.y:.1f})",
             'Fx (kN)': f"{n.reactions[0]:.2f}", 
             'Fy (kN)': f"{n.reactions[1]:.2f}", 
             'Fz (kN)': f"{n.reactions[2]:.2f}",
             'Mx (kNm)': f"{n.reactions[3]:.2f}", 
             'My (kNm)': f"{n.reactions[4]:.2f}", 
             'Mz (kNm)': f"{n.reactions[5]:.2f}"}
            for n in st.session_state['current_nodes'] if all(n.restraints)
        ]

        if support_nodes_data:
            df_reactions = pd.DataFrame(support_nodes_data)
            st.dataframe(df_reactions.set_index('Node ID'), use_container_width=True)
        else:
            st.info("No fixed support nodes found.")

    with tab_displacements:
        st.header("Nodal Displacements")
        
        displacement_data = [
            {'Node ID': n.id, 
             'Coords': f"({n.x:.1f}, {n.y:.1f}, {n.z:.1f})",
             'dx (mm)': f"{n.displacements[0] * 1000:.4f}",
             'dy (mm)': f"{n.displacements[1] * 1000:.4f}",
             'dz (mm)': f"{n.displacements[2] * 1000:.4f}",
             'rz (rad)': f"{n.displacements[5]:.4e}"}
            for n in st.session_state['current_nodes']
        ]

        df_displacements = pd.DataFrame(displacement_data)
        st.dataframe(df_displacements.set_index('Node ID'), use_container_width=True)

    with tab_info:
        st.header("Key Model Parameters")
        st.write(f"**Total Global DOFs:** {len(st.session_state['current_nodes']) * 6}")
        st.write(f"**Total Nodes:** {len(st.session_state['current_nodes'])}")
        st.write(f"**Total Elements:** {len(st.session_state['current_elements'])}")
        
        # Display applied loads check
        total_loads = sum(np.sum(n.load) for n in st.session_state['current_nodes'])
        total_reactions = sum(np.sum(n.reactions) for n in st.session_state['current_nodes'])
        
        st.write(f"**Total Applied Load Sum (Check):** {total_loads:.2f} kN")
        st.write(f"**Total Reaction Sum (Check):** {total_reactions:.2f} kN")
        st.warning("Note: Sum of loads and reactions should be approximately zero for static equilibrium.")
    
# --- End of Streamlit App Layout ---
