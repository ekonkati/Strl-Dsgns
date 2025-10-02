import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
from typing import List, Dict, Any, Tuple

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Full 3D FEA Frame Analyzer (IS Code)")

# --- 1. Object-Oriented Model for the Structure ---
# Reintroducing structure classes to support full FEA functionality

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

# --- 2. 3D Beam Element Stiffness & Transformation Functions ---

def get_element_stiffness_matrix(element: Element) -> np.ndarray:
    """Generates the 12x12 local stiffness matrix for a 3D beam element."""
    L = element.length
    E, G = element.props['E'], element.props['G']
    A = element.props['A']
    Iy, Iz = element.props['Iy'], element.props['Iz']
    J = element.props['J']

    # Pre-calculated terms
    EIy_L = E * Iy / L
    EIz_L = E * Iz / L
    GA_J_L = G * J / L

    # Local stiffness matrix (K_local) 12x12
    K_local = np.zeros((12, 12))

    # Axial (DOF 1, 7)
    K_local[0, 0] = K_local[6, 6] = E * A / L
    K_local[0, 6] = K_local[6, 0] = -E * A / L

    # Torsion (DOF 4, 10)
    K_local[3, 3] = K_local[9, 9] = GA_J_L
    K_local[3, 9] = K_local[9, 3] = -GA_J_L

    # Shear/Bending in Y-Z plane (Bending about Z-axis, Shear in Y-dir: DOF 2, 6, 8, 12)
    K_local[1, 1] = K_local[7, 7] = 12 * EIz_L / L**2
    K_local[1, 7] = K_local[7, 1] = -12 * EIz_L / L**2
    K_local[1, 5] = K_local[5, 1] = 6 * EIz_L / L
    K_local[1, 11] = K_local[11, 1] = 6 * EIz_L / L
    K_local[7, 5] = K_local[5, 7] = -6 * EIz_L / L
    K_local[7, 11] = K_local[11, 7] = -6 * EIz_L / L
    K_local[5, 5] = K_local[11, 11] = 4 * EIz_L
    K_local[5, 11] = K_local[11, 5] = 2 * EIz_L

    # Shear/Bending in X-Z plane (Bending about Y-axis, Shear in Z-dir: DOF 3, 5, 9, 11)
    # Note: Indices are 0-based, so dx=0, dy=1, dz=2, rx=3, ry=4, rz=5.
    # DOF (3, 9) for bending about Y axis (index 4) and shear in Z direction (index 2)
    K_local[2, 2] = K_local[8, 8] = 12 * EIy_L / L**2
    K_local[2, 8] = K_local[8, 2] = -12 * EIy_L / L**2
    K_local[2, 4] = K_local[4, 2] = -6 * EIy_L / L # Note the sign change for Y-bending convention
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
    
    if L < 1e-9: # Avoid division by zero for zero-length elements
        return np.eye(12)

    # Direction cosines
    lx = (n2.x - n1.x) / L
    mx = (n2.y - n1.y) / L
    nx = (n2.z - n1.z) / L
    
    # Standard 3D frame assumption: the local y-axis is in the plane of the local x-axis and the global Z-axis.
    
    # Case 1: Element is vertical (parallel to global Z-axis)
    if abs(lx) < 1e-6 and abs(mx) < 1e-6:
        # If aligned with Z-axis, local x is global z
        # Local y is assumed to align with global X
        ly, my, ny = 1, 0, 0
        lz, mz, nz = 0, nx, -mx # (0, 0, 1) or (0, 0, -1) depending on direction
    # Case 2: General case
    else:
        D = np.sqrt(lx**2 + mx**2)
        ly = -mx / D
        my = lx / D
        ny = 0
        
        # Reference vector for 'up' is [0, 0, 1] (Global Z)
        V1 = np.array([lx, mx, nx])
        
        # Vector V2 (local y-axis) is defined as V3 x V1 (cross product)
        V2 = np.cross([0, 0, 1], V1)
        
        # Handle case where V1 is parallel to V3 (nx=1 or nx=-1)
        if np.linalg.norm(V2) < 1e-6:
            V2 = np.array([1, 0, 0]) # Assume local y is global x
        
        V2 = V2 / np.linalg.norm(V2)
        
        # Vector V3_local (local z-axis) is V1 x V2 (cross product)
        V3_local = np.cross(V1, V2)
        V3_local = V3_local / np.linalg.norm(V3_local)
        
        ly, my, ny = V2[0], V2[1], V2[2]
        lz, mz, nz = V3_local[0], V3_local[1], V3_local[2]
        
    R = np.array([
        [lx, mx, nx],
        [ly, my, ny],
        [lz, mz, nz]
    ])

    T_3x3 = R
    
    # Transformation matrix T is block diagonal: T_3x3, T_3x3, T_3x3, T_3x3
    T = np.zeros((12, 12))
    T[:3, :3] = T_3x3
    T[3:6, 3:6] = T_3x3
    T[6:9, 6:9] = T_3x3
    T[9:12, 9:12] = T_3x3

    return T

# --- 3. Structure Class and Solver ---

class Structure:
    """Manages the full finite element analysis model."""
    def __init__(self, nodes: List[Node], elements: List[Element]):
        self.nodes = nodes
        self.elements = elements
        self.dof_map = self._create_dof_map()
        self.num_dof = len(self.dof_map)
        self.K_global = np.zeros((self.num_dof, self.num_dof))
        self.F_global = np.zeros(self.num_dof)
        
        # Store initial restraints for result processing
        self.initial_restraints = {n.id: n.restraints for n in nodes}

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
        # Reset matrices
        self.K_global.fill(0.0)
        self.F_global.fill(0.0)
        
        for element in self.elements:
            T = get_transformation_matrix(element)
            K_local = get_element_stiffness_matrix(element)
            
            # Global Stiffness: K_global = T^T * K_local * T
            K_global_e = T.T @ K_local @ T
            
            # Map element DOFs (12) to global DOFs
            node_ids = [element.start_node.id, element.end_node.id]
            global_indices = []
            for node_id in node_ids:
                for i in range(6):
                    global_indices.append(self.dof_map[(node_id, i)])

            # Assembly: Add element stiffness to global matrix
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
        Kfs = self.K_global[np.ix_(free_dof, support_dof)]
        Ksf = self.K_global[np.ix_(support_dof, free_dof)]
        Kss = self.K_global[np.ix_(support_dof, support_dof)]
        
        Ff = self.F_global[free_dof]
        Fs = self.F_global[support_dof]
        
        # Displacements at supports (Ds) are zero for fixed/pinned/roller
        Ds = np.zeros(len(support_dof))

        # Solve for free displacements (Df): Kff * Df = Ff - Kfs * Ds
        # Since Ds = 0, Kff * Df = Ff
        Df = np.linalg.solve(Kff, Ff)

        # Reconstruct full displacement vector D
        D_full = np.zeros(self.num_dof)
        D_full[free_dof] = Df
        D_full[support_dof] = Ds
        
        # Calculate support reactions (Rs): Rs = Ksf * Df + Kss * Ds - Fs
        # Since Ds = 0, Rs = Ksf * Df - Fs
        Rs = Ksf @ Df - Fs

        # Distribute results back to nodes
        for node in self.nodes:
            for i in range(6):
                global_dof = self.dof_map[(node.id, i)]
                
                # Displacement
                node.displacements[i] = D_full[global_dof]
                
                # Reaction (only for restrained DOFs)
                if node.restraints[i]:
                    idx = support_dof.index(global_dof)
                    node.reactions[i] = Rs[idx]
                else:
                    node.reactions[i] = 0.0 # Ensure non-restrained reactions are zero

        return D_full

# --- 4. Geometry Generation and Load Application (IS Code Proxies) ---

def generate_default_structure(params) -> Tuple[List[Node], List[Element]]:
    """Generates a simple 1-bay, 1-storey frame based on user parameters."""
    
    L = params['Bay Length']
    H = params['Storey Height']
    
    # 8 Nodes: 4 at base (z=0), 4 at roof (z=H)
    nodes = []
    node_coords = [
        (1, 0.0, 0.0, 0.0), (2, L, 0.0, 0.0), (3, L, L, 0.0), (4, 0.0, L, 0.0),
        (5, 0.0, 0.0, H), (6, L, 0.0, H), (7, L, L, H), (8, 0.0, L, H),
    ]
    for id, x, y, z in node_coords:
        node = Node(id, x, y, z)
        # Apply Fixed restraints at the base (nodes 1-4)
        if z == 0.0:
            node.restraints = [True] * 6 # Fully fixed
        nodes.append(node)
        
    node_map = {n.id: n for n in nodes}
    
    # Material and Section Properties (Assumed RC structure)
    E_c = 5000 * np.sqrt(params['fck']) * 1e6 # N/m^2 (Approximate E for concrete M20, fck=20MPa -> 22.3e9 N/m2)
    E = E_c / (1e6) # kN/m2 (approx 22.3e6 kN/m2)
    G = E / (2 * (1 + 0.2)) # Shear Modulus (v=0.2)
    # Corrected: Use density from parameters
    rho = params['rho'] # Density kN/m3 (RC)
    
    # Beam/Column section properties (Assumed 0.4x0.4m square sections)
    b, d = params['Section Size'], params['Section Size']
    A = b * d # Area
    Iy = (b * d**3) / 12.0 # Moment of inertia about Y
    Iz = (d * b**3) / 12.0 # Moment of inertia about Z
    J = Iy + Iz # Torsional constant (approx for square)
    
    element_props = {'E': E, 'G': G, 'A': A, 'Iy': Iy, 'Iz': Iz, 'J': J, 'rho': rho}
    
    elements = []
    elem_counter = 1
    
    # Columns (Vertical elements: 1-5, 2-6, 3-7, 4-8)
    column_pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]
    for n_i, n_j in column_pairs:
        elements.append(Element(elem_counter, node_map[n_i], node_map[n_j], element_props.copy()))
        elem_counter += 1
        
    # Beams (Horizontal elements: 5-6, 6-7, 7-8, 8-5)
    beam_pairs = [(5, 6), (6, 7), (7, 8), (8, 5)]
    for n_i, n_j in beam_pairs:
        elements.append(Element(elem_counter, node_map[n_i], node_map[n_j], element_props.copy()))
        elem_counter += 1
        
    return nodes, elements

def apply_dead_load(nodes, elements, params):
    """Calculates self-weight (dead load) and applies it to nodes."""
    # Reset nodal loads
    for n in nodes:
        n.load.fill(0.0)

    # Distributed self-weight (q) = density * area (kN/m)
    q = params['rho'] * params['Section Size']**2 # This now correctly accesses 'rho'
    
    # Apply load to nodes (Pz)
    for element in elements:
        L = element.length
        # Assume element is under vertical gravity load (-Z direction)
        # Nodal equivalent forces (approximate P_i = qL/2)
        P_z = -q * L / 2.0 # P_z is load in global Z direction

        element.start_node.load[2] += P_z
        element.end_node.load[2] += P_z
        
    # Also add slab/finish load to top beams (approximate by adding point load)
    # Total slab load = (2.5 kN/m2 * L * L) / 4 nodes at roof
    roof_load_per_node = -params['Slab DL (kN/m2)'] * params['Bay Length']**2 / 4.0
    
    for n in nodes:
        if n.z == params['Storey Height']:
            n.load[2] += roof_load_per_node

def apply_live_load(nodes, elements, params):
    """Applies simplified live load to the roof nodes."""
    # Only applies vertical Live Load (Pz) to roof nodes (z=H)
    roof_load_per_node = -params['Live Load (kN/m2)'] * params['Bay Length']**2 / 4.0
    
    # Reset existing Live Load contribution
    for n in nodes:
        n.load.fill(0.0)
        
    for n in nodes:
        if n.z == params['Storey Height']:
            n.load[2] += roof_load_per_node

def apply_wind_load(nodes, elements, params, direction='X'):
    """Applies simplified lateral (wind) load to the nodes."""
    
    # Wind Pressure (Pz) = 0.6 * Vz^2 (Simplified IS 875 Part 3 approximation)
    Vb = params['Wind Speed (m/s)'] # Basic wind speed
    Vz = Vb * 1.0 # Assuming Kd=1, Kt=1, Ka=1 (simplification)
    P_wind = 0.6 * Vz**2 * 1e-3 # kN/m2
    
    # Area tributary to roof nodes (A = L * H / 2)
    trib_area = params['Bay Length'] * params['Storey Height'] / 2.0
    P_node = P_wind * trib_area # Nodal force magnitude
    
    # Reset existing Wind Load contribution
    for n in nodes:
        n.load.fill(0.0)

    for n in nodes:
        if n.z == params['Storey Height']:
            if direction == 'X':
                # Apply load in +X direction (index 0)
                n.load[0] += P_node / 4.0 # Distributed to 4 roof nodes
            elif direction == 'Y':
                # Apply load in +Y direction (index 1)
                n.load[1] += P_node / 4.0

# --- 5. Main Analysis & Visualization Logic ---

REACTION_SCALE = 2.0 # Default visualization scale

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
        line=dict(color='rgba(40, 40, 40, 0.8)', width=4),
        name='Structural Elements',
        hoverinfo='none'
    ))

    # --- 2. Plot Nodes (Markers) ---
    node_x = [n.x for n in nodes]
    node_y = [n.y for n in nodes]
    node_z = [n.z for n in nodes]
    node_text = [f"Node {n.id}<br>({n.x}, {n.y}, {n.z})" for n in nodes]
    # Highlight fixed supports (base nodes)
    node_colors = ['#c0392b' if all(n.restraints) else '#2ecc71' for n in nodes] 

    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(size=6, color=node_colors, symbol='circle', line=dict(width=1, color='Black')),
        text=[n.id for n in nodes],
        textfont=dict(color="black", size=10),
        textposition='bottom center',
        hovertext=node_text,
        hoverinfo='text',
        name='Nodes'
    ))

    # --- 3. Plot Support Reactions (Vectors) ---
    reaction_traces = []
    
    # Forces Fx, Fy, Fz are indices 0, 1, 2 in n.reactions
    reaction_props = [
        (0, 'red', 'Reaction Fx (kN)'),
        (1, 'green', 'Reaction Fy (kN)'),
        (2, 'blue', 'Reaction Fz (kN)'),
    ]

    for index, color, name in reaction_props:
        rx_x, rx_y, rx_z = [], [], []
        arrow_hover_text = []

        for n in nodes:
            # Check if restrained in this DOF AND reaction is significant
            if n.restraints[index] and abs(n.reactions[index]) > 1e-3: 
                start_point = [n.x, n.y, n.z]
                # Reaction vector: Magnitude is scaled for visibility
                reaction_magnitude = n.reactions[index] 
                
                end_point = list(start_point)
                
                # Determine the axis for the reaction (0=X, 1=Y, 2=Z)
                # The visual arrow direction is OPPOSITE the calculated reaction for clarity.
                end_point[index] += -reaction_magnitude * scale / 25 

                # Add coordinates for the vector line
                rx_x.extend([start_point[0], end_point[0], None])
                rx_y.extend([start_point[1], end_point[1], None])
                rx_z.extend([start_point[2], end_point[2], None])
                
                # Store hover text for the arrow head
                arrow_hover_text.append(f"Node {n.id}<br>{name}: {reaction_magnitude:.2f} kN")


        if rx_x:
            # Line trace for the reaction vector
            reaction_traces.append(go.Scatter3d(
                x=rx_x, y=rx_y, z=rx_z,
                mode='lines',
                line=dict(color=color, width=5),
                name=name,
                showlegend=True,
                hoverinfo='none'
            ))

            # Arrow head markers (using diamond symbol)
            # Filter out the None separators to get end points
            arrow_x = [rx_x[i] for i in range(1, len(rx_x), 3) if rx_x[i] is not None]
            arrow_y = [rx_y[i] for i in range(1, len(rx_y), 3) if rx_y[i] is not None]
            arrow_z = [rx_z[i] for i in range(1, len(rx_z), 3) if rx_z[i] is not None]

            reaction_traces.append(go.Scatter3d(
                x=arrow_x, y=arrow_y, z=arrow_z,
                mode='markers',
                marker=dict(size=6, color=color, symbol='diamond'),
                name=f"{name} (Value)",
                showlegend=False,
                hovertext=arrow_hover_text,
                hoverinfo='text'
            ))

    fig.add_traces(reaction_traces)

    # --- 4. Layout Configuration ---
    
    # Calculate bounds for cubic aspect ratio
    node_x_coords = [n.x for n in nodes]
    node_y_coords = [n.y for n in nodes]
    node_z_coords = [n.z for n in nodes]
    
    max_x, min_x = max(node_x_coords), min(node_x_coords)
    max_y, min_y = max(node_y_coords), min(node_y_coords)
    max_z, min_z = max(node_z_coords), min(node_z_coords)
    
    range_x = max_x - min_x
    range_y = max_y - min_y
    range_z = max_z - min_z
    max_range = max(range_x, range_y, range_z)
    
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    center_z = (max_z + min_z) / 2

    # Set axis limits to ensure a cubic view regardless of model shape
    x_range = [center_x - max_range/2, center_x + max_range/2]
    y_range = [center_y - max_range/2, center_y + max_range/2]
    z_range = [center_z - max_range/2, center_z + max_range/2]
    
    fig.update_layout(
        title=f'Interactive 3D Frame View: Load Case {st.session_state["current_load_case"]}',
        height=700,
        scene=dict(
            xaxis=dict(title='X Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True, range=x_range),
            yaxis=dict(title='Y Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True, range=y_range),
            zaxis=dict(title='Z Axis (m)', backgroundcolor="#f0f0f0", gridcolor="white", showbackground=True, range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1), # Set cubic aspect ratio
            camera=dict(
                up=dict(x=0, y=0, z=1), # Z is up
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5) # Default viewing angle
            )
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

# --- 6. Streamlit App Layout and Data Management ---

def run_analysis(nodes: List[Node], elements: List[Element], case: str, params: dict):
    """Calculates loads, runs the FEA solver, and stores results."""
    
    # 1. Reset loads on all nodes
    for n in nodes:
        n.load.fill(0.0)

    # 2. Apply load for the selected case
    if case == 'DL (Dead Load)':
        apply_dead_load(nodes, elements, params)
    elif case == 'LL (Live Load)':
        apply_live_load(nodes, elements, params)
    elif case == 'WLX (Wind Load +X)':
        apply_wind_load(nodes, elements, params, 'X')
    elif case == 'WLY (Wind Load +Y)':
        apply_wind_load(nodes, elements, params, 'Y')
    else:
        # --- Handle Load Combinations ---
        
        # Helper function to generate and retrieve loads for a specific case
        def get_case_loads(case_type, current_params):
            temp_nodes, _ = generate_default_structure(current_params)
            
            # Apply appropriate loading function
            if case_type == 'DL':
                apply_dead_load(temp_nodes, [], current_params) # Elements aren't strictly needed here for nodal load summation
            elif case_type == 'LL':
                apply_live_load(temp_nodes, [], current_params)
            elif case_type == 'WLX':
                apply_wind_load(temp_nodes, [], current_params, 'X')
            
            return {n.id: n.load for n in temp_nodes}
            
        if case == '1.5(DL+LL)':
            dl_loads = get_case_loads('DL', params)
            ll_loads = get_case_loads('LL', params)
            
            for n in nodes:
                n.load = 1.5 * (dl_loads[n.id] + ll_loads[n.id])
            
        elif case == '1.5(DL+WLX)':
            dl_loads = get_case_loads('DL', params)
            wlx_loads = get_case_loads('WLX', params)
            
            for n in nodes:
                n.load = 1.5 * (dl_loads[n.id] + wlx_loads[n.id])
        # Add more load combinations as needed...

    # 3. Run Solver
    try:
        structure = Structure(nodes, elements)
        structure.assemble_matrices()
        structure.solve()
        st.session_state['analysis_success'] = True
    except np.linalg.LinAlgError:
        st.session_state['analysis_success'] = False
        st.error("Analysis Failed: Singular matrix detected. Check structure stability or restraints.")
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

st.title("Full 3D Frame Analysis and Visualization (FEA & IS Code Demo)")
st.markdown("This application performs a simplified 3D Finite Element Analysis on a single-bay, single-story frame.")
st.markdown("---")

# --- Sidebar for Parameters and Controls ---
with st.sidebar:
    st.header("1. Geometry and Material")
    
    # Input Parameters
    BAY_LENGTH = st.number_input("Bay Length (L) (m)", min_value=1.0, value=10.0, step=1.0, key='L')
    STOREY_HEIGHT = st.number_input("Storey Height (H) (m)", min_value=1.0, value=5.0, step=0.5, key='H')
    SECTION_SIZE = st.number_input("Square Section Size (b=d) (m)", min_value=0.1, value=0.4, step=0.05, format="%.2f", key='b')
    FCK = st.number_input("Concrete Grade (fck) (MPa)", min_value=15, value=20, step=5, key='fck')
    
    # FIX: Added Density Input for self-weight calculation
    DENSITY_RC = st.number_input("RC Density (rho) (kN/m³)", min_value=10.0, value=25.0, step=1.0, key='rho_input', help="Used for calculating element self-weight (Dead Load).")
    
    st.header("2. Load Inputs (Simplified IS Code)")
    SLAB_DL = st.number_input("Slab Dead Load (kN/m²)", min_value=0.5, value=2.5, step=0.5, key='slab_dl')
    LIVE_LOAD = st.number_input("Live Load (kN/m²)", min_value=0.5, value=3.0, step=0.5, key='ll')
    WIND_SPEED = st.number_input("Basic Wind Speed (Vb) (m/s)", min_value=10, value=39, step=1, key='wind_speed')
    
    st.header("3. Analysis Control")
    LOAD_CASES = [
        'DL (Dead Load)', 'LL (Live Load)', 
        'WLX (Wind Load +X)', 'WLY (Wind Load +Y)', 
        '1.5(DL+LL)', '1.5(DL+WLX)'
    ]
    
    selected_case = st.selectbox("Select Load Case to Analyze", options=LOAD_CASES, key='case_select')
    
    if st.button("Run Full Analysis", use_container_width=True):
        # 1. Collect all parameters
        analysis_params = {
            'Bay Length': BAY_LENGTH,
            'Storey Height': STOREY_HEIGHT,
            'Section Size': SECTION_SIZE,
            'fck': FCK,
            'rho': DENSITY_RC, # FIX: Added rho to params
            'Slab DL (kN/m2)': SLAB_DL,
            'Live Load (kN/m2)': LIVE_LOAD,
            'Wind Speed (m/s)': WIND_SPEED,
        }
        
        # 2. Generate new structure based on current parameters
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
    st.info("Click 'Run Full Analysis' in the sidebar to generate the structure and perform the FEA calculation.")
elif st.session_state['analysis_success']:
    
    # 3D View is now the primary and only visualization
    st.header(f"3D Model View: {st.session_state['current_load_case']}")
    fig = plot_3d_frame_with_reactions(
        st.session_state['current_nodes'], 
        st.session_state['current_elements'], 
        st.session_state['reaction_scale']
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Tabs for detailed results
    tab_reactions, tab_displacements, tab_internal_forces = st.tabs([
        "Support Reactions (Calculated)", 
        "Nodal Displacements", 
        "Detailed Element Forces (Not Implemented)"
    ])

    with tab_reactions:
        st.header("Calculated Support Reactions")
        
        support_nodes_data = [
            {'Node ID': n.id, 
             'Fx (kN)': f"{n.reactions[0]:.2f}", 
             'Fy (kN)': f"{n.reactions[1]:.2f}", 
             'Fz (kN)': f"{n.reactions[2]:.2f}",
             'Mx (kNm)': f"{n.reactions[3]:.2f}", 
             'My (kNm)': f"{n.reactions[4]:.2f}", 
             'Mz (kNm)': f"{n.reactions[5]:.2f}"}
            for n in st.session_state['current_nodes'] if any(n.restraints)
        ]

        if support_nodes_data:
            df_reactions = pd.DataFrame(support_nodes_data)
            st.dataframe(df_reactions.set_index('Node ID'), use_container_width=True)
        else:
            st.info("No supported (restrained) nodes found in the structure data.")

    with tab_displacements:
        st.header("Nodal Displacements")
        
        displacement_data = [
            {'Node ID': n.id, 
             'dx (mm)': f"{n.displacements[0] * 1000:.4f}", # Convert to mm
             'dy (mm)': f"{n.displacements[1] * 1000:.4f}", # Convert to mm
             'dz (mm)': f"{n.displacements[2] * 1000:.4f}", # Convert to mm
             'rx (rad)': f"{n.displacements[3]:.4e}",
             'ry (rad)': f"{n.displacements[4]:.4e}",
             'rz (rad)': f"{n.displacements[5]:.4e}"}
            for n in st.session_state['current_nodes']
        ]

        df_displacements = pd.DataFrame(displacement_data)
        st.dataframe(df_displacements.set_index('Node ID'), use_container_width=True)

    with tab_internal_forces:
        st.info("To show internal element forces (Axial, Shear, Moment), we would need to back-calculate them from the global displacements, which is complex and requires more code. This section is reserved for future expansion.")
    
# --- End of Streamlit App Layout ---
