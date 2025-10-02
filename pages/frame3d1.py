import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Improved 3D Frame Analyzer")

# --- 1. Object-Oriented Model for the Structure ---
class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.restraints = [False] * 6
        self.reactions = [0.0] * 6

    def __repr__(self):
        return f"Node(id={self.id}, pos=({self.x}, {self.y}, {self.z}))"

class Element:
    """Represents a single beam/column element in the 3D structure."""
    def __init__(self, id, start_node, end_node, props):
        self.id = int(id)
        self.start_node = start_node
        self.end_node = end_node
        self.props = props 
        self.length = self.calculate_length()
        self.results = {} 

    def calculate_length(self):
        """Calculates the element's length based on its node coordinates."""
        return np.sqrt(
            (self.end_node.x - self.start_node.x)**2 +
            (self.end_node.y - self.start_node.y)**2 +
            (self.end_node.z - self.start_node.z)**2
        )

    # ... [Element stiffness and transformation matrix methods are unchanged] ...
    def get_local_stiffness_matrix(self):
        """Calculates the 12x12 local stiffness matrix for a 3D frame element."""
        E, G = self.props['E'], self.props['G']
        A, Iyy, Izz, J = self.props['A'], self.props['Iyy'], self.props['Izz'], self.props['J']
        L = self.length
        L2, L3 = L**2, L**3

        k = np.zeros((12, 12))
        if L == 0: return k

        # Populate the matrix based on standard 3D beam-column theory
        k[0,0] = k[6,6] = E*A/L
        k[0,6] = k[6,0] = -E*A/L

        k[1,1] = k[7,7] = 12*E*Izz/L3
        k[1,7] = k[7,1] = -12*E*Izz/L3
        k[1,5] = k[5,1] = 6*E*Izz/L2
        k[7,5] = k[5,7] = -6*E*Izz/L2
        k[7,11] = k[11,7] = -6*E*Izz/L2
        k[1,11] = k[11,1] = 6*E*Izz/L2
        k[5,11] = k[11,5] = 2*E*Izz/L

        k[2,2] = k[8,8] = 12*E*Iyy/L3
        k[2,8] = k[8,2] = -12*E*Iyy/L3
        k[2,4] = k[4,2] = 6*E*Iyy/L2
        k[8,4] = k[4,8] = -6*E*Iyy/L2
        k[8,10] = k[10,8] = -6*E*Iyy/L2
        k[2,10] = k[10,2] = 6*E*Iyy/L2
        k[4,10] = k[10,4] = 2*E*Iyy/L

        k[3,3] = k[9,9] = G*J/L
        k[3,9] = k[9,3] = -G*J/L

        k[4,4] = k[10,10] = 4*E*Iyy/L
        k[4,10] = k[10,4] = 2*E*Iyy/L

        k[5,5] = k[11,11] = 4*E*Izz/L
        k[5,11] = k[11,5] = 2*E*Izz/L

        # Populate symmetric parts
        for i in range(12):
            for j in range(i + 1, 12):
                k[j, i] = k[i, j]
        
        # NOTE: Sign corrections applied for off-diagonal terms based on convention (using explicit terms above)
        k[1,7] = k[7,1] = -12*E*Izz/L3
        k[2,8] = k[8,2] = -12*E*Iyy/L3
        k[1,5] = k[5,1] = k[7,11] = k[11,7] = 6*E*Izz/L2 # M-V terms
        k[1,11] = k[11,1] = -6*E*Izz/L2 # V-M terms
        k[2,4] = k[4,2] = k[8,10] = k[10,8] = 6*E*Iyy/L2 # M-V terms
        k[2,10] = k[10,2] = -6*E*Iyy/L2 # V-M terms

        return k

    def get_transformation_matrix(self):
        """Calculates the 12x12 transformation matrix T."""
        T = np.zeros((12, 12))
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        dz = self.end_node.z - self.start_node.z

        if self.length == 0: return np.identity(12)
        
        # Local x-axis direction cosines
        cx_x, cx_y, cx_z = dx / self.length, dy / self.length, dz / self.length

        # Determine reference vector for local z-axis (assuming orientation parallel to global Y-Z plane for beams, X-Z for columns)
        is_column = np.isclose(dx, 0) and np.isclose(dy, 0)
        if is_column:
            # Column: Z-axis aligned. If Z is up, local x-axis is [0, 0, 1]. Use [1, 0, 0] as ref to align local Y with global Y or X
            ref_vec = np.array([1, 0, 0])
        else:
            # Beam: Local z should be 'up' (parallel to global Z) if possible
            if np.isclose(dz, 0): # Horizontal beam
                ref_vec = np.array([0, 0, 1])
            else: # Ramped beam, use a horizontal vector
                 ref_vec = np.array([-dy, dx, 0]) # Horizontal vector orthogonal to the projection of x-axis

        
        local_x_vec = np.array([cx_x, cx_y, cx_z])
        # Find local z-axis: must be orthogonal to local x. 
        local_z_vec = np.cross(local_x_vec, ref_vec)
        
        # Handle the case where local_x_vec is parallel to ref_vec (e.g., column on Z-axis with ref [0, 0, 1])
        if np.linalg.norm(local_z_vec) < 1e-6:
             if np.isclose(cx_x, 1) or np.isclose(cx_x, -1): ref_vec = np.array([0, 0, 1])
             elif np.isclose(cx_y, 1) or np.isclose(cx_y, -1): ref_vec = np.array([1, 0, 0])
             else: ref_vec = np.array([1, 0, 0])
             local_z_vec = np.cross(local_x_vec, ref_vec)

        local_z_vec /= np.linalg.norm(local_z_vec)
        local_y_vec = np.cross(local_z_vec, local_x_vec) # Recalculate y-axis to ensure right-handed system
        
        R = np.vstack([local_x_vec, local_y_vec, local_z_vec])
        
        for i in range(4):
            T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        return T


class Structure:
    """Represents the entire 3D frame structure and handles the FEA."""
    def __init__(self):
        self.nodes, self.elements, self.dof_map = {}, {}, {}
        self.K_global, self.F_global, self.U_global = None, None, None

    # ... [Node, Element, Support, and Assembly methods are unchanged] ...
    def add_node(self, id, x, y, z):
        if id not in self.nodes: self.nodes[id] = Node(id, x, y, z)
        return self.nodes[id]

    def add_element(self, id, start_node_id, end_node_id, props):
        if id not in self.elements and start_node_id in self.nodes and end_node_id in self.nodes:
            self.elements[id] = Element(id, self.nodes[start_node_id], self.nodes[end_node_id], props)
        return self.elements.get(id)

    def set_support(self, node_id, restraints):
        if node_id in self.nodes: self.nodes[node_id].restraints = restraints

    def assemble_matrices(self):
        num_dof = len(self.nodes) * 6
        self.K_global, self.F_global = np.zeros((num_dof, num_dof)), np.zeros(num_dof)
        
        dof_index = 0
        for node_id in sorted(self.nodes.keys()):
            for i in range(6): self.dof_map[(node_id, i)] = dof_index; dof_index += 1

        for elem in self.elements.values():
            k_local, T = elem.get_local_stiffness_matrix(), elem.get_transformation_matrix()
            k_global_elem = T @ k_local @ T.T
            
            node_ids = [elem.start_node.id, elem.end_node.id]
            dof_indices = [self.dof_map[(nid, i)] for nid in node_ids for i in range(6)]
            
            for i, global_i in enumerate(dof_indices):
                for j, global_j in enumerate(dof_indices):
                    self.K_global[global_i, global_j] += k_global_elem[i, j]

    def add_gravity_loads(self, q_gravity, levels):
        # Apply load in -Z direction (index 2 for Fz)
        for z in levels:
            level_nodes = {n.id for n in self.nodes.values() if np.isclose(n.z, z)}
            if not level_nodes: continue
            
            # Find horizontal beams at this level
            level_beams = [e for e in self.elements.values() if e.start_node.z == e.end_node.z and e.start_node.id in level_nodes and e.end_node.id in level_nodes]
            for beam in level_beams:
                load_at_node = q_gravity * beam.length / 2
                # Apply load in -Z direction (index 2 for Fz)
                self.F_global[self.dof_map[(beam.start_node.id, 2)]] -= load_at_node
                self.F_global[self.dof_map[(beam.end_node.id, 2)]] -= load_at_node

    # NEW FUNCTION: Apply uniformly distributed wall loads
    def add_wall_loads(self, wall_load_data):
        """Applies UDL loads on specified element IDs as equivalent nodal forces (Fz)."""
        for elem_id, load_q in wall_load_data.items():
            if elem_id in self.elements:
                beam = self.elements[elem_id]
                load_at_node = load_q * beam.length / 2
                # Apply load in -Z direction (index 2 for Fz)
                self.F_global[self.dof_map[(beam.start_node.id, 2)]] -= load_at_node
                self.F_global[self.dof_map[(beam.end_node.id, 2)]] -= load_at_node

    def solve(self):
        active_dofs = [self.dof_map[(n.id, i)] for n in self.nodes.values() for i in range(6) if not n.restraints[i]]
        active_dofs = np.array(active_dofs)
        K_reduced, F_reduced = self.K_global[active_dofs[:, np.newaxis], active_dofs], self.F_global[active_dofs]
        try:
            U_reduced = np.linalg.solve(K_reduced, F_reduced)
            self.U_global = np.zeros_like(self.F_global)
            self.U_global[active_dofs] = U_reduced
            return True, "Analysis successful."
        except np.linalg.LinAlgError:
            self.U_global = None
            return False, "Analysis failed. The structure may be unstable (singular matrix)."

    def calculate_element_results(self):
        if self.U_global is None: return
        keys = ['Axial_Start', 'Shear_Y_Start', 'Shear_Z_Start', 'Torsion_Start', 'Moment_Y_Start', 'Moment_Z_Start',
                'Axial_End', 'Shear_Y_End', 'Shear_Z_End', 'Torsion_End', 'Moment_Y_End', 'Moment_Z_End']
        for elem in self.elements.values():
            dof_indices = [self.dof_map[(nid, i)] for nid in [elem.start_node.id, elem.end_node.id] for i in range(6)]
            u_global_elem = self.U_global[dof_indices]
            u_local_elem = elem.get_transformation_matrix().T @ u_global_elem
            f_local = elem.get_local_stiffness_matrix() @ u_local_elem
            
            elem.results = {keys[i]: f_local[i] for i in range(12)}
            # Keep only Moment_Z (Myz in 3D frame, but Mz in the local beam coordinate is the main bending moment for beams in X-Y plane)
            # Use Moment_Y for columns bent about their strong axis (Y-Y in local coordinates if Z is vertical)
            elem.results['Max_Abs_Moment'] = max(abs(f_local[4]), abs(f_local[5]), abs(f_local[10]), abs(f_local[11]))
            # Also store displacement for 2D deflection plot
            elem.results['Disp_Global_Start'] = u_global_elem[:6]
            elem.results['Disp_Global_End'] = u_global_elem[6:]

    def calculate_reactions(self):
        if self.U_global is None: return
        R = self.K_global @ self.U_global - self.F_global
        for node in self.nodes.values():
            if any(node.restraints):
                for i in range(6):
                    if node.restraints[i]: node.reactions[i] = R[self.dof_map[(node.id, i)]]

# --- 2. Utility & Calculation Functions ---

def parse_grid_input(input_string):
    if not input_string: return []
    lengths = []
    for segment in [s.strip() for s in input_string.split(',') if s.strip()]:
        match = re.match(r'^(\d+)x([0-9.]+)$', segment)
        if match:
            count, length = int(match.group(1)), float(match.group(2))
            if count > 0 and length > 0: lengths.extend([length] * count)
        else:
            try:
                if float(segment) > 0: lengths.append(float(segment))
            except ValueError: pass
    return lengths

def calculate_rc_properties(b, h, E, nu=0.2):
    A, Izz, Iyy, G = b*h, (b*h**3)/12, (h*b**3)/12, E/(2*(1+nu))
    a, c = max(b, h), min(b, h)
    J = a*(c**3)*(1/3 - 0.21*(c/a)*(1-(c**4)/(12*a**4)))
    return {'E':E, 'G':G, 'A':A, 'Iyy':Iyy, 'Izz':Izz, 'J':J}

# NEW FUNCTION: Parses the brick load input string
def parse_and_apply_wall_loads(load_string):
    """Parses a string like '2, 3, 4: 5\n10, 12: 7' into a dictionary of {elem_id: load_q}."""
    wall_load_data = {}
    if not load_string: return wall_load_data

    for line in load_string.strip().split('\n'):
        line = line.strip()
        if not line or ':' not in line: continue

        parts = line.split(':')
        elem_ids_str = parts[0].strip()
        load_q_str = parts[1].strip()
        
        try:
            load_q = float(load_q_str)
            if load_q <= 0: continue
        except ValueError:
            continue
        
        elem_ids = []
        for segment in elem_ids_str.split(','):
            try:
                elem_ids.append(int(segment.strip()))
            except ValueError:
                pass
        
        for elem_id in elem_ids:
            wall_load_data[elem_id] = load_q
            
    return wall_load_data


# --- 3. Streamlit Caching ---

@st.cache_data
def generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, load_params, wall_load_data):
    s = Structure()
    x_coords, y_coords, z_coords = [0]+list(np.cumsum(x_dims)), [0]+list(np.cumsum(y_dims)), [0]+list(np.cumsum(z_dims))
    node_id, elem_id, node_map = 1, 1, {}

    # 1. Create Nodes
    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                s.add_node(node_id, x, y, z)
                node_map[(ix, iy, iz)] = node_id
                if np.isclose(z, 0): s.set_support(node_id, restraints=[True]*6)
                node_id += 1

    # 2. Create Elements (Columns)
    for iz in range(len(z_coords)-1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)): 
                s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy,iz+1)], col_props)
                elem_id += 1
    
    # 3. Create Elements (Beams in X-direction)
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)-1): 
                s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix+1,iy,iz)], beam_props)
                elem_id += 1
    
    # 4. Create Elements (Beams in Y-direction)
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)-1):
            for ix in range(len(x_coords)): 
                s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy+1,iz)], beam_props)
                elem_id += 1
    
    # 5. Assemble and Apply Loads
    s.assemble_matrices()
    # Gravity load is typically applied on the whole frame based on slab/live load
    s.add_gravity_loads(load_params['q_total_gravity'], z_coords[1:])
    # Specific uniform loads for the 2D case from the user's sketch
    s.add_wall_loads(wall_load_data) 
    
    # 6. Solve and Calculate Results
    success, message = s.solve()
    if success: 
        s.calculate_element_results()
        s.calculate_reactions()
    
    if not success: return {'success': False, 'message': message}

    # 7. Package Results
    return {
        'success': True, 'message': message,
        'nodes': [{'id':n.id, 'x':n.x, 'y':n.y, 'z':n.z, 'restraints':n.restraints, 'reactions':n.reactions} for n in s.nodes.values()],
        'elements': [{'id':e.id, 'start_node_id':e.start_node.id, 'end_node_id':e.end_node.id, 'start_node_pos':(e.start_node.x,e.start_node.y,e.start_node.z), 'end_node_pos':(e.end_node.x,e.end_node.y,e.end_node.z), 'length':e.length, 'results':e.results} for e in s.elements.values()],
        'summary': {'num_nodes':len(s.nodes), 'num_elements':len(s.elements), 'k_shape':s.K_global.shape if s.K_global is not None else (0,0)}
    }

# --- 4. Plotting Functions ---

def get_hover_text(elem):
    """Generates detailed hover text for an element."""
    text = f"**Element {elem['id']}** (L={elem.get('length', 0.0):.2f}m)<br>"
    for key, value in elem['results'].items():
        if not key.startswith('Disp'):
            unit = 'kNm' if key.startswith('Moment') or key.startswith('Torsion') else 'kN'
            text += f"{key.replace('_', ' ')}: {value:.2f} {unit}<br>"
    return text

def plot_3d_frame(nodes, elements, display_mode='Structure', show_nodes=False, show_elems=False):
    # This function is unchanged from the previous version, but is kept for completeness.
    # ... [Omitted for brevity] ...
    fig = go.Figure()
    
    # 1. Structure/Moment Lines (Using individual traces for color/width control)
    max_abs_result = 1 # Avoid division by zero
    result_key = None
    
    if display_mode == 'Bending Moment (Myz)':
        max_abs_result = max((abs(e['results'].get('Max_Abs_Moment', 0)) for e in elements), default=0)
        result_key = 'Max_Abs_Moment'
    elif display_mode == 'Axial Force (Fx)':
        max_abs_result = max((abs(e['results'].get('Axial_End', 0)) for e in elements), default=0)
        result_key = 'Axial_End'

    for elem in elements:
        start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']

        line_color = 'darkblue'
        line_width = 4
        
        if result_key:
            result_val = elem['results'].get(result_key, 0)
            normalized_val = result_val / max_abs_result if max_abs_result > 0 else 0
            
            if display_mode == 'Bending Moment (Myz)':
                color_val = int(255 * (abs(normalized_val) ** 0.5))
                line_color = f'rgb({color_val}, 50, 50)'
            
            elif display_mode == 'Axial Force (Fx)':
                if np.isclose(result_val, 0):
                    line_color = 'gray'
                elif result_val > 0: 
                    line_color = f'rgb({int(255*normalized_val)}, 50, 50)'
                else: 
                    line_color = f'rgb(50, 50, {int(255*abs(normalized_val))})'
                    
            line_width = 5
        
        fig.add_trace(go.Scatter3d(x=[start_pos[0],end_pos[0]], y=[start_pos[1],end_pos[1]], z=[start_pos[2],end_pos[2]], 
                                   mode='lines', line=dict(color=line_color, width=line_width), 
                                   hoverinfo='text', hovertext=get_hover_text(elem), name=f"Elem {elem['id']}", showlegend=False))

    node_x, node_y, node_z = [n['x'] for n in nodes], [n['y'] for n in nodes], [n['z'] for n in nodes]
    node_texts = [f"Node {n['id']}" for n in nodes]
    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(size=5, color='purple'), 
                               name='Nodes', text=node_texts, hoverinfo='text', showlegend=False))
    
    if show_nodes:
        fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='text', text=[str(n['id']) for n in nodes], 
                                   textfont=dict(color='black', size=10), name='Node IDs', showlegend=False))

    if show_elems:
        elem_x, elem_y, elem_z = [], [], []
        for elem in elements:
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            elem_x.append((start_pos[0] + end_pos[0]) / 2)
            elem_y.append((start_pos[1] + end_pos[1]) / 2)
            elem_z.append((start_pos[2] + end_pos[2]) / 2)
        fig.add_trace(go.Scatter3d(x=elem_x, y=elem_y, z=elem_z, mode='text', text=[str(e['id']) for e in elements], 
                                   textfont=dict(color='darkred', size=10), name='Element IDs', showlegend=False))

    fig.update_layout(title=f"3D Frame Visualization - {display_mode}", 
                      scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', 
                                 aspectratio=dict(x=1.5, y=1.5, z=1)), 
                      margin=dict(l=0,r=0,b=0,t=40), showlegend=False)
    return fig


# UPDATED FUNCTION: Plot 2D frame with result diagrams
def plot_2d_frame(nodes, elements, plane_axis, coordinate, display_mode='Structure'):
    fig = go.Figure()
    
    # 1. Filter Nodes and Elements for the 2D plane
    if plane_axis == 'Y': 
        plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['y'], coordinate)], 'x', 'z'
    else: 
        plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['x'], coordinate)], 'y', 'z'
        
    plane_node_ids = {n['id'] for n in plane_nodes_list}

    # Map the display mode to the element result key and plotting parameters
    result_map = {
        'Structure': (None, None, None, 1),
        'Bending Moment': ('Moment_Z', 'kNm', 'Deflection in X', 0.2), # Plot Mz as it's the main beam bending moment
        'Shear Force': ('Shear_Z', 'kN', 'Deflection in X', 0.2),  # Plot Sz as it's the main beam shear force
        'Axial Force': ('Axial', 'kN', 'Deflection in X', 0.2),
        'Deflection': (None, 'm', 'Deflection in Z', 100), # Plot Deflection
    }
    
    result_key_base, unit, defl_key, scale_factor = result_map.get(display_mode, (None, None, None, 1))

    # 2. Iterate and Plot Elements and Diagrams
    for elem in elements:
        if elem['start_node_id'] in plane_node_ids and elem['end_node_id'] in plane_node_ids:
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            
            start_x = start_pos[0 if plane_axis=='Y' else 1]
            end_x = end_pos[0 if plane_axis=='Y' else 1]
            z_coords = [start_pos[2], end_pos[2]]
            x_coords = [start_x, end_x]
            L = elem['length']
            
            # --- BASE ELEMENT LINE ---
            fig.add_trace(go.Scatter(x=x_coords, y=z_coords, mode='lines', line=dict(color='darkblue', width=3), 
                                     hoverinfo='text', hovertext=get_hover_text(elem), showlegend=False))
            
            # --- RESULT DIAGRAMS ---
            if display_mode != 'Structure':
                
                # A. Prepare points for the curve
                num_points = 20
                local_x = np.linspace(0, L, num_points)
                diagram_coords = []
                
                # Function to calculate internal force/moment/deflection
                def calculate_internal_value(x, elem, key_base, defl_type):
                    # We use the end forces/moments (f_local) and displacements (u_local) for interpolation.
                    # This is a simplification; a full FEA plot requires interpolation functions (e.g., cubic for V and M).
                    # For this request, we'll draw straight lines between end points for force/moment,
                    # and an exaggerated deflection line.
                    
                    if display_mode in ['Bending Moment', 'Shear Force', 'Axial Force']:
                        # Simple linear interpolation between end forces (valid for constant or linear loads)
                        val_start = elem['results'].get(key_base + '_Start', 0)
                        val_end = elem['results'].get(key_base + '_End', 0)
                        
                        # In the case of UDL (as in your sketch), the moment diagram is parabolic. 
                        # This FEA only gives the end moments/shears.
                        # For the beam, we calculate the internal moment using the FEA end moments (M_end) and the UDL (w).
                        
                        # Assuming the beam is horizontal (common case for UDL) and load is Fz (vertical).
                        if L > 0 and np.isclose(start_pos[2], end_pos[2]):
                            w_eff = 0
                            # Find the applied load on this element (approximation for plotting)
                            for e_id, q in st.session_state.get('wall_load_data', {}).items():
                                if e_id == elem['id']: w_eff = q; break
                            
                            # Standard beam moment calculation: M(x) = M_start + V_start*x + w*x^2/2 (sign conventions vary)
                            M_start = elem['results'].get('Moment_Z_Start', 0)
                            V_start = elem['results'].get('Shear_Z_Start', 0)
                            
                            # Moment Diagram (Mz for beams in the X-Y plane)
                            if key_base == 'Moment_Z':
                                # M(x) = M_start + V_start * x + w_eff * x * (x/2 - L/2) (assuming global Z is up, local X is right)
                                return M_start + V_start * x + w_eff * x * (x/2 - L/2)
                            
                            # Shear Diagram (Sz for beams in the X-Y plane)
                            elif key_base == 'Shear_Z':
                                return V_start + w_eff * x # V(x) = V_start + w*x
                                
                            # Axial Force (Axial) is constant on a non-loaded member
                            elif key_base == 'Axial':
                                return val_start # Constant
                                
                        # Simple linear interpolation for all other cases (e.g., columns)
                        return val_start + (val_end - val_start) * (x / L) if L > 0 else 0
                        
                    elif display_mode == 'Deflection':
                        # Deflection in global X and Z directions.
                        # For a beam, the deflection is primarily in global Z (index 2) or global X (index 0 for columns)
                        # We use the Global Displacements (u_global) stored in results
                        u_start = elem['results'].get('Disp_Global_Start', np.zeros(6))
                        u_end = elem['results'].get('Disp_Global_End', np.zeros(6))
                        
                        # We need the relative position along the member (which involves T matrix), 
                        # but for 2D visualization, we can just linearly interpolate the Global displacements 
                        # and use a cubic interpolation function for better accuracy (which is complex).
                        
                        # Simplification: Use a linear interpolation of global displacement for visualization
                        if L > 0:
                            if defl_type == 'Deflection in Z':
                                disp_start, disp_end = u_start[2], u_end[2] # Global Uz
                            elif defl_type == 'Deflection in X':
                                disp_start, disp_end = u_start[0], u_end[0] # Global Ux
                            else: return 0
                            
                            return disp_start + (disp_end - disp_start) * (x / L)

                        return 0

                # B. Generate Diagram Plotting Coordinates
                for x in local_x:
                    value = calculate_internal_value(x, elem, result_key_base, defl_key)
                    
                    # Convert local x-position (x) and value to global plotting coordinates (x_plot, z_plot)
                    # For a beam in the X-Z plane (y=0):
                    # x_mid = start_x + x * (end_x - start_x) / L
                    # z_mid = start_z + x * (end_z - start_z) / L
                    # Normal vector (for plotting offset) is complicated.
                    
                    # SIMPLIFICATION: We only plot the diagram offset perpendicular to the member's projection.
                    
                    # Determine the member's orientation
                    is_horizontal = np.isclose(z_coords[0], z_coords[1])
                    
                    if is_horizontal:
                        # Horizontal member: diagram offset is in Z (vertical)
                        x_plot = start_x + x * (end_x - start_x) / L
                        z_plot = z_coords[0] + (value * scale_factor) # Offset in Z
                    else: # Vertical member (column)
                        # Vertical member: diagram offset is in X (horizontal)
                        x_plot = x_coords[0] + (value * scale_factor) # Offset in X
                        z_plot = z_coords[0] + x * (end_x - start_x) / L
                    
                    # Note on sign convention: The diagram is plotted opposite to the force/moment direction
                    # to follow standard engineering convention (e.g., tension-side for moment).
                    # We negate the value for plotting offset here.
                    diagram_coords.append((x_plot, z_plot))

                diag_x = [c[0] for c in diagram_coords]
                diag_z = [c[1] for c in diagram_coords]

                # Add diagram trace (Moment/Shear)
                if display_mode != 'Deflection':
                    fig.add_trace(go.Scatter(x=diag_x, y=diag_z, mode='lines', line=dict(color='red', width=2), 
                                             fill='tonexty' if is_horizontal else 'tozerox', opacity=0.3, # Fill for moment/shear
                                             hoverinfo='none', name=f"{display_mode} E{elem['id']}"))
                else:
                    # Deflection line (no fill)
                    fig.add_trace(go.Scatter(x=diag_x, y=diag_z, mode='lines', line=dict(color='orange', width=2, dash='dash'), 
                                             hoverinfo='none', name=f"Deflection E{elem['id']}"))

    # 3. Node Markers and Layout
    fig.add_trace(go.Scatter(x=[n[x_key] for n in plane_nodes_list], y=[n[z_key] for n in plane_nodes_list], 
                             mode='markers', marker=dict(size=8, color='purple'), name='Nodes', 
                             text=[f"Node {n['id']}" for n in plane_nodes_list], hoverinfo='text', showlegend=False))

    title_unit = f"({unit})" if unit else ''
    title_scale = f" (Scale: 1:{1/scale_factor:.0f})" if display_mode != 'Structure' and display_mode != 'Deflection' else ''
    title_scale_def = f" (Exaggerated Scale)" if display_mode == 'Deflection' else ''
    
    fig.update_layout(title=f"2D Elevation: {display_mode} {title_unit}{title_scale}{title_scale_def} on {plane_axis.replace('Y', 'X-Z').replace('X', 'Y-Z')} Plane at {plane_axis}={coordinate}m", 
                      xaxis_title=f'{x_key.upper()}-axis (m)', yaxis_title='Z-axis (m)', showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --- 5. Main Streamlit App UI ---
st.title("🏗️ 2D/3D Frame Analyzer")
st.write("Define your building grid, sections, and loads to generate and analyze a 3D frame.")

with st.sidebar:
    st.header("1. Frame Geometry (For 2D Sketch)")
    # Use inputs matching the 2D sketch for this run
    x_grid = st.text_input("X-spans (m)", "7.0") 
    y_grid = st.text_input("Y-spans (m)", "0.001") # Set a minimal Y-span to force X-Z plane
    z_grid = st.text_input("Z-heights (m)", "3.5, 3.5")
    
    st.header("2. Section & Material")
    E = st.number_input("E (GPa)", 200.0, help="200 GPa for Steel")*1e6 # Steel E for IPE section
    with st.expander("Column & Beam Sizes"):
        # IPE 240 is typical. We'll use a large rectangle as a simple approximation.
        col_b, col_h = st.number_input("Col b (mm)", 200)/1000, st.number_input("Col h (mm)", 200)/1000
        beam_b, beam_h = st.number_input("Beam b (mm)", 200)/1000, st.number_input("Beam h (mm)", 300)/1000
        
    st.header("3. Gravity Loads")
    with st.expander("Slab & Live Loads (Set to 0 for 2D Sketch Load Case)"):
        slab_d, slab_t = st.number_input("Slab Density (kN/m³)", 0.0), st.number_input("Slab Thickness (m)", 0.0)
        fin_l, live_l = st.number_input("Finishes (kN/m²)", 0.0), st.number_input("Live Load (kN/m²)", 0.0)
        
    st.subheader("4. Specific Uniform Loads (UDL)")
    # Input for the 10 kN/m on elements 3 and 4
    wall_load_input = st.text_area("Element Loads (e.g., 2, 3: 5 kN/m)", 
                                   value="3, 4: 10.0", 
                                   height=100,
                                   help="Enter Element IDs and the load magnitude (in kN/m applied in -Z direction).")
    
    analyze_button = st.button("Generate & Analyze Frame", type="primary")

if analyze_button:
    x_dims, y_dims, z_dims = parse_grid_input(x_grid), parse_grid_input(y_grid), parse_grid_input(z_grid)
    wall_load_data = parse_and_apply_wall_loads(wall_load_input)
    
    if not all([x_dims, y_dims, z_dims]): 
        st.error("Invalid grid input.")
    else:
        col_p, beam_p = calculate_rc_properties(col_b, col_h, E), calculate_rc_properties(beam_b, beam_h, E)
        q_total = slab_d*slab_t + fin_l + live_l
        
        with st.spinner("Running Finite Element Analysis..."):
            analysis_results = generate_and_analyze_structure(x_dims, y_dims, z_dims, col_p, beam_p, 
                                                              {'q_total_gravity': q_total}, wall_load_data)
                                                              
        if not analysis_results['success']: 
            st.error(f"Analysis Failed: {analysis_results['message']}")
        else:
            st.success("Analysis complete!")
            st.session_state['analysis_results'] = analysis_results
            st.session_state['wall_load_data'] = wall_load_data # Store load data for plotting

            st.subheader("FEA Results Summary")
            summary = analysis_results['summary']
            max_moment = max((abs(e['results'].get('Max_Abs_Moment',0)) for e in analysis_results['elements']), default=0)
            
            c1,c2,c3 = st.columns(3)
            c1.metric("Nodes", summary['num_nodes'])
            c1.metric("Elements", summary['num_elements'])
            c2.metric("Base Pressure", f"{q_total:.2f} kN/m²")
            c2.metric("Load Applied", f"{len(wall_load_data)} elements")
            c3.metric("Max Moment", f"{max_moment:.2f} kNm")


if 'analysis_results' in st.session_state and st.session_state['analysis_results']['success']:
    results, nodes, elements = st.session_state['analysis_results'], st.session_state['analysis_results']['nodes'], st.session_state['analysis_results']['elements']
    
    # 3D Visualization Section (Still useful for 3D checks)
    st.subheader("Interactive 3D Visualization")
    c_3d_1, c_3d_2 = st.columns([0.6, 0.4])
    with c_3d_1:
        display_mode_3d = st.selectbox("Display Mode", ['Structure', 'Bending Moment (Myz)', 'Axial Force (Fx)'], key='display_mode_3d')
    with c_3d_2:
        col_viz_1, col_viz_2 = st.columns(2)
        show_nodes_3d = col_viz_1.checkbox("Show Node IDs", key='show_nodes_3d')
        show_elems_3d = col_viz_2.checkbox("Show Element IDs", key='show_elems_3d')
    
    st.plotly_chart(plot_3d_frame(nodes, elements, display_mode_3d, show_nodes_3d, show_elems_3d), use_container_width=True)
    
    # Tabs for Detailed Results
    tab1, tab2, tab3 = st.tabs(["2D Elevation View", "Support Reactions", "Detailed Element Results"])
    
    with tab1:
        st.subheader("2D Elevation View and Result Diagrams")
        
        col_2d_1, col_2d_2 = st.columns(2)
        
        # Since the frame is narrow in Y, we default to the X-Z plane (Y-Gridline)
        plane_axis_display = col_2d_1.radio("Grid Plane", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'), key='plane_axis_radio')
        
        # New options for the required plots
        diagram_options = ['Structure', 'Bending Moment', 'Shear Force', 'Axial Force', 'Deflection']
        display_mode_2d = col_2d_2.selectbox("Result Diagram", diagram_options, key='display_mode_2d')
        
        col_2d_3, col_2d_4 = st.columns([0.6, 0.4])
        
        # Force the selection to the plane of the 2D frame
        if plane_axis_display == 'X-Z (Y-Gridline)':
            y_coords = sorted(list(set(n['y'] for n in nodes)))
            # Force selection of the only or first Y coordinate for the 2D frame
            selected_y = y_coords[0] if y_coords else 0
            plane_key = 'Y'
            coordinate = selected_y
        else:
            x_coords = sorted(list(set(n['x'] for n in nodes)))
            # Force selection of a valid X coordinate
            selected_x = x_coords[0] if x_coords else 0
            plane_key = 'X'
            coordinate = selected_x
        
        st.write(f"Showing results on the **{plane_key}-Gridline** at coordinate **{coordinate:.3f} m**.")
        
        st.plotly_chart(plot_2d_frame(nodes, elements, plane_key, coordinate, display_mode_2d), use_container_width=True)

    with tab2:
        st.subheader("Support Reactions (Global X, Y, Z)")
        support_nodes = {n['id']: n for n in nodes if any(n['restraints'])}
        if support_nodes:
            node_id = st.selectbox("Select support node", options=sorted(list(support_nodes.keys())), key='support_node_select')
            st.dataframe(pd.DataFrame({
                "DOF": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"], 
                "Value (kN, kNm)": support_nodes[node_id]['reactions']
            }).round(2), use_container_width=True)
        else: st.write("No support nodes found.")
        
    with tab3:
        st.subheader("All Element End Forces & Moments (Local Coordinates)")
        data = []
        for e in elements:
            res = e['results']
            data.append({
                'ID': e['id'], 
                'Start Node': e['start_node_id'], 
                'End Node': e['end_node_id'], 
                'Length (m)': e.get('length', 0.0), 
                'Max |M| (kNm)': res.get('Max_Abs_Moment', 0),
                'Axial Start (kN)': res.get('Axial_Start', 0), 
                'Axial End (kN)': res.get('Axial_End', 0),
                'Shear Y Start (kN)': res.get('Shear_Y_Start', 0),
                'Shear Y End (kN)': res.get('Shear_Y_End', 0),
                'Shear Z Start (kN)': res.get('Shear_Z_Start', 0),
                'Shear Z End (kN)': res.get('Shear_Z_End', 0),
                'Torsion Start (kNm)': res.get('Torsion_Start', 0),
                'Torsion End (kNm)': res.get('Torsion_End', 0),
                'Moment Y Start (kNm)': res.get('Moment_Y_Start', 0),
                'Moment Y End (kNm)': res.get('Moment_Y_End', 0),
                'Moment Z Start (kNm)': res.get('Moment_Z_Start', 0),
                'Moment Z End (kNm)': res.get('Moment_Z_End', 0),
            })
        st.dataframe(pd.DataFrame(data).round(2), use_container_width=True)
