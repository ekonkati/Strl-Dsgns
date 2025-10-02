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

    # --- Stiffness and Transformation Matrix methods (Omitted for brevity) ---
    def get_local_stiffness_matrix(self):
        E, G = self.props['E'], self.props['G']
        A, Iyy, Izz, J = self.props['A'], self.props['Iyy'], self.props['Izz'], self.props['J']
        L = self.length
        L2, L3 = L**2, L**3
        k = np.zeros((12, 12))
        if L == 0: return k
        k[0,0] = k[6,6] = E*A/L; k[0,6] = k[6,0] = -E*A/L
        k[1,1] = k[7,7] = 12*E*Izz/L3; k[1,7] = k[7,1] = -12*E*Izz/L3
        k[1,5] = k[5,1] = 6*E*Izz/L2; k[7,5] = k[5,7] = -6*E*Izz/L2
        k[7,11] = k[11,7] = -6*E*Izz/L2; k[1,11] = k[11,1] = 6*E*Izz/L2
        k[5,11] = k[11,5] = 2*E*Izz/L; k[2,2] = k[8,8] = 12*E*Iyy/L3
        k[2,8] = k[8,2] = -12*E*Iyy/L3; k[2,4] = k[4,2] = 6*E*Iyy/L2
        k[8,4] = k[4,8] = -6*E*Iyy/L2; k[8,10] = k[10,8] = -6*E*Iyy/L2
        k[2,10] = k[10,2] = 6*E*Iyy/L2; k[4,10] = k[10,4] = 2*E*Iyy/L
        k[3,3] = k[9,9] = G*J/L; k[3,9] = k[9,3] = -G*J/L
        k[4,4] = k[10,10] = 4*E*Iyy/L; k[4,10] = k[10,4] = 2*E*Iyy/L
        k[5,5] = k[11,11] = 4*E*Izz/L; k[5,11] = k[11,5] = 2*E*Izz/L
        for i in range(12):
            for j in range(i + 1, 12): k[j, i] = k[i, j]
        k[1,7] = k[7,1] = -12*E*Izz/L3; k[2,8] = k[8,2] = -12*E*Iyy/L3
        k[1,5] = k[5,1] = k[7,11] = k[11,7] = 6*E*Izz/L2; k[1,11] = k[11,1] = -6*E*Izz/L2
        k[2,4] = k[4,2] = k[8,10] = k[10,8] = 6*E*Iyy/L2; k[2,10] = k[10,2] = -6*E*Iyy/L2
        return k

    def get_transformation_matrix(self):
        T = np.zeros((12, 12))
        dx, dy, dz = self.end_node.x - self.start_node.x, self.end_node.y - self.start_node.y, self.end_node.z - self.start_node.z
        if self.length == 0: return np.identity(12)
        cx_x, cx_y, cx_z = dx / self.length, dy / self.length, dz / self.length
        is_column = np.isclose(dx, 0) and np.isclose(dy, 0)
        ref_vec = np.array([1, 0, 0]) if is_column else np.array([0, 0, 1])
        local_x_vec = np.array([cx_x, cx_y, cx_z])
        local_z_vec = np.cross(local_x_vec, ref_vec)
        if np.linalg.norm(local_z_vec) < 1e-6:
             ref_vec = np.array([1, 0, 0]) if not is_column and np.isclose(cx_z, 1) else np.array([0, 0, 1])
             local_z_vec = np.cross(local_x_vec, ref_vec)
        local_z_vec /= np.linalg.norm(local_z_vec)
        local_y_vec = np.cross(local_z_vec, local_x_vec)
        R = np.vstack([local_x_vec, local_y_vec, local_z_vec])
        for i in range(4): T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        return T

class Structure:
    """Represents the entire 3D frame structure and handles the FEA."""
    def __init__(self):
        self.nodes, self.elements, self.dof_map = {}, {}, {}
        self.K_global, self.F_global, self.U_global = None, None, None

    def add_node(self, id, x, y, z):
        if id not in self.nodes: self.nodes[id] = Node(id, x, y, z)
        return self.nodes[id]

    def add_element(self, id, start_node_id, end_node_id, props):
        if id not in self.elements and start_node_id in self.nodes and end_node_id in self.nodes:
            self.elements[id] = Element(id, self.nodes[start_node_id], self.nodes[end_node_id], props)
        return self.elements.get(id)

    def assemble_matrices(self):
        # (Omitted for brevity)
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

    def add_wall_loads(self, wall_load_data):
        """Applies UDL loads on specified element IDs as equivalent nodal forces (Fz)."""
        # NOTE: This only applies EQUIVALENT NODAL FORCES. The moment calculation in 
        # plot_2d_frame must account for the UDL between the nodes.
        for elem_id, load_q in wall_load_data.items():
            if elem_id in self.elements:
                beam = self.elements[elem_id]
                # Assuming load is vertically (Global -Z, index 2)
                load_at_node_shear = load_q * beam.length / 2 
                load_at_node_moment = load_q * beam.length**2 / 12 # Fixed end moments for UDL
                
                # Apply load_at_node_shear in -Z direction (index 2 for Fz)
                self.F_global[self.dof_map[(beam.start_node.id, 2)]] -= load_at_node_shear
                self.F_global[self.dof_map[(beam.end_node.id, 2)]] -= load_at_node_shear
                
                # Apply fixed-end moments (index 4=My, index 5=Mz) in 3D. Assuming Mz for bending moment (index 5)
                # Convention: M_start = -wL^2/12, M_end = +wL^2/12 (Local coordinates)
                # Since the global force vector is used, this is more complex.
                # Simplification: Only applying shear forces for basic FEA, rely on plot_2d_frame for internal moment calculation.
                pass 

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
            elem.results['Max_Abs_Moment'] = max(abs(f_local[4]), abs(f_local[5]), abs(f_local[10]), abs(f_local[11]))
            # Store displacement for 2D deflection plot
            elem.results['Disp_Global_Start'] = u_global_elem[:6]
            elem.results['Disp_Global_End'] = u_global_elem[6:]

    def calculate_reactions(self):
        if self.U_global is None: return
        R = self.K_global @ self.U_global # No F_global subtraction if we only want end forces/moments
        for node in self.nodes.values():
            if any(node.restraints):
                for i in range(6):
                    if node.restraints[i]: node.reactions[i] = R[self.dof_map[(node.id, i)]]

# --- 2. Utility & Calculation Functions (Omitted for brevity) ---

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

def parse_and_apply_wall_loads(load_string):
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
        except ValueError: continue
        elem_ids = [int(segment.strip()) for segment in elem_ids_str.split(',') if segment.strip().isdigit()]
        for elem_id in elem_ids: wall_load_data[elem_id] = load_q
    return wall_load_data

# --- 3. Streamlit Caching & Analysis (Omitted for brevity) ---

@st.cache_data
def generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, load_params, wall_load_data):
    s = Structure()
    x_coords, y_coords, z_coords = [0]+list(np.cumsum(x_dims)), [0]+list(np.cumsum(y_dims)), [0]+list(np.cumsum(z_dims))
    node_id, elem_id, node_map = 1, 1, {}

    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                s.add_node(node_id, x, y, z)
                node_map[(ix, iy, iz)] = node_id
                if np.isclose(z, 0): s.set_support(node_id, restraints=[True]*6)
                node_id += 1

    for iz in range(len(z_coords)-1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)): 
                s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy,iz+1)], col_props)
                elem_id += 1
    
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)-1): 
                s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix+1,iy,iz)], beam_props)
                elem_id += 1
    
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)-1):
            for ix in range(len(x_coords)): 
                s.add_element(elem_id, node_map[(ix,iy+1,iz)], node_map[(ix,iy,iz)], beam_props) # Note: order matters for T
                elem_id += 1
    
    s.assemble_matrices()
    # Gravity load is typically applied on the whole frame based on slab/live load
    # s.add_gravity_loads(load_params['q_total_gravity'], z_coords[1:])
    # Specific uniform loads for the 2D case from the user's sketch
    s.add_wall_loads(wall_load_data) 
    
    success, message = s.solve()
    if success: 
        s.calculate_element_results()
        s.calculate_reactions()
    
    if not success: return {'success': False, 'message': message}

    # Store element length for plotting
    elements_data = [{'id':e.id, 'start_node_id':e.start_node.id, 'end_node_id':e.end_node.id, 'start_node_pos':(e.start_node.x,e.start_node.y,e.start_node.z), 'end_node_pos':(e.end_node.x,e.end_node.y,e.end_node.z), 'length':e.length, 'results':e.results} for e in s.elements.values()]
    
    return {
        'success': True, 'message': message,
        'nodes': [{'id':n.id, 'x':n.x, 'y':n.y, 'z':n.z, 'restraints':n.restraints, 'reactions':n.reactions} for n in s.nodes.values()],
        'elements': elements_data,
        'summary': {'num_nodes':len(s.nodes), 'num_elements':len(s.elements), 'k_shape':s.K_global.shape if s.K_global is not None else (0,0)}
    }


# --- 4. Plotting Functions ---

def get_hover_text(elem):
    """Generates detailed hover text for an element."""
    text = f"**Element {elem['id']}** (L={elem.get('length', 0.0):.2f}m)<br>"
    for key, value in elem['results'].items():
        if not key.startswith('Disp') and not key.startswith('Max'):
            unit = 'kNm' if key.startswith('Moment') or key.startswith('Torsion') else 'kN'
            text += f"{key.replace('_', ' ')}: {value:.2f} {unit}<br>"
    return text

# UPDATED FUNCTION: Plot 2D frame with result diagrams
def plot_2d_frame(nodes, elements, plane_axis, coordinate, display_mode, show_values, wall_load_data):
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
        'Bending Moment': ('Moment_Z', 'kNm', 'Deflection in X', 0.2, 'red'), 
        'Shear Force': ('Shear_Z', 'kN', 'Deflection in X', 0.2, 'green'),  
        'Axial Force': ('Axial', 'kN', 'Deflection in X', 0.2, 'blue'),
        'Deflection': (None, 'm', 'Deflection in Z', 100, 'orange'), 
    }
    
    result_key_base, unit, defl_key, scale_factor_base, diagram_color = result_map.get(display_mode, (None, None, None, 1, 'black'))

    # Calculate max dimension for scaling
    x_coords = [n[x_key] for n in plane_nodes_list]
    z_coords_all = [n[z_key] for n in plane_nodes_list]
    max_dim = max(max(x_coords, default=1) - min(x_coords, default=0), 
                  max(z_coords_all, default=1) - min(z_coords_all, default=0)) or 1
    
    # Calculate a global scale factor based on max result and max frame dimension
    max_abs_result = 0.0
    if display_mode != 'Structure':
        max_abs_result = max(abs(e['results'].get(f'{result_key_base}_Start', 0.0)) for e in elements if e['start_node_id'] in plane_node_ids)
        max_abs_result = max(max_abs_result, max(abs(e['results'].get(f'{result_key_base}_End', 0.0)) for e in elements if e['end_node_id'] in plane_node_ids))
        max_abs_result = max(max_abs_result, 1e-3) # Prevent division by zero
        
    # Scale factor: 1/8th of the max frame dimension, divided by the max result
    # We use 1/8 to ensure the diagram is visible but doesn't completely overwhelm the frame.
    global_scale = (max_dim / 8.0) / max_abs_result if display_mode != 'Structure' and display_mode != 'Deflection' else 1.0
    
    # Deflection scale is fixed and high for visualization
    if display_mode == 'Deflection':
        global_scale = 1000 * max_dim # Highly exaggerated scale

    # 2. Iterate and Plot Elements and Diagrams
    for elem in elements:
        if elem['start_node_id'] in plane_node_ids and elem['end_node_id'] in plane_node_ids:
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            
            start_x_plot = start_pos[0 if plane_axis=='Y' else 1]
            end_x_plot = end_pos[0 if plane_axis=='Y' else 1]
            start_z, end_z = start_pos[2], end_pos[2]
            
            x_coords_frame = [start_x_plot, end_x_plot]
            z_coords_frame = [start_z, end_z]
            L = elem['length']
            
            # --- BASE ELEMENT LINE ---
            fig.add_trace(go.Scatter(x=x_coords_frame, y=z_coords_frame, mode='lines', line=dict(color='darkblue', width=3), 
                                     hoverinfo='text', hovertext=get_hover_text(elem), showlegend=False))
            
            # --- RESULT DIAGRAMS ---
            if display_mode != 'Structure':
                
                # A. Prepare points for the curve
                num_points = 51 if display_mode in ['Bending Moment', 'Shear Force'] else 2
                local_x = np.linspace(0, L, num_points)
                diagram_coords = []
                
                w_eff = wall_load_data.get(elem['id'], 0.0)
                
                val_start = elem['results'].get(f'{result_key_base}_Start', 0.0)
                val_end = elem['results'].get(f'{result_key_base}_End', 0.0)

                # Determine the member's orientation
                is_horizontal = np.isclose(start_z, end_z)
                
                # ----------------------------------------------------
                # Core Logic: Calculating Internal Force/Moment/Deflection
                # ----------------------------------------------------
                
                if display_mode in ['Bending Moment', 'Shear Force']:
                    # Assuming local Mz (index 5, 11) is the primary bending moment for beams in this X-Z plane
                    M_start = elem['results'].get('Moment_Z_Start', 0.0)
                    V_start = elem['results'].get('Shear_Z_Start', 0.0)
                    
                    for x in local_x:
                        if L == 0: value = 0.0
                        elif result_key_base == 'Moment_Z':
                            # Internal Moment M(x) = M_start + V_start*x - w*x^2/2 (Standard beam convention)
                            # NOTE: FEA forces are the **reversal** of the internal action.
                            # We use the calculated V_start and M_start from FEA.
                            # If load 'w' is applied, the moment equation must include it.
                            # M_int(x) = M_end_start + V_end_start*x + w*x* (L-x)/2 (UDL on simply supported, not what we have)
                            # M_int(x) = M_start + V_start*x + w*(x^2)/2 (If V_start is the shear at x=0)
                            # The most robust way is to use the FEA end forces and the UDL:
                            # M(x) = M_start + V_start * x - w_eff * x**2 / 2
                            # *Sign correction for plotting:* FEA gives local end moments. For tension-side plotting (standard), we need to negate.
                            value = M_start + V_start * x - w_eff * x**2 / 2
                        elif result_key_base == 'Shear_Z':
                            # Internal Shear V(x) = V_start - w*x
                            value = V_start - w_eff * x
                        else: value = 0.0
                        
                        diagram_coords.append((x, value))

                elif display_mode == 'Axial Force':
                    # Axial force is constant on a non-loaded member (linear interpolation works)
                    for x in local_x:
                        value = val_start + (val_end - val_start) * (x / L) if L > 0 else 0
                        diagram_coords.append((x, value))
                        
                elif display_mode == 'Deflection':
                    u_start = elem['results'].get('Disp_Global_Start', np.zeros(6))
                    u_end = elem['results'].get('Disp_Global_End', np.zeros(6))
                    
                    for x in local_x:
                        # Linear interpolation of global displacement for visualization (cubic is complex)
                        if L > 0:
                            if defl_key == 'Deflection in Z':
                                disp_start, disp_end = u_start[2], u_end[2] # Global Uz
                            elif defl_key == 'Deflection in X':
                                disp_start, disp_end = u_start[0], u_end[0] # Global Ux
                            else: continue
                            value = disp_start + (disp_end - disp_start) * (x / L)
                        else: value = 0.0
                        diagram_coords.append((x, value))
                        
                # ----------------------------------------------------
                
                # B. Convert Local Diagram Coords to Global Plotting Coords
                diag_x, diag_z = [], []
                value_at_point = []
                
                for x_local, value in diagram_coords:
                    value_at_point.append(value)
                    
                    # Calculate position along the member
                    # dx_mem = x_coords_frame[1] - x_coords_frame[0]
                    # dz_mem = z_coords_frame[1] - z_coords_frame[0]
                    
                    # Normalized vector along the member
                    if L > 0:
                        nx = (x_coords_frame[1] - x_coords_frame[0]) / L
                        nz = (z_coords_frame[1] - z_coords_frame[0]) / L
                    else: nx, nz = 1, 0
                    
                    # Normalized vector perpendicular to the member (for offset)
                    # Perpendicular vector (n_perp) = [-nz, nx] (rotated 90 deg clockwise)
                    nx_perp = -nz 
                    nz_perp = nx 
                    
                    # Current position on the member
                    x_mid = x_coords_frame[0] + nx * x_local
                    z_mid = z_coords_frame[0] + nz * x_local
                    
                    # Apply offset (negate value for standard moment/shear plotting)
                    plot_value = value * global_scale * (-1 if display_mode in ['Bending Moment', 'Shear Force'] else 1)

                    x_plot = x_mid + nx_perp * plot_value
                    z_plot = z_mid + nz_perp * plot_value
                    
                    diag_x.append(x_plot)
                    diag_z.append(z_plot)

                # Add diagram trace 
                fig.add_trace(go.Scatter(x=diag_x, y=diag_z, mode='lines', line=dict(color=diagram_color, width=2), 
                                         fill='tozeroy' if is_horizontal else 'tozerox', opacity=0.3 if display_mode != 'Deflection' else 0,
                                         hoverinfo='none', name=f"{display_mode} E{elem['id']}"))
                
                # C. Plot Values if toggled
                if show_values and display_mode != 'Deflection':
                    # Find max/min values for labeling
                    val_abs_max = max(abs(v) for v in value_at_point)
                    
                    # Label positions (Start, End, and Mid-span only)
                    label_indices = [0, len(local_x) - 1] # Start and End
                    if display_mode == 'Bending Moment': label_indices.append(np.argmax(np.abs(value_at_point))) # Max moment position
                    
                    for idx in sorted(list(set(label_indices))):
                        val = value_at_point[idx]
                        if abs(val) < 1e-3: continue # Skip near-zero values
                        
                        fig.add_annotation(
                            x=diag_x[idx], y=diag_z[idx],
                            text=f"{val:.2f}",
                            showarrow=False,
                            xshift=nx_perp * (val * global_scale * 10), # Slight offset perpendicular to the member
                            yshift=nz_perp * (val * global_scale * 10),
                            font=dict(color=diagram_color, size=10),
                        )
                        

    # 3. Node Markers and Layout
    fig.add_trace(go.Scatter(x=[n[x_key] for n in plane_nodes_list], y=[n[z_key] for n in plane_nodes_list], 
                             mode='markers', marker=dict(size=8, color='purple'), name='Nodes', 
                             text=[f"Node {n['id']}" for n in plane_nodes_list], hoverinfo='text', showlegend=False))

    title_scale = f" (Scale: 1:{1/global_scale:.0f})" if display_mode in ['Bending Moment', 'Shear Force', 'Axial Force'] else ""
    title_scale_def = f" (Exaggerated Scale)" if display_mode == 'Deflection' else ""
    
    fig.update_layout(title=f"2D Elevation: {display_mode} ({unit}) on {plane_axis.replace('Y', 'X-Z').replace('X', 'Y-Z')} Plane at {plane_axis}={coordinate}m{title_scale}{title_scale_def}", 
                      xaxis_title=f'{x_key.upper()}-axis (m)', yaxis_title='Z-axis (m)', showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --- 5. Main Streamlit App UI ---
# (The Streamlit UI part remains largely the same, but inputs are set for the 2D sketch and plot_2d_frame call is updated)

st.title("🏗️ 2D/3D Frame Analyzer")
st.write("Define your building grid, sections, and loads to generate and analyze a 3D frame.")

with st.sidebar:
    st.header("1. Frame Geometry (For 2D Sketch)")
    x_grid = st.text_input("X-spans (m)", "7.0") 
    y_grid = st.text_input("Y-spans (m)", "0.001") # Set a minimal Y-span to force X-Z plane
    z_grid = st.text_input("Z-heights (m)", "3.5, 3.5")
    
    st.header("2. Section & Material")
    E = st.number_input("E (GPa)", 200.0, help="200 GPa for Steel")*1e6 
    with st.expander("Column & Beam Sizes"):
        col_b, col_h = st.number_input("Col b (mm)", 240)/1000, st.number_input("Col h (mm)", 240)/1000 # Use IPE240 dim
        beam_b, beam_h = st.number_input("Beam b (mm)", 120)/1000, st.number_input("Beam h (mm)", 240)/1000 # Use IPE240 dim
        
    st.header("3. Gravity Loads")
    with st.expander("Slab & Live Loads (Set to 0 for 2D Sketch Load Case)"):
        slab_d, slab_t = st.number_input("Slab Density (kN/m³)", 0.0), st.number_input("Slab Thickness (m)", 0.0)
        fin_l, live_l = st.number_input("Finishes (kN/m²)", 0.0), st.number_input("Live Load (kN/m²)", 0.0)
        
    st.subheader("4. Specific Uniform Loads (UDL)")
    wall_load_input = st.text_area("Element Loads (e.g., 2, 3: 5 kN/m)", 
                                   value="3, 4: 10.0", # Loads from sketch
                                   height=100,
                                   help="Enter Element IDs and the load magnitude (in kN/m applied in -Z direction).")
    
    analyze_button = st.button("Generate & Analyze Frame", type="primary")

if analyze_button:
    # (Analysis logic - Omitted for brevity)
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
            st.session_state['wall_load_data'] = wall_load_data 

            st.subheader("FEA Results Summary")
            # (Summary metrics - Omitted for brevity)


if 'analysis_results' in st.session_state and st.session_state['analysis_results']['success']:
    results, nodes, elements = st.session_state['analysis_results'], st.session_state['analysis_results']['nodes'], st.session_state['analysis_results']['elements']
    
    # 3D Visualization Section (Omitted for brevity)
    
    tab1, tab2, tab3 = st.tabs(["2D Elevation View", "Support Reactions", "Detailed Element Results"])
    
    with tab1:
        st.subheader("2D Elevation View and Result Diagrams")
        
        col_2d_1, col_2d_2, col_2d_3 = st.columns([0.3, 0.4, 0.3])
        
        plane_axis_display = col_2d_1.radio("Grid Plane", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'), key='plane_axis_radio_1')
        diagram_options = ['Structure', 'Bending Moment', 'Shear Force', 'Axial Force', 'Deflection']
        display_mode_2d = col_2d_2.selectbox("Result Diagram", diagram_options, key='display_mode_2d_1')
        show_values = col_2d_3.checkbox("Show Values", key='show_values_1', value=True)
        
        # Force the selection to the plane of the 2D frame (y=0.001)
        if plane_axis_display == 'X-Z (Y-Gridline)':
            y_coords = sorted(list(set(n['y'] for n in nodes)))
            coordinate = y_coords[0] if y_coords else 0
            plane_key = 'Y'
        else:
            x_coords = sorted(list(set(n['x'] for n in nodes)))
            coordinate = x_coords[0] if x_coords else 0
            plane_key = 'X'
        
        st.info(f"Showing results on the **{plane_key}-Gridline** at coordinate **{coordinate:.3f} m**.")
        
        st.plotly_chart(plot_2d_frame(nodes, elements, plane_key, coordinate, display_mode_2d, show_values, st.session_state['wall_load_data']), use_container_width=True)

    with tab2:
        # (Support Reactions - Omitted for brevity)
        pass
        
    with tab3:
        # (Detailed Results - Omitted for brevity)
        pass
