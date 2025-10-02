import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from collections import defaultdict

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Improved 3D Frame Analyzer")

# --- 1. Object-Oriented Model for the Structure ---
# Using classes makes the code cleaner, more organized, and easier to extend.

class Node:
    """Represents a single node in the 3D structure."""
    def __init__(self, id, x, y, z):
        self.id = int(id)
        self.x, self.y, self.z = float(x), float(y), float(z)
        # 6 degrees of freedom (DOF): [dx, dy, dz, rx, ry, rz]
        # True means the DOF is restrained (fixed).
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
        self.props = props # Dictionary with E, G, A, Iyy, Izz, J
        self.length = self.calculate_length()
        self.results = {} # To store analysis results like forces and moments

    def calculate_length(self):
        """Calculates the element's length based on its node coordinates."""
        return np.sqrt(
            (self.end_node.x - self.start_node.x)**2 +
            (self.end_node.y - self.start_node.y)**2 +
            (self.end_node.z - self.start_node.z)**2
        )

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
        k[1,5] = k[5,1] = k[1,11] = k[11,1] = 6*E*Izz/L2
        k[5,1] = k[1,11] = -k[1,5] # Sign correction based on convention

        k[2,2] = k[8,8] = 12*E*Iyy/L3
        k[2,8] = k[8,2] = -12*E*Iyy/L3
        k[2,4] = k[4,2] = k[2,10] = k[10,2] = 6*E*Iyy/L2
        k[4,2] = k[10,2] = -k[2,4] # Sign correction

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
        return k

    def get_transformation_matrix(self):
        """Calculates the 12x12 transformation matrix T."""
        T = np.zeros((12, 12))
        dx = self.end_node.x - self.start_node.x
        dy = self.end_node.y - self.start_node.y
        dz = self.end_node.z - self.start_node.z

        # Handle zero-length element case
        if self.length == 0: return np.identity(12)
        
        cx_x = dx / self.length
        cx_y = dy / self.length
        cx_z = dz / self.length

        # Robust approach for y' and z' axes using a reference vector (global Z-axis)
        # This avoids issues with vertical members.
        if abs(cx_x) < 1e-6 and abs(cx_y) < 1e-6: # Vertical member
            # Local x' is aligned with global Z. Let's align local y' with global Y.
            ref_vec = np.array([0, 1, 0])
        else:
            # For non-vertical members, use global Z as the reference vector.
            ref_vec = np.array([0, 0, 1])
        
        local_x_vec = np.array([cx_x, cx_y, cx_z])
        
        # Local z' = local x' X ref_vec
        local_z_vec = np.cross(local_x_vec, ref_vec)
        local_z_vec /= np.linalg.norm(local_z_vec)
        
        # Local y' = local z' X local x'
        local_y_vec = np.cross(local_z_vec, local_x_vec)
        
        # Rotation matrix
        R = np.vstack([local_x_vec, local_y_vec, local_z_vec])
        
        for i in range(4):
            T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        return T


class Structure:
    """Represents the entire 3D frame structure and handles the FEA."""
    def __init__(self):
        self.nodes = {}
        self.elements = {}
        self.dof_map = {}
        self.K_global = None
        self.F_global = None
        self.U_global = None

    def add_node(self, id, x, y, z):
        if id not in self.nodes:
            self.nodes[id] = Node(id, x, y, z)
        return self.nodes[id]

    def add_element(self, id, start_node_id, end_node_id, props):
        if id not in self.elements and start_node_id in self.nodes and end_node_id in self.nodes:
            start_node = self.nodes[start_node_id]
            end_node = self.nodes[end_node_id]
            self.elements[id] = Element(id, start_node, end_node, props)
        return self.elements.get(id)

    def set_support(self, node_id, restraints):
        if node_id in self.nodes:
            self.nodes[node_id].restraints = restraints

    def assemble_matrices(self):
        """Assembles the global stiffness and force matrices."""
        num_dof = len(self.nodes) * 6
        self.K_global = np.zeros((num_dof, num_dof))
        self.F_global = np.zeros(num_dof)
        
        dof_index = 0
        for node_id in sorted(self.nodes.keys()):
            for i in range(6):
                self.dof_map[(node_id, i)] = dof_index
                dof_index += 1

        for elem in self.elements.values():
            k_local = elem.get_local_stiffness_matrix()
            T = elem.get_transformation_matrix()
            k_global_elem = T @ k_local @ T.T
            
            node_ids = [elem.start_node.id, elem.end_node.id]
            dof_indices = [self.dof_map[(nid, i)] for nid in node_ids for i in range(6)]
            
            for i, global_i in enumerate(dof_indices):
                for j, global_j in enumerate(dof_indices):
                    self.K_global[global_i, global_j] += k_global_elem[i, j]

    def add_gravity_loads(self, q_gravity, levels):
        """Distributes slab gravity loads to adjacent beams."""
        for z in levels:
            level_nodes = {n.id: (n.x, n.y) for n in self.nodes.values() if np.isclose(n.z, z)}
            if not level_nodes: continue
            
            level_beams = [e for e in self.elements.values() if e.start_node.id in level_nodes and e.end_node.id in level_nodes]
            
            # This is a simplification using fixed-end forces for a uniformly distributed load (wL/2)
            for beam in level_beams:
                w = q_gravity # Assuming tributary width is 1m for simplicity
                L = beam.length
                load_at_node = w * L / 2
                
                start_node_dof = self.dof_map[(beam.start_node.id, 2)] # Fz
                end_node_dof = self.dof_map[(beam.end_node.id, 2)] # Fz
                
                self.F_global[start_node_dof] -= load_at_node
                self.F_global[end_node_dof] -= load_at_node

    def solve(self):
        """Solves the system [K]{U} = {F} after applying boundary conditions."""
        active_dofs = []
        for node in self.nodes.values():
            for i in range(6):
                if not node.restraints[i]:
                    active_dofs.append(self.dof_map[(node.id, i)])
        
        active_dofs = np.array(active_dofs)
        K_reduced = self.K_global[active_dofs[:, np.newaxis], active_dofs]
        F_reduced = self.F_global[active_dofs]

        try:
            U_reduced = np.linalg.solve(K_reduced, F_reduced)
            self.U_global = np.zeros_like(self.F_global)
            self.U_global[active_dofs] = U_reduced
            return True, "Analysis successful."
        except np.linalg.LinAlgError:
            self.U_global = None
            return False, "Analysis failed. The structure may be unstable (singular matrix)."

    def calculate_element_results(self):
        """Calculates internal forces and moments for each element."""
        if self.U_global is None: return

        for elem in self.elements.values():
            node_ids = [elem.start_node.id, elem.end_node.id]
            dof_indices = [self.dof_map[(nid, i)] for nid in node_ids for i in range(6)]
            
            u_global_elem = self.U_global[dof_indices]
            T = elem.get_transformation_matrix()
            u_local_elem = T.T @ u_global_elem
            
            k_local = elem.get_local_stiffness_matrix()
            f_local = k_local @ u_local_elem
            
            elem.results = {
                'Axial_Start': f_local[0], 'Axial_End': f_local[6],
                'Shear_Y_Start': f_local[1], 'Shear_Y_End': f_local[7],
                'Shear_Z_Start': f_local[2], 'Shear_Z_End': f_local[8],
                'Torsion_Start': f_local[3], 'Torsion_End': f_local[9],
                'Moment_Y_Start': f_local[4], 'Moment_Y_End': f_local[10],
                'Moment_Z_Start': f_local[5], 'Moment_Z_End': f_local[11],
            }
            elem.results['Max_Abs_Moment'] = max(abs(f_local[5]), abs(f_local[11]))

    def calculate_reactions(self):
        """Calculates reaction forces at support nodes."""
        if self.U_global is None or self.K_global is None: return
        R = self.K_global @ self.U_global - self.F_global
        for node in self.nodes.values():
            if any(node.restraints):
                for i in range(6):
                    if node.restraints[i]:
                        dof_index = self.dof_map[(node.id, i)]
                        node.reactions[i] = R[dof_index]


# --- 2. Utility & Calculation Functions ---

def parse_grid_input(input_string):
    """Parses flexible grid input strings (e.g., "3x5, 2x4.5, 5") into an array of lengths."""
    if not input_string: return []
    segments = [s.strip() for s in input_string.split(',') if s.strip()]
    lengths = []
    for segment in segments:
        match = re.match(r'^(\d+)x([0-9.]+)$', segment)
        if match:
            count, length = int(match.group(1)), float(match.group(2))
            if count > 0 and length > 0: lengths.extend([length] * count)
        else:
            try:
                single_length = float(segment)
                if single_length > 0: lengths.append(single_length)
            except ValueError: pass
    return lengths

def calculate_rc_properties(b, h, E, nu=0.2):
    """Calculates 3D structural properties for a rectangular section."""
    A = b * h
    Izz = (b * h**3) / 12
    Iyy = (h * b**3) / 12
    G = E / (2 * (1 + nu))
    a, c = max(b, h), min(b, h)
    J = a * (c**3) * (1/3 - 0.21 * (c/a) * (1 - (c**4)/(12*a**4)))
    return {'E': E, 'G': G, 'A': A, 'Iyy': Iyy, 'Izz': Izz, 'J': J}


# --- 3. Streamlit Caching ---

@st.cache_data
def generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, load_params):
    s = Structure()
    x_coords, y_coords, z_coords = [0] + list(np.cumsum(x_dims)), [0] + list(np.cumsum(y_dims)), [0] + list(np.cumsum(z_dims))
    levels = z_coords[1:]
    node_id, elem_id = 1, 1
    node_map = {}

    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                s.add_node(node_id, x, y, z)
                node_map[(ix, iy, iz)] = node_id
                if np.isclose(z, 0): s.set_support(node_id, restraints=[True]*6)
                node_id += 1

    for iz in range(len(z_coords) - 1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)):
                s.add_element(elem_id, node_map[(ix, iy, iz)], node_map[(ix, iy, iz + 1)], col_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords) - 1):
                s.add_element(elem_id, node_map[(ix, iy, iz)], node_map[(ix + 1, iy, iz)], beam_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords) - 1):
            for ix in range(len(x_coords)):
                s.add_element(elem_id, node_map[(ix, iy, iz)], node_map[(ix, iy + 1, iz)], beam_props); elem_id += 1
    
    s.assemble_matrices()
    s.add_gravity_loads(load_params['q_total_gravity'], levels)
    success, message = s.solve()
    if success:
        s.calculate_element_results()
        s.calculate_reactions()
        
    return s, success, message


# --- 4. Plotting Functions ---

def plot_3d_frame(structure, display_mode='Structure'):
    fig = go.Figure()
    
    if display_mode == 'Bending Moment (Myz)':
        max_moment = max((abs(e.results.get('Max_Abs_Moment', 0)) for e in structure.elements.values()), default=0)
        for elem in structure.elements.values():
            moment = elem.results.get('Max_Abs_Moment', 0)
            color_val = moment / max_moment if max_moment > 0 else 0
            color = f'rgb({int(255 * color_val)}, 0, {int(255 * (1-color_val))})'
            fig.add_trace(go.Scatter3d(x=[elem.start_node.x, elem.end_node.x], y=[elem.start_node.y, elem.end_node.y], z=[elem.start_node.z, elem.end_node.z], mode='lines', line=dict(color=color, width=5), hoverinfo='text', hovertext=f"Elem {elem.id}<br>Moment: {moment:.2f} kNm", name=f"Elem {elem.id}"))
    else:
        edge_x, edge_y, edge_z = [], [], []
        for elem in structure.elements.values():
            edge_x.extend([elem.start_node.x, elem.end_node.x, None])
            edge_y.extend([elem.start_node.y, elem.end_node.y, None])
            edge_z.extend([elem.start_node.z, elem.end_node.z, None])
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='darkblue', width=4), name='Elements'))

    node_x, node_y, node_z = [n.x for n in structure.nodes.values()], [n.y for n in structure.nodes.values()], [n.z for n in structure.nodes.values()]
    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(size=5, color='purple', symbol='circle'), name='Nodes', text=[f"Node {n.id}" for n in structure.nodes.values()], hoverinfo='text'))

    fig.update_layout(title=f"3D Frame Visualization - {display_mode}", scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectratio=dict(x=1.5, y=1.5, z=1)), margin=dict(l=0, r=0, b=0, t=40), showlegend=False)
    return fig

def plot_2d_frame(structure, plane_axis, coordinate):
    fig = go.Figure()
    plane_nodes = {}
    if plane_axis == 'Y':
        plane_nodes = {nid: n for nid, n in structure.nodes.items() if np.isclose(n.y, coordinate)}
        x_axis, z_axis = 'x', 'z'
    else: # plane_axis == 'X'
        plane_nodes = {nid: n for nid, n in structure.nodes.items() if np.isclose(n.x, coordinate)}
        x_axis, z_axis = 'y', 'z'

    for elem in structure.elements.values():
        if elem.start_node.id in plane_nodes and elem.end_node.id in plane_nodes:
            fig.add_trace(go.Scatter(x=[getattr(elem.start_node, x_axis), getattr(elem.end_node, x_axis)], y=[getattr(elem.start_node, z_axis), getattr(elem.end_node, z_axis)], mode='lines', line=dict(color='darkblue', width=3), hoverinfo='none'))

    node_x = [getattr(n, x_axis) for n in plane_nodes.values()]
    node_y = [getattr(n, z_axis) for n in plane_nodes.values()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=8, color='purple'), name='Nodes', text=[f"Node {n.id}" for n in plane_nodes.values()], hoverinfo='text'))

    fig.update_layout(title=f"2D Elevation on {plane_axis.replace('Y', 'X').replace('X', 'Y')}-Z Plane at {plane_axis}={coordinate}m", xaxis_title=f'{x_axis.upper()}-axis (m)', yaxis_title='Z-axis (m)', showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --- 5. Main Streamlit App UI ---
st.title("🏗️ Improved 3D Frame Analyzer")
st.write("Define your building grid, sections, and loads to generate and analyze a 3D frame.")

with st.sidebar:
    st.header("1. Frame Geometry")
    x_grid_input = st.text_input("X-direction Spans (m)", "3x6, 5.5")
    y_grid_input = st.text_input("Y-direction Spans (m)", "2x5, 4")
    z_grid_input = st.text_input("Floor Heights (m)", "4, 2x3.5")
    st.header("2. Section & Material Properties")
    with st.expander("Expand for Details"):
        E = st.number_input("Young's Modulus (E, in GPa)", value=30.0) * 1e6
        col_b, col_h = st.number_input("Column Width (b, mm)", 400)/1000, st.number_input("Column Depth (h, mm)", 400)/1000
        beam_b, beam_h = st.number_input("Beam Width (b, mm)", 300)/1000, st.number_input("Beam Depth (h, mm)", 500)/1000
    st.header("3. Gravity Loads")
    with st.expander("Expand for Details"):
        slab_density, slab_thickness = st.number_input("Slab Concrete Density (kN/m³)", 25.0), st.number_input("Slab Thickness (m)", 0.150)
        finish_load, live_load = st.number_input("Finishes Load (kN/m²)", 1.5), st.number_input("Live Load (kN/m²)", 3.0)
    analyze_button = st.button("Generate & Analyze Frame", type="primary")

if analyze_button:
    x_dims, y_dims, z_dims = parse_grid_input(x_grid_input), parse_grid_input(y_grid_input), parse_grid_input(z_grid_input)
    if not all([x_dims, y_dims, z_dims]):
        st.error("Invalid grid input. Please check your span and height definitions.")
    else:
        col_props, beam_props = calculate_rc_properties(col_b, col_h, E), calculate_rc_properties(beam_b, beam_h, E)
        q_total_gravity = slab_density * slab_thickness + finish_load + live_load
        load_params = {'q_total_gravity': q_total_gravity}
        with st.spinner("Running Finite Element Analysis..."):
            structure, success, message = generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, load_params)
        if not success: st.error(f"Analysis Failed: {message}")
        else:
            st.success("Analysis complete!")
            st.session_state['structure'] = structure
            st.subheader("FEA Results Summary")
            max_abs_moment = max((abs(elem.results.get('Max_Abs_Moment', 0)) for elem in structure.elements.values()), default=0)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Nodes", len(structure.nodes)); col1.metric("Total Elements", len(structure.elements))
            col2.metric("Floor Pressure", f"{q_total_gravity:.2f} kN/m²"); col2.metric("Global K Size", f"{structure.K_global.shape}")
            col3.metric("Max Bending Moment", f"{max_abs_moment:.2f} kNm")

if 'structure' in st.session_state:
    st.subheader("Interactive 3D Visualization")
    display_mode = st.selectbox("Display Mode", ['Structure', 'Bending Moment (Myz)'])
    frame_fig = plot_3d_frame(st.session_state['structure'], display_mode)
    st.plotly_chart(frame_fig, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["2D Elevation View", "Support Reactions", "Detailed Element Results"])
    with tab1:
        plane_axis_choice = st.radio("Select Grid Plane", ('X-Z Plane (Gridline along Y)', 'Y-Z Plane (Gridline along X)'))
        if plane_axis_choice == 'X-Z Plane (Gridline along Y)':
            y_coords = sorted(list(set(n.y for n in st.session_state['structure'].nodes.values())))
            selected_y = st.selectbox("Select Y-coordinate for the grid", options=y_coords, key='y_coord')
            st.plotly_chart(plot_2d_frame(st.session_state['structure'], 'Y', selected_y), use_container_width=True)
        else:
            x_coords = sorted(list(set(n.x for n in st.session_state['structure'].nodes.values())))
            selected_x = st.selectbox("Select X-coordinate for the grid", options=x_coords, key='x_coord')
            st.plotly_chart(plot_2d_frame(st.session_state['structure'], 'X', selected_x), use_container_width=True)
    with tab2:
        support_nodes = [n.id for n in st.session_state['structure'].nodes.values() if any(n.restraints)]
        if support_nodes:
            selected_node_id = st.selectbox("Select a support node", options=support_nodes)
            node = st.session_state['structure'].nodes[selected_node_id]
            df_reactions = pd.DataFrame({
                "Force/Moment": ["Fx (kN)", "Fy (kN)", "Fz (kN)", "Mx (kNm)", "My (kNm)", "Mz (kNm)"],
                "Value": node.reactions
            }).round(2)
            st.dataframe(df_reactions)
        else: st.write("No support nodes found.")
    with tab3:
        results_data = [{'ID': e.id, 'Start': e.start_node.id, 'End': e.end_node.id, 'Max Moment (kNm)': e.results.get('Max_Abs_Moment',0), 'Axial (kN)': e.results.get('Axial_Start',0)} for e in st.session_state['structure'].elements.values()]
        st.dataframe(pd.DataFrame(results_data).round(2), use_container_width=True)

