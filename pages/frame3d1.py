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

        if self.length == 0: return np.identity(12)
        
        cx_x, cx_y, cx_z = dx / self.length, dy / self.length, dz / self.length

        if abs(cx_x) < 1e-6 and abs(cx_y) < 1e-6:
            ref_vec = np.array([0, 1, 0])
        else:
            ref_vec = np.array([0, 0, 1])
        
        local_x_vec = np.array([cx_x, cx_y, cx_z])
        local_z_vec = np.cross(local_x_vec, ref_vec)
        local_z_vec /= np.linalg.norm(local_z_vec)
        local_y_vec = np.cross(local_z_vec, local_x_vec)
        
        R = np.vstack([local_x_vec, local_y_vec, local_z_vec])
        
        for i in range(4):
            T[i*3:(i+1)*3, i*3:(i+1)*3] = R
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
        for z in levels:
            level_nodes = {n.id for n in self.nodes.values() if np.isclose(n.z, z)}
            if not level_nodes: continue
            
            level_beams = [e for e in self.elements.values() if e.start_node.id in level_nodes and e.end_node.id in level_nodes]
            for beam in level_beams:
                load_at_node = q_gravity * beam.length / 2
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
        for elem in self.elements.values():
            dof_indices = [self.dof_map[(nid, i)] for nid in [elem.start_node.id, elem.end_node.id] for i in range(6)]
            u_global_elem = self.U_global[dof_indices]
            u_local_elem = elem.get_transformation_matrix().T @ u_global_elem
            f_local = elem.get_local_stiffness_matrix() @ u_local_elem
            elem.results = {'Axial_Start':f_local[0],'Axial_End':f_local[6],'Shear_Y_Start':f_local[1],'Shear_Y_End':f_local[7],'Shear_Z_Start':f_local[2],'Shear_Z_End':f_local[8],'Torsion_Start':f_local[3],'Torsion_End':f_local[9],'Moment_Y_Start':f_local[4],'Moment_Y_End':f_local[10],'Moment_Z_Start':f_local[5],'Moment_Z_End':f_local[11]}
            elem.results['Max_Abs_Moment'] = max(abs(f_local[5]), abs(f_local[11]))

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

# --- 3. Streamlit Caching ---

@st.cache_data
def generate_and_analyze_structure(x_dims, y_dims, z_dims, col_props, beam_props, load_params):
    s = Structure()
    x_coords, y_coords, z_coords = [0]+list(np.cumsum(x_dims)), [0]+list(np.cumsum(y_dims)), [0]+list(np.cumsum(z_dims))
    node_id, elem_id, node_map = 1, 1, {}

    for iz, z in enumerate(z_coords):
        for iy, y in enumerate(y_coords):
            for ix, x in enumerate(x_coords):
                s.add_node(node_id, x, y, z); node_map[(ix, iy, iz)] = node_id
                if np.isclose(z, 0): s.set_support(node_id, restraints=[True]*6)
                node_id += 1

    for iz in range(len(z_coords)-1):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)): s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy,iz+1)], col_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)):
            for ix in range(len(x_coords)-1): s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix+1,iy,iz)], beam_props); elem_id += 1
    for iz in range(1, len(z_coords)):
        for iy in range(len(y_coords)-1):
            for ix in range(len(x_coords)): s.add_element(elem_id, node_map[(ix,iy,iz)], node_map[(ix,iy+1,iz)], beam_props); elem_id += 1
    
    s.assemble_matrices()
    s.add_gravity_loads(load_params['q_total_gravity'], z_coords[1:])
    success, message = s.solve()
    if success: s.calculate_element_results(); s.calculate_reactions()
    
    if not success: return {'success': False, 'message': message}

    return {
        'success': True, 'message': message,
        'nodes': [{'id':n.id, 'x':n.x, 'y':n.y, 'z':n.z, 'restraints':n.restraints, 'reactions':n.reactions} for n in s.nodes.values()],
        'elements': [{'id':e.id, 'start_node_id':e.start_node.id, 'end_node_id':e.end_node.id, 'start_node_pos':(e.start_node.x,e.start_node.y,e.start_node.z), 'end_node_pos':(e.end_node.x,e.end_node.y,e.end_node.z), 'results':e.results} for e in s.elements.values()],
        'summary': {'num_nodes':len(s.nodes), 'num_elements':len(s.elements), 'k_shape':s.K_global.shape if s.K_global is not None else (0,0)}
    }

# --- 4. Plotting Functions ---

def plot_3d_frame(nodes, elements, display_mode='Structure'):
    fig = go.Figure()
    if display_mode == 'Bending Moment (Myz)':
        max_moment = max((abs(e['results'].get('Max_Abs_Moment', 0)) for e in elements), default=0)
        for elem in elements:
            moment = elem['results'].get('Max_Abs_Moment', 0)
            color_val = moment/max_moment if max_moment > 0 else 0
            color = f'rgb({int(255*color_val)}, 0, {int(255*(1-color_val))})'
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            fig.add_trace(go.Scatter3d(x=[start_pos[0],end_pos[0]], y=[start_pos[1],end_pos[1]], z=[start_pos[2],end_pos[2]], mode='lines', line=dict(color=color, width=5), hoverinfo='text', hovertext=f"Elem {elem['id']}<br>Moment: {moment:.2f} kNm", name=f"Elem {elem['id']}"))
    else:
        edge_x, edge_y, edge_z = [], [], []
        for elem in elements:
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            edge_x.extend([start_pos[0],end_pos[0],None]); edge_y.extend([start_pos[1],end_pos[1],None]); edge_z.extend([start_pos[2],end_pos[2],None])
        fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='darkblue', width=4), name='Elements'))
    
    node_x, node_y, node_z = [n['x'] for n in nodes], [n['y'] for n in nodes], [n['z'] for n in nodes]
    fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers', marker=dict(size=5, color='purple'), name='Nodes', text=[f"Node {n['id']}" for n in nodes], hoverinfo='text'))
    fig.update_layout(title=f"3D Frame Visualization - {display_mode}", scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectratio=dict(x=1.5, y=1.5, z=1)), margin=dict(l=0,r=0,b=0,t=40), showlegend=False)
    return fig

def plot_2d_frame(nodes, elements, plane_axis, coordinate):
    fig = go.Figure()
    if plane_axis == 'Y': plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['y'], coordinate)], 'x', 'z'
    else: plane_nodes_list, x_key, z_key = [n for n in nodes if np.isclose(n['x'], coordinate)], 'y', 'z'
    plane_node_ids = {n['id'] for n in plane_nodes_list}

    for elem in elements:
        if elem['start_node_id'] in plane_node_ids and elem['end_node_id'] in plane_node_ids:
            start_pos, end_pos = elem['start_node_pos'], elem['end_node_pos']
            x_coords, z_coords = [start_pos[0 if plane_axis=='Y' else 1], end_pos[0 if plane_axis=='Y' else 1]], [start_pos[2], end_pos[2]]
            fig.add_trace(go.Scatter(x=x_coords, y=z_coords, mode='lines', line=dict(color='darkblue', width=3), hoverinfo='none'))

    fig.add_trace(go.Scatter(x=[n[x_key] for n in plane_nodes_list], y=[n[z_key] for n in plane_nodes_list], mode='markers', marker=dict(size=8, color='purple'), name='Nodes', text=[f"Node {n['id']}" for n in plane_nodes_list], hoverinfo='text'))
    fig.update_layout(title=f"2D Elevation on {plane_axis.replace('Y', 'X-Z').replace('X', 'Y-Z')} Plane at {plane_axis}={coordinate}m", xaxis_title=f'{x_key.upper()}-axis (m)', yaxis_title='Z-axis (m)', showlegend=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

# --- 5. Main Streamlit App UI ---
st.title("🏗️ Improved 3D Frame Analyzer")
st.write("Define your building grid, sections, and loads to generate and analyze a 3D frame.")

with st.sidebar:
    st.header("1. Frame Geometry"); x_grid = st.text_input("X-spans (m)", "3x6, 5.5"); y_grid = st.text_input("Y-spans (m)", "2x5, 4"); z_grid = st.text_input("Z-heights (m)", "4, 2x3.5")
    st.header("2. Section & Material"); E = st.number_input("E (GPa)", 30.0)*1e6
    with st.expander("Column & Beam Sizes"):
        col_b, col_h = st.number_input("Col b (mm)", 400)/1000, st.number_input("Col h (mm)", 400)/1000
        beam_b, beam_h = st.number_input("Beam b (mm)", 300)/1000, st.number_input("Beam h (mm)", 500)/1000
    st.header("3. Gravity Loads")
    with st.expander("Load Details"):
        slab_d, slab_t = st.number_input("Slab Density (kN/m³)", 25.0), st.number_input("Slab Thickness (m)", 0.150)
        fin_l, live_l = st.number_input("Finishes (kN/m²)", 1.5), st.number_input("Live Load (kN/m²)", 3.0)
    analyze_button = st.button("Generate & Analyze Frame", type="primary")

if analyze_button:
    x_dims, y_dims, z_dims = parse_grid_input(x_grid), parse_grid_input(y_grid), parse_grid_input(z_grid)
    if not all([x_dims, y_dims, z_dims]): st.error("Invalid grid input.")
    else:
        col_p, beam_p = calculate_rc_properties(col_b, col_h, E), calculate_rc_properties(beam_b, beam_h, E)
        q_total = slab_d*slab_t + fin_l + live_l
        with st.spinner("Running Finite Element Analysis..."):
            analysis_results = generate_and_analyze_structure(x_dims, y_dims, z_dims, col_p, beam_p, {'q_total_gravity': q_total})
        if not analysis_results['success']: st.error(f"Analysis Failed: {analysis_results['message']}")
        else:
            st.success("Analysis complete!"); st.session_state['analysis_results'] = analysis_results
            st.subheader("FEA Results Summary")
            summary = analysis_results['summary']
            max_moment = max((abs(e['results'].get('Max_Abs_Moment',0)) for e in analysis_results['elements']), default=0)
            c1,c2,c3 = st.columns(3); c1.metric("Nodes", summary['num_nodes']); c1.metric("Elements", summary['num_elements']); c2.metric("Pressure", f"{q_total:.2f} kN/m²"); c2.metric("K Size", f"{summary['k_shape']}"); c3.metric("Max Moment", f"{max_moment:.2f} kNm")

if 'analysis_results' in st.session_state and st.session_state['analysis_results']['success']:
    results, nodes, elements = st.session_state['analysis_results'], st.session_state['analysis_results']['nodes'], st.session_state['analysis_results']['elements']
    st.subheader("Interactive 3D Visualization"); display_mode = st.selectbox("Display Mode", ['Structure', 'Bending Moment (Myz)'])
    st.plotly_chart(plot_3d_frame(nodes, elements, display_mode), use_container_width=True)
    tab1, tab2, tab3 = st.tabs(["2D Elevation View", "Support Reactions", "Detailed Element Results"])
    with tab1:
        plane_axis = st.radio("Grid Plane", ('X-Z (Y-Gridline)', 'Y-Z (X-Gridline)'))
        if plane_axis == 'X-Z (Y-Gridline)':
            y_coords = sorted(list(set(n['y'] for n in nodes)))
            selected_y = st.selectbox("Select Y-grid", options=y_coords, key='y_coord')
            st.plotly_chart(plot_2d_frame(nodes, elements, 'Y', selected_y), use_container_width=True)
        else:
            x_coords = sorted(list(set(n['x'] for n in nodes)))
            selected_x = st.selectbox("Select X-grid", options=x_coords, key='x_coord')
            st.plotly_chart(plot_2d_frame(nodes, elements, 'X', selected_x), use_container_width=True)
    with tab2:
        support_nodes = {n['id']: n for n in nodes if any(n['restraints'])}
        if support_nodes:
            node_id = st.selectbox("Select support node", options=list(support_nodes.keys()))
            st.dataframe(pd.DataFrame({"Force/Moment": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"], "Value (kN, kNm)": support_nodes[node_id]['reactions']}).round(2))
        else: st.write("No support nodes found.")
    with tab3:
        data = [{'ID':e['id'], 'Start':e['start_node_id'], 'End':e['end_node_id'], 'Max Moment':e['results'].get('Max_Abs_Moment',0), 'Axial':e['results'].get('Axial_Start',0)} for e in elements]
        st.dataframe(pd.DataFrame(data).round(2), use_container_width=True)

