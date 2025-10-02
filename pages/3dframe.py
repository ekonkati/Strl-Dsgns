import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="3D Streamlit Frame Generator")

# --- 1. Geometry Parsing and Generation Functions (Python equivalents of the React logic) ---

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

def generate_grid_geometry(x_lengths, y_heights, z_lengths):
    """
    Generates the structural geometry (nodes and elements) based on parsed grid lengths.
    Returns a DataFrame for nodes and a list of dictionaries for elements.
    """
    nodes = []
    elements = []
    node_id_counter = 1
    element_id_counter = 1

    cum_x = np.cumsum([0] + x_lengths)
    cum_y = np.cumsum([0] + y_heights)
    cum_z = np.cumsum([0] + z_lengths)

    x_grid_count = len(cum_x)
    y_grid_count = len(cum_y)
    z_grid_count = len(cum_z)

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

                # Connection in X direction
                if i < x_grid_count - 1:
                    neighbor_id = node_grid_map[(i + 1, j, k)]
                    elements.append({'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-x'})
                    element_id_counter += 1

                # Connection in Y direction (Columns)
                if j < y_grid_count - 1:
                    neighbor_id = node_grid_map[(i, j + 1, k)]
                    elements.append({'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'column'})
                    element_id_counter += 1

                # Connection in Z direction
                if k < z_grid_count - 1:
                    neighbor_id = node_grid_map[(i, j, k + 1)]
                    elements.append({'id': element_id_counter, 'start': current_id, 'end': neighbor_id, 'type': 'beam-z'})
                    element_id_counter += 1

    return nodes_df, elements, node_coords

# --- 2. Plotly Visualization Function ---

def plot_3d_frame(nodes_df, elements, node_coords):
    """
    Creates an interactive 3D Plotly figure for the frame structure.
    """
    fig = go.Figure()

    # --- 1. Plot Nodes (Scatter Plot) ---
    fig.add_trace(go.Scatter3d(
        x=nodes_df['x'],
        y=nodes_df['y'],
        z=nodes_df['z'],
        mode='markers',
        marker=dict(size=5, color='green', opacity=0.8),
        hoverinfo='text',
        text=[f"Node ID: {row['id']}<br>X:{row['x']:.2f}, Y:{row['y']:.2f}, Z:{row['z']:.2f}" for index, row in nodes_df.iterrows()],
        name='Nodes'
    ))

    # --- 2. Plot Elements (Line Segments) ---
    # Create lists of coordinates for line segments, separated by None for plotting efficiency
    element_x = []
    element_y = []
    element_z = []
    
    # Define colors for different types
    type_colors = {'column': 'darkorchid', 'beam-x': 'royalblue', 'beam-z': 'orange'}
    
    # Group elements by type for trace separation (cleaner legend/styling)
    elements_by_type = {'column': [], 'beam-x': [], 'beam-z': []}
    for elem in elements:
        elements_by_type[elem['type']].append(elem)

    # Create one trace per element type
    for elem_type, elems in elements_by_type.items():
        x_coords, y_coords, z_coords = [], [], []
        
        for elem in elems:
            p1 = node_coords[elem['start']]
            p2 = node_coords[elem['end']]
            
            # Add coordinates for start, end, and None separator
            x_coords.extend([p1[0], p2[0], None])
            y_coords.extend([p1[1], p2[1], None])
            z_coords.extend([p1[2], p2[2], None])

        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines',
            line=dict(color=type_colors[elem_type], width=5),
            name=elem_type.replace('-', ' ').title(),
            hoverinfo='none'
        ))

    # --- 3. Layout Configuration ---
    # Calculate the maximum dimension to ensure consistent scaling
    max_x = nodes_df['x'].max() if not nodes_df.empty else 10
    max_y = nodes_df['y'].max() if not nodes_df.empty else 10
    max_z = nodes_df['z'].max() if not nodes_df.empty else 10
    max_range = max(max_x, max_y, max_z)
    
    # Add a small buffer to the max range for better visualization padding
    plot_limit = max_range * 1.1 
    
    fig.update_layout(
        title='Interactive 3D Frame Geometry',
        scene=dict(
            xaxis=dict(title='X (Length)', showgrid=True, zeroline=False, range=[0, plot_limit]),
            yaxis=dict(title='Y (Height)', showgrid=True, zeroline=False, range=[0, plot_limit]),
            zaxis=dict(title='Z (Width)', showgrid=True, zeroline=False, range=[0, plot_limit]),
            aspectmode='data' # Ensures aspect ratio is 1:1:1
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700
    )

    return fig

# --- 3. Streamlit App Layout ---

# Sidebar for User Inputs
st.sidebar.header("📐 Define Grid Geometry")
st.sidebar.markdown("Use format: `L1, N2xL2, ...` (e.g., `5, 2x4.5`). All units are arbitrary (e.g., meters).")

# Input fields
default_x = "3x5, 2x4"
x_input = st.sidebar.text_input(
    "X Direction Bays (Length)", 
    default_x,
    help="Example: 3x5, 4, 2x6"
)

default_y = "4.5, 3x3.5"
y_input = st.sidebar.text_input(
    "Y Direction Heights (Floor)", 
    default_y,
    help="Example: 4.5, 3x3.5 (Ground + 3 floors)"
)

default_z = "4x6"
z_input = st.sidebar.text_input(
    "Z Direction Bays (Width)", 
    default_z,
    help="Example: 4x6, 5"
)

# Button to generate structure
if st.sidebar.button("Generate & Visualize 🏗️"):
    # Set the state flag to trigger the main logic block
    st.session_state['run_generation'] = True

# Main Page Content
st.title("3D Frame Geometry Generator")
st.markdown("""
This application models the geometry of a 3D frame structure based on flexible bay and floor inputs.
The structure is visualized using an interactive 3D plot powered by **Plotly**.
""")

# --- 4. Main Logic ---

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
        with st.spinner('Generating 3D geometry...'):
            nodes_df, elements, node_coords = generate_grid_geometry(x_lengths, y_heights, z_lengths)
        
        st.success(f"Generated successfully: {len(nodes_df)} nodes and {len(elements)} elements.")

        # 4. Visualization
        frame_fig = plot_3d_frame(nodes_df, elements, node_coords)
        st.plotly_chart(frame_fig, use_container_width=True)

        # 5. Display Summary Data
        st.subheader("Structure Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Total Nodes:** `{len(nodes_df)}`")
            st.markdown(f"**Total Elements:** `{len(elements)}`")
        with col2:
            st.markdown(f"**X Dimensions:** `{x_lengths}`")
            st.markdown(f"**Y Dimensions (Heights):** `{y_heights}`")
            st.markdown(f"**Z Dimensions:** `{z_lengths}`")

        st.subheader("Node Coordinates (First 10)")
        st.dataframe(nodes_df[['id', 'x', 'y', 'z']].head(10))

        # Reset flag for immediate re-run on subsequent button clicks
        st.session_state['run_generation'] = False
