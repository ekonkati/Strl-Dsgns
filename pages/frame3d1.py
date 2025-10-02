import streamlit as st
import streamlit.components.v1 as components
import json

# Structural Constants (can be made input fields later)
E_CONCRETE = 30e9 # Elastic Modulus for Concrete (N/m^2)
FRAME_Z_OFFSET = 0
FORCE_SCALE = 0.04 # Multiplier for visualization

# --- Data Generation Utility Functions ---

def calculate_rect_properties(d, w, E):
    """
    Calculates Area (A) and Moment of Inertia (Izz, Iyy) for a rectangle.
    d (Depth) is dimension parallel to Y-axis (strong axis for beam).
    w (Width) is dimension parallel to Z-axis (strong axis for column).
    Returns a dictionary of calculated properties.
    """
    A = d * w
    Izz = (w * (d**3)) / 12
    Iyy = (d * (w**3)) / 12
    return {'A': A, 'Izz': Izz, 'Iyy': Iyy, 'E': E}

def create_section(name, d, w, E):
    """Utility to create a complete section definition dictionary."""
    return {
        'name': name, 
        'd': d, 
        'w': w, 
        **calculate_rect_properties(d, w, E)
    }

def generate_frame_data(span_x, height_y, col_d, col_w, beam_d, beam_w):
    """
    Generates all node, section, and member data based on user inputs.
    """
    # 1. Node Data (based on Span and Height)
    nodes_data = {
        'N1': { 'x': 0, 'y': 0, 'z': FRAME_Z_OFFSET },
        'N2': { 'x': span_x, 'y': 0, 'z': FRAME_Z_OFFSET },
        'N3': { 'x': 0, 'y': height_y, 'z': FRAME_Z_OFFSET },
        'N4': { 'x': span_x, 'y': height_y, 'z': FRAME_Z_OFFSET },
    }

    # 2. Section Properties
    section_properties = {
        'COLUMN': create_section('Column', col_d, col_w, E_CONCRETE),
        'BEAM': create_section('Beam', beam_d, beam_w, E_CONCRETE),
    }

    # Mock Force Data (These should come from analysis, but are mocked here for visualization)
    # The forces scale relative to geometry for a basic visual representation.
    M_MAX = span_x * height_y * 10 
    V_MAX = span_x * height_y * 8

    # 3. Member Data (connectivity and forces)
    members_data = [
        {
            'id': 'C1', 'start': 'N1', 'end': 'N3',
            'section_id': 'COLUMN', 
            'type': 'Column',
            'forces': {
                'M': { 'max': M_MAX * 0.8, 'profile': 'linear' }, 
                'V': { 'max': V_MAX * 0.6, 'profile': 'constant' }
            }
        },
        {
            'id': 'B1', 'start': 'N3', 'end': 'N4',
            'section_id': 'BEAM',
            'type': 'Beam',
            'forces': {
                'M': { 'max': M_MAX, 'profile': 'parabolic' }, 
                'V': { 'max': V_MAX, 'profile': 'linear' }
            }
        },
        {
            'id': 'C2', 'start': 'N2', 'end': 'N4',
            'section_id': 'COLUMN', 
            'type': 'Column',
            'forces': {
                'M': { 'max': M_MAX * 0.7, 'profile': 'linear' }, 
                'V': { 'max': V_MAX * 0.5, 'profile': 'constant' }
            }
        }
    ]
    return nodes_data, members_data, section_properties

# --- Streamlit Layout and Inputs ---

st.set_page_config(layout="wide", page_title="Interactive Structural Frame Analysis")

st.title("Interactive Structural Frame Analysis")
st.markdown("Adjust the geometry and section sizes below to see the 3D frame update in real-time.")
st.markdown("---")

# 1. Input Sidebar (The recommended place for controls)
with st.sidebar:
    st.header("Frame Geometry (m)")
    span_x = st.slider("Span (X-axis)", min_value=3.0, max_value=12.0, value=6.0, step=0.5)
    height_y = st.slider("Height (Y-axis)", min_value=2.0, max_value=8.0, value=4.0, step=0.5)
    
    st.header("Section Sizes (m)")
    st.subheader("Columns (B x H)")
    col_w = st.number_input("Column Width (W, Z-axis)", min_value=0.1, max_value=1.0, value=0.30, step=0.05, format="%.2f")
    col_d = st.number_input("Column Depth (D, Y-axis)", min_value=0.1, max_value=1.0, value=0.60, step=0.05, format="%.2f")
    
    st.subheader("Beams (B x H)")
    beam_w = st.number_input("Beam Width (W, Z-axis)", min_value=0.1, max_value=1.0, value=0.23, step=0.05, format="%.2f")
    beam_d = st.number_input("Beam Depth (D, Y-axis)", min_value=0.1, max_value=1.0, value=0.45, step=0.05, format="%.2f")

# 2. Generate Data based on Inputs
nodes_data, members_data, section_properties = generate_frame_data(
    span_x, height_y, col_d, col_w, beam_d, beam_w
)

# 3. Serialize Python data into JSON strings for JavaScript
nodes_json = json.dumps(nodes_data)
members_json = json.dumps(members_data)
sections_json = json.dumps(section_properties) 

# --- START OF VISUALIZATION HTML/JS CONTENT ---
# Injecting Python variables directly into the JavaScript string using f-string syntax
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Structural Frame Visualization</title>
    <!-- Core Three.js Library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- Orbit Controls for interactive camera manipulation -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/controls/OrbitControls.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom CSS for layout and aesthetics */
        body {{
            font-family: 'Inter', sans-serif;
            margin: 0;
            background-color: #f0f4f8;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            height: 100vh;
        }}
        /* ... (styles for info-panel, tooltip, visualization-wrapper, canvas-3d, etc. are the same) ... */
        #info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            max-width: 300px;
            z-index: 10;
        }}
        #tooltip {{
            position: absolute;
            background-color: #1f2937;
            color: #fcd34d;
            padding: 8px 12px;
            border-radius: 6px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 20;
            font-size: 0.875rem;
            line-height: 1.4;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }}
        #tooltip.visible {{
            opacity: 1;
        }}
        #visualization-wrapper {{
            width: 100vw;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        #canvas-3d {{
            height: 65%;
            border-bottom: 2px solid #ccc;
            position: relative;
            width: 100%;
        }}
        #canvas-bottom-row {{
            height: 35%; 
            width: 100%;
            display: flex;
        }}
        #canvas-2d, #canvas-focused-3d {{
            width: 50%;
            position: relative;
        }}
        #canvas-2d {{
            border-right: 1px solid #ddd;
        }}
        canvas {{
            display: block;
        }}
    </style>
</head>
<body>

    <div id="info-panel">
        <h2 class="text-lg font-bold mb-2 text-gray-800">Frame Visualization</h2>
        <p class="text-sm text-gray-600">Interact and highlight in the **3D View** (top). **Click and drag to rotate the view.**</p>
        <div class="mt-3 p-2 bg-blue-50 border-l-4 border-blue-500 rounded text-sm">
            <p class="font-semibold text-blue-800">Legend:</p>
            <p class="text-blue-700">Moment Diagram: <span class="font-mono text-blue-500">Blue</span></p>
            <p class="text-blue-700">Shear Diagram: <span class="font-mono text-red-500">Red</span></p>
        </div>
    </div>

    <div id="tooltip"></div>
    <div id="visualization-wrapper">
        <div id="canvas-3d">
            <div class="absolute top-2 left-2 p-2 bg-white rounded-lg shadow-md text-sm font-semibold">3D Interactive Frame View</div>
        </div>
        <div id="canvas-bottom-row">
            <div id="canvas-2d">
                <div class="absolute top-2 left-2 p-2 bg-white rounded-lg shadow-md text-sm font-semibold">2D Frame Section (X-Y Elevation)</div>
            </div>
            <div id="canvas-focused-3d">
                <div class="absolute top-2 left-2 p-2 bg-white rounded-lg shadow-md text-sm font-semibold">Focused 3D Member View</div>
                <div id="focused-member-prompt" class="absolute inset-0 flex items-center justify-center text-gray-500 bg-gray-100/70 pointer-events-none rounded-lg p-4 m-2">
                    <p class="text-center">Hover over a member in the main 3D view to inspect it here.</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // --- Global Constants and Setup (Updated to use dynamic values) ---
        const SCALED_SECTION_FACTOR = 0.1; 
        const FORCE_DIAGRAM_SCALE = {FORCE_SCALE}; // Passed from Python
        const FRAME_Z_OFFSET = 0; 

        // --- Dynamic Data Structure (Passed from Python as JSON strings) ---
        const nodes = JSON.parse(`{nodes_json}`);
        const members = JSON.parse(`{members_json}`);
        const sections = JSON.parse(`{sections_json}`);
        
        // Dynamic world dimensions for orthographic camera calculation
        const worldWidth = {span_x}; 
        const worldHeight = {height_y}; 

        // --- Three.js Variables ---
        const container3D = document.getElementById('canvas-3d');
        const container2D = document.getElementById('canvas-2d');
        const containerFocused = document.getElementById('canvas-focused-3d');
        const focusedPrompt = document.getElementById('focused-member-prompt');
        const tooltip = document.getElementById('tooltip');

        let scene3D, camera3D, renderer3D, membersGroup3D;
        let controls3D; 
        let scene2D, camera2D, renderer2D, membersGroup2D;
        let sceneFocused, cameraFocused, rendererFocused, membersGroupFocused;

        let raycaster, mouse;
        let highlightedMember = null;
        
        // --- Core Geometry Utility Functions ---
        
        /**
         * Creates a scaled member geometry mesh.
         */
        function createScaledMember(memberData, is2DView = false, p1, p2) {{
            const isColumn = memberData.type === 'Column';
            const section = sections[memberData.section_id]; 

            const length = p1.distanceTo(p2);

            // Create Box Geometry based on cross-section (scaled)
            const dim1 = section.d * SCALED_SECTION_FACTOR;
            const dim2 = section.w * SCALED_SECTION_FACTOR;

            const geometry = new THREE.BoxGeometry(
                isColumn ? dim2 : length,
                isColumn ? length : dim1,
                isColumn ? dim1 : dim2
            );

            // Material: Grey for structure
            const material = new THREE.MeshLambertMaterial({{ 
                color: is2DView ? 0x4b5563 : 0x5a67d8, 
                transparent: true, 
                opacity: 0.9 
            }});
            const mesh = new THREE.Mesh(geometry, material);

            // Position and Rotate the member
            mesh.position.addVectors(p1, p2).divideScalar(2);
            
            if (!is2DView) {{
                 mesh.userData.originalColor = material.color.getHex();
                 mesh.userData.member = memberData; // Attach member data
            }}

            return mesh;
        }}

        /**
         * Creates a force diagram profile (Moment/Shear).
         */
        function createForceDiagram(memberData, forceType, lineColor, is2DView = false, p1, p2) {{
            const isColumn = memberData.type === 'Column';
            const section = sections[memberData.section_id]; 
            
            const points = [];
            const segments = 20;
            const maxForce = memberData.forces[forceType].max;
            const profile = memberData.forces[forceType].profile;

            let offset;
            if (isColumn) {{
                // Offset perpendicular to the column, along the Z-axis for a beam force
                offset = new THREE.Vector3(section.w * SCALED_SECTION_FACTOR * 1.5, 0, 0); 
            }} else {{
                // Offset perpendicular to the beam, along the Y-axis for gravity load
                offset = new THREE.Vector3(0, -section.d * SCALED_SECTION_FACTOR * 1.5, 0); 
            }}

            const getForceMagnitude = (t) => {{
                switch (profile) {{
                    case 'constant': return maxForce;
                    case 'linear': return maxForce * (1 - t); 
                    case 'parabolic': return maxForce * 4 * t * (1 - t); 
                    default: return 0;
                }}
            }};

            for (let i = 0; i <= segments; i++) {{
                const t = i / segments;

                const pointOnMember = new THREE.Vector3().lerpVectors(p1, p2, t);
                const magnitude = getForceMagnitude(t) * FORCE_DIAGRAM_SCALE;
                const forceVector = offset.clone().normalize().multiplyScalar(magnitude);

                const finalPoint = pointOnMember.add(forceVector);
                points.push(finalPoint);
            }}

            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const material = new THREE.LineBasicMaterial({{ 
                color: lineColor, 
                linewidth: is2DView ? 5 : 2 
            }});
            const line = new THREE.Line(geometry, material);
            
            line.position.z = 0.01; 
            
            return line;
        }}
        
        /**
         * Creates node geometry (small spheres).
         */
        function createNodeGeometry(sceneGroup, is2DView) {{
            if (is2DView) return; 

            for (const nodeId in nodes) {{
                const nodeData = nodes[nodeId];
                const position = new THREE.Vector3(nodeData.x, nodeData.y, nodeData.z);
                
                const geometry = new THREE.SphereGeometry(SCALED_SECTION_FACTOR * 1.5, 16, 16); 
                const material = new THREE.MeshPhongMaterial({{ color: 0x000000, shininess: 100 }}); 
                const nodeMesh = new THREE.Mesh(geometry, material);
                nodeMesh.position.copy(position);
                
                sceneGroup.add(nodeMesh);
            }}
        }}

        // --- Frame Assembly Function ---
        function createFrameGeometry(sceneGroup, is2DView) {{
            // Clear existing geometry before redrawing
            sceneGroup.children.forEach(child => child.geometry.dispose());
            while(sceneGroup.children.length > 0){{
                sceneGroup.remove(sceneGroup.children[0]);
            }}

            // 1. Draw Members and Force Diagrams
            members.forEach(memberData => {{
                const startNode = nodes[memberData.start];
                const endNode = nodes[memberData.end];
                
                const p1 = new THREE.Vector3(startNode.x, startNode.y, startNode.z);
                const p2 = new THREE.Vector3(endNode.x, endNode.y, endNode.z);

                const mesh = createScaledMember(memberData, is2DView, p1, p2);
                sceneGroup.add(mesh);

                const momentLine = createForceDiagram(memberData, 'M', 0x3b82f6, is2DView, p1, p2);
                sceneGroup.add(momentLine);

                const shearLine = createForceDiagram(memberData, 'V', 0xf87171, is2DView, p1, p2);
                sceneGroup.add(shearLine);
            }});

            // 2. Draw Nodes (only in the main 3D view)
            createNodeGeometry(sceneGroup, is2DView);
        }}


        // --- Initialization Functions ---
        
        function init3D() {{
            scene3D = new THREE.Scene();
            scene3D.background = new THREE.Color(0xe5e7eb); 
            
            camera3D = new THREE.PerspectiveCamera(50, container3D.clientWidth / container3D.clientHeight, 0.1, 1000);
            camera3D.position.set(worldWidth, worldHeight/2 + 2, worldWidth * 1.5);
            camera3D.lookAt(worldWidth / 2, worldHeight / 2, 0);
            
            renderer3D = new THREE.WebGLRenderer({{ antialias: true }});
            renderer3D.setSize(container3D.clientWidth, container3D.clientHeight);
            container3D.appendChild(renderer3D.domElement);
            
            // Initialize Orbit Controls
            controls3D = new THREE.OrbitControls(camera3D, renderer3D.domElement);
            controls3D.target.set(worldWidth / 2, worldHeight / 2, 0); 
            controls3D.update();
            controls3D.enableDamping = true; 

            // Lighting Setup
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            scene3D.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(5, 10, 7.5);
            scene3D.add(directionalLight);

            membersGroup3D = new THREE.Group();
            scene3D.add(membersGroup3D);

            createFrameGeometry(membersGroup3D, false);
            
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();
            renderer3D.domElement.addEventListener('mousemove', onMouseMove, false);
        }}

        function init2D() {{
            scene2D = new THREE.Scene();
            scene2D.background = new THREE.Color(0xf0f4f8); 

            const aspect = container2D.clientWidth / container2D.clientHeight;
            const sizeX = worldWidth * 1.1; 
            const sizeY = worldHeight * 1.3; 
            
            let viewSize;
            if (aspect > sizeX / sizeY) {{
                viewSize = sizeY; 
            }} else {{
                viewSize = sizeX / aspect; 
            }}

            const halfSize = viewSize / 2;
            camera2D = new THREE.OrthographicCamera(
                -halfSize * aspect, halfSize * aspect,
                halfSize, -halfSize,
                0.1, 1000
            );

            camera2D.position.set(worldWidth / 2, worldHeight / 2, 5);
            camera2D.lookAt(worldWidth / 2, worldHeight / 2, 0);

            renderer2D = new THREE.WebGLRenderer({{ antialias: true }});
            renderer2D.setSize(container2D.clientWidth, container2D.clientHeight);
            container2D.appendChild(renderer2D.domElement);

            scene2D.add(new THREE.AmbientLight(0xffffff, 0.8));

            membersGroup2D = new THREE.Group();
            scene2D.add(membersGroup2D);

            createFrameGeometry(membersGroup2D, true); 
        }}

        function initFocused3D() {{
            sceneFocused = new THREE.Scene();
            sceneFocused.background = new THREE.Color(0xf5f5f5); 
            
            cameraFocused = new THREE.PerspectiveCamera(60, containerFocused.clientWidth / containerFocused.clientHeight, 0.1, 100);
            cameraFocused.position.set(0, 0, 5); 
            cameraFocused.lookAt(0, 0, 0);
            
            rendererFocused = new THREE.WebGLRenderer({{ antialias: true }});
            rendererFocused.setSize(containerFocused.clientWidth, containerFocused.clientHeight);
            containerFocused.appendChild(rendererFocused.domElement);

            sceneFocused.add(new THREE.AmbientLight(0xffffff, 0.9));
            sceneFocused.add(new THREE.DirectionalLight(0xffffff, 0.5).position.set(3, 5, 2));

            membersGroupFocused = new THREE.Group();
            sceneFocused.add(membersGroupFocused);
        }}
        
        // --- Focused View Rendering Functions (same logic) ---
        function updateFocusedView(memberData) {{
            while(membersGroupFocused.children.length > 0){{
                membersGroupFocused.remove(membersGroupFocused.children[0]);
            }}
            focusedPrompt.style.display = 'none';

            const isColumn = memberData.type === 'Column';
            const section = sections[memberData.section_id]; 
            const focusedLength = 5; 
            const focusedSectionScale = SCALED_SECTION_FACTOR * 2.0; 

            let p1_local, p2_local;
            let offset_x, offset_y;

            if (isColumn) {{
                p1_local = new THREE.Vector3(0, 0, 0); 
                p2_local = new THREE.Vector3(0, focusedLength, 0); 
                offset_x = section.w * focusedSectionScale * 1.5;
                offset_y = 0;
            }} else {{
                p1_local = new THREE.Vector3(0, section.d, 0); 
                p2_local = new THREE.Vector3(focusedLength, section.d, 0); 
                offset_x = 0;
                offset_y = -section.d * focusedSectionScale * 1.5;
            }}
            
            const dim1 = section.d * focusedSectionScale;
            const dim2 = section.w * focusedSectionScale;

            const geometry = new THREE.BoxGeometry(
                isColumn ? dim2 : focusedLength, 
                isColumn ? focusedLength : dim1,  
                isColumn ? dim1 : dim2            
            );
            
            const mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({{ color: 0x5a67d8, transparent: false }}));
            mesh.position.addVectors(p1_local, p2_local).divideScalar(2);
            membersGroupFocused.add(mesh);

            // Re-create Force Diagrams (using local coordinates)
            
            const createFocusedForceDiagram = (type, color) => {{
                const points = [];
                const segments = 20;
                const maxForce = memberData.forces[type].max;
