import streamlit as st
import streamlit.components.v1 as components
import json

# --- START OF PYTHON DATA DEFINITION & UTILITY FUNCTIONS ---

# Structural Constants
E_CONCRETE = 30e9 # Elastic Modulus for Concrete (N/m^2)
FRAME_Z_OFFSET = 0

def calculate_rect_properties(d, w, E):
    """
    Calculates Area (A) and Moment of Inertia (Izz, Iyy) for a rectangle.
    d (Depth) is dimension parallel to Y-axis (strong axis for beam).
    w (Width) is dimension parallel to Z-axis (strong axis for column).
    Returns a dictionary of calculated properties.
    """
    A = d * w
    # Izz (Moment of inertia about the Z-axis, or parallel to the width)
    Izz = (w * (d**3)) / 12
    # Iyy (Moment of inertia about the Y-axis, or parallel to the depth)
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

# Node Data (coordinates)
nodes_data = {
    'N1': { 'x': 0, 'y': 0, 'z': FRAME_Z_OFFSET },
    'N2': { 'x': 6, 'y': 0, 'z': FRAME_Z_OFFSET },
    'N3': { 'x': 0, 'y': 4, 'z': FRAME_Z_OFFSET },
    'N4': { 'x': 6, 'y': 4, 'z': FRAME_Z_OFFSET },
}

# Define geometric dimensions first (more centralized)
COL_RECT_D = 0.60
COL_RECT_W = 0.30
B_RECT_D = 0.45
B_RECT_W = 0.23
SQ_COL_D = 0.40
SQ_COL_W = 0.40

# Section Properties: Using the utility function for cleaner creation
section_properties = {
    'COL_RECT': create_section('R-600x300', COL_RECT_D, COL_RECT_W, E_CONCRETE),
    'B_RECT': create_section('R-450x230', B_RECT_D, B_RECT_W, E_CONCRETE),
    'SQ_COL': create_section('R-400x400', SQ_COL_D, SQ_COL_W, E_CONCRETE)
}

# Member Data (sections, connectivity, and calculated internal forces)
members_data = [
    {
        'id': 'C1', 'start': 'N1', 'end': 'N3',
        'section_id': 'COL_RECT', 
        'type': 'Column',
        'forces': {
            'M': { 'max': 30, 'profile': 'linear' }, 
            'V': { 'max': 25, 'profile': 'constant' }
        }
    },
    {
        'id': 'B1', 'start': 'N3', 'end': 'N4',
        'section_id': 'B_RECT',
        'type': 'Beam',
        'forces': {
            'M': { 'max': 100, 'profile': 'parabolic' }, 
            'V': { 'max': 50, 'profile': 'linear' }
        }
    },
    {
        'id': 'C2', 'start': 'N2', 'end': 'N4',
        'section_id': 'SQ_COL', 
        'type': 'Column',
        'forces': {
            'M': { 'max': 35, 'profile': 'linear' }, 
            'V': { 'max': 30, 'profile': 'constant' }
        }
    }
]

# 1. Serialize Python data into JSON strings
nodes_json = json.dumps(nodes_data)
members_json = json.dumps(members_data)
sections_json = json.dumps(section_properties) 

# --- START OF VISUALIZATION HTML/JS CONTENT ---
html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D/2D Structural Frame Visualization</title>
    <!-- Core Three.js Library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <!-- NEW: Orbit Controls for interactive camera manipulation -->
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
            height: 65%; /* Primary view */
            border-bottom: 2px solid #ccc;
            position: relative;
            width: 100%;
        }}
        #canvas-bottom-row {{
            height: 35%; /* Bottom panel */
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
        // --- Global Constants and Setup ---
        const SCALED_SECTION_FACTOR = 0.1; 
        const FORCE_DIAGRAM_SCALE = 0.04; 
        const FRAME_Z_OFFSET = 0; 

        // --- Dynamic Data Structure (Passed from Python) ---
        const nodes = JSON.parse(`{nodes_json}`);
        const members = JSON.parse(`{members_json}`);
        const sections = JSON.parse(`{sections_json}`);


        // --- Three.js Variables ---
        const container3D = document.getElementById('canvas-3d');
        const container2D = document.getElementById('canvas-2d');
        const containerFocused = document.getElementById('canvas-focused-3d');
        const focusedPrompt = document.getElementById('focused-member-prompt');
        const tooltip = document.getElementById('tooltip');

        let scene3D, camera3D, renderer3D, membersGroup3D;
        let controls3D; // NEW: Camera controls variable
        let scene2D, camera2D, renderer2D, membersGroup2D;
        let sceneFocused, cameraFocused, rendererFocused, membersGroupFocused;

        let raycaster, mouse;
        let highlightedMember = null;
        
        // Define world dimensions for orthographic camera calculation
        const worldWidth = 6; 
        const worldHeight = 4; 

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
                offset = new THREE.Vector3(section.w * SCALED_SECTION_FACTOR * 1.5, 0, 0); 
            }} else {{
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
            camera3D.position.set(3, 3, 10);
            camera3D.lookAt(3, 2, 0);
            
            renderer3D = new THREE.WebGLRenderer({{ antialias: true }});
            renderer3D.setSize(container3D.clientWidth, container3D.clientHeight);
            container3D.appendChild(renderer3D.domElement);
            
            // Initialize Orbit Controls
            controls3D = new THREE.OrbitControls(camera3D, renderer3D.domElement);
            controls3D.target.set(3, 2, 0); // Focus on the center of the frame
            controls3D.update();
            controls3D.enableDamping = true; // For smoother rotation

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
            // We attach mousemove listener to the renderer for interaction/highlighting
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
        
        // --- Focused View Rendering Functions ---
        function updateFocusedView(memberData) {{
            while(membersGroupFocused.children.length > 0){{
                membersGroupFocused.remove(membersGroupFocused.children[0]);
            }}
            focusedPrompt.style.display = 'none';

            const isColumn = memberData.type === 'Column';
            const section = sections[memberData.section_id]; 
            const focusedLength = 5; 
            const focusedScaleFactor = focusedLength / 4; 
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
                const profile = memberData.forces[type].profile;

                let offset;
                if (isColumn) {{
                    offset = new THREE.Vector3(offset_x, 0, 0);
                }} else {{
                    offset = new THREE.Vector3(0, offset_y, 0);
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

                    const pointOnMember = new THREE.Vector3().lerpVectors(p1_local, p2_local, t);
                    const magnitude = getForceMagnitude(t) * FORCE_DIAGRAM_SCALE * focusedScaleFactor; 

                    const forceVector = offset.clone().normalize().multiplyScalar(magnitude);
                    const finalPoint = pointOnMember.add(forceVector);
                    points.push(finalPoint);
                }}

                const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), 
                                            new THREE.LineBasicMaterial({{ color: color, linewidth: 3 }}));
                line.position.z = 0.01;
                return line;
            }};

            membersGroupFocused.add(createFocusedForceDiagram('M', 0x3b82f6));
            membersGroupFocused.add(createFocusedForceDiagram('V', 0xf87171));
            
            // Add small nodes at the start/end of the focused member
            const nodeGeo = new THREE.SphereGeometry(SCALED_SECTION_FACTOR * 2.5, 16, 16); 
            const nodeMat = new THREE.MeshPhongMaterial({{ color: 0x000000 }}); 

            const node1 = new THREE.Mesh(nodeGeo, nodeMat);
            node1.position.copy(p1_local);
            membersGroupFocused.add(node1);

            const node2 = new THREE.Mesh(nodeGeo, nodeMat);
            node2.position.copy(p2_local);
            membersGroupFocused.add(node2);


            // Center the camera on the member
            const centerPoint = mesh.position;
            if (isColumn) {{
                cameraFocused.position.set(centerPoint.x + 3.5, centerPoint.y + 0, 5); 
            }} else {{
                cameraFocused.position.set(centerPoint.x + 0.5, centerPoint.y, 5); 
            }}
            cameraFocused.lookAt(centerPoint);
        }}

        function clearFocusedView() {{
             while(membersGroupFocused.children.length > 0){{
                membersGroupFocused.remove(membersGroupFocused.children[0]);
            }}
            focusedPrompt.style.display = 'flex';
        }}


        // --- Interaction and Tooltip (Only on 3D View) ---

        function onMouseMove(event) {{
            // This is only for raycasting (highlighting), not rotation (which is handled by controls)
            const rect = renderer3D.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = - ((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, camera3D);

            const intersects = raycaster.intersectObjects(membersGroup3D.children.filter(obj => obj.type === 'Mesh'));

            if (intersects.length > 0) {{
                const intersect = intersects[0];
                const newHighlightedMember = intersect.object;

                if (highlightedMember !== newHighlightedMember) {{
                    if (highlightedMember) {{
                        highlightedMember.material.color.setHex(highlightedMember.userData.originalColor);
                    }}

                    highlightedMember = newHighlightedMember;
                    highlightedMember.material.color.set(0xfcd34d); 

                    showTooltip(intersect.point, highlightedMember.userData.member);
                    updateFocusedView(highlightedMember.userData.member);
                }}
            }} else {{
                if (highlightedMember) {{
                    highlightedMember.material.color.setHex(highlightedMember.userData.originalColor);
                    highlightedMember = null;

                    clearFocusedView();
                }}
                hideTooltip();
            }}
        }}

        function showTooltip(point, data) {{
            const rect = container3D.getBoundingClientRect();
            const vector = point.clone().project(camera3D);
            const clientX = (vector.x * 0.5 + 0.5) * rect.width + rect.left;
            const clientY = (vector.y * -0.5 + 0.5) * rect.height + rect.top;
            
            // Lookup section properties (now contains A, Izz, Iyy, etc.)
            const section = sections[data.section_id];

            tooltip.style.left = `${{clientX + 10}}px`;
            tooltip.style.top = `${{clientY + 10}}px`;

            // Displaying structural properties
            tooltip.innerHTML = `
                <p class="font-bold text-lg">${{data.id}} (${{data.type}})</p>
                <p>Section: ${{section.name}} ($ {{section.d*100}} x ${{section.w*100}} \\, mm)</p>
                <hr class="my-1 border-yellow-600"/>
                <p>Area (A): <span class="text-gray-200 font-mono">${{section.A.toFixed(3)}} \\, m^2</span></p>
                <p>I_zz: <span class="text-gray-200 font-mono">${{section.Izz.toFixed(5)}} \\, m^4</span></p>
                <p>I_yy: <span class="text-gray-200 font-mono">${{section.Iyy.toFixed(5)}} \\, m^4</span></p>
                <hr class="my-1 border-yellow-600"/>
                <p>Max Moment: <span class="text-blue-400 font-mono">${{data.forces.M.max.toFixed(1)}} \\, kNm</span></p>
                <p>Max Shear: <span class="text-red-400 font-mono">${{data.forces.V.max.toFixed(1)}} \\, kN</span></p>
            `;
            tooltip.classList.add('visible');
        }}

        function hideTooltip() {{
            tooltip.classList.remove('visible');
        }}

        // --- Resize Handler and Animation Loop ---
        function onWindowResize() {{
            // ... (Resize logic remains the same for all cameras/renderers)
            camera3D.aspect = container3D.clientWidth / container3D.clientHeight;
            camera3D.updateProjectionMatrix();
            renderer3D.setSize(container3D.clientWidth, container3D.clientHeight);

            const aspect2D = container2D.clientWidth / container2D.clientHeight;
            const sizeX = worldWidth * 1.1;
            const sizeY = worldHeight * 1.3;
            let viewSize2D;

            if (aspect2D > sizeX / sizeY) {{
                viewSize2D = sizeY; 
            }} else {{
                viewSize2D = sizeX / aspect2D; 
            }}

            const halfSize2D = viewSize2D / 2;
            camera2D.left = -halfSize2D * aspect2D;
            camera2D.right = halfSize2D * aspect2D;
            camera2D.top = halfSize2D;
            camera2D.bottom = -halfSize2D;

            camera2D.updateProjectionMatrix();
            renderer2D.setSize(container2D.clientWidth, container2D.clientHeight);

            cameraFocused.aspect = containerFocused.clientWidth / containerFocused.clientHeight;
            cameraFocused.updateProjectionMatrix();
            rendererFocused.setSize(containerFocused.clientWidth, containerFocused.clientHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);

            // NEW: Update controls only in the 3D view
            if (controls3D) controls3D.update(); 

            if (renderer3D) renderer3D.render(scene3D, camera3D);
            if (renderer2D) renderer2D.render(scene2D, camera2D);
            if (rendererFocused) rendererFocused.render(sceneFocused, cameraFocused);
        }}

        // Start the application
        window.onload = function () {{
            init3D();
            init2D();
            initFocused3D(); 
            animate();
            window.addEventListener('resize', onWindowResize, false);
        }}
    </script>
</body>
</html>
"""
# --- END OF VISUALIZATION HTML/JS CONTENT ---


st.set_page_config(layout="wide", page_title="Structural Frame Viewer")

st.title("Interactive Structural Frame Analysis")
st.markdown("---")

# Use the components.html function to embed the visualization
components.html(
    html_content,
    height=850, 
    scrolling=False
)

st.caption("Visualization powered by Three.js and Streamlit. Main 3D view is now rotatable.")

# --- NEW: Data Visibility Section ---
with st.expander("Show Hardcoded Input Data"):
    st.subheader("Node Coordinates")
    st.json(nodes_data)

    st.subheader("Section Properties")
    st.json(section_properties)

    st.subheader("Member Connectivity and Forces")
    st.json(members_data)
# --- END NEW SECTION ---
