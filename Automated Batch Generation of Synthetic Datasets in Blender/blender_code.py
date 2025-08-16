"""
Automated Batch Generation of Synthetic Datasets in Blender
-----------------------------------
This script is designed for automatic dataset generation by randomizing:
- Object pose
- Tabletop placement & scaling
- HDRI environment lighting strength
- Additional light sources (type, position, size, intensity)
- Camera pose

Related to the paper " Transparent Fragments Contour Estimation via
 Visual-Tactile Fusion for Autonomous Reassembly" (Lin et al., 2025)

Uploaded:2025-08-17 
"""
import bpy
import sys
import os
import random
import numpy as np
import math
from mathutils import Vector

# List of target object names
target_names = [
    "cup_fracture_4_78",
    "cup_fracture_6_31",
    "cup_fracture_6_44"
]

# Load target objects and assign pass_index
targets = []
for i, name in enumerate(target_names, start=1):
    obj = bpy.data.objects.get(name)
    if obj:
        obj.pass_index = i
        targets.append(obj)
    else:
        print(f"Error: Target object '{name}' not found")
        sys.exit()

# Reference plane object
plane_object_name = "Plane"
plane_object = bpy.data.objects.get(plane_object_name)
if not plane_object:
    print(f"Error: Plane object '{plane_object_name}' not found")
    sys.exit()

# Reference light object
light_object_name = "Light"
light_object = bpy.data.objects.get(light_object_name)
if not light_object:
    print(f"Error: Light object '{light_object_name}' not found")

# Render settings
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.image_settings.color_mode = 'RGB'
bpy.context.scene.cycles.samples = 256
bpy.context.scene.cycles.device = 'GPU'

# Number of images to render
num_images = 20
start_index = 0
use_light = 1
min_distance = 0.74  # minimal allowed distance between center of target objects
position_range = 1.4  # Range for randomizing positions of target objects on XY plane

# ========================Output directories============================
rgb_dir = r"<YOUR_RGB_SAVE_DIR>"
mask_dir = r"<YOUR_MASK_SAVE_DIR>"
os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
#========================================================================

# Enable object index pass
bpy.context.scene.view_layers["ViewLayer"].use_pass_object_index = True

# Setup compositor nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Clear existing nodes
for node in tree.nodes:
    tree.nodes.remove(node)

render_layer = tree.nodes.new('CompositorNodeRLayers')

# Create IDMask nodes dynamically
id_masks = []
for i in range(len(targets)):
    id_mask = tree.nodes.new('CompositorNodeIDMask')
    id_mask.index = i + 1
    id_masks.append(id_mask)

# Create Math nodes to merge all masks
math_nodes = []
for _ in range(len(targets) - 1):
    math_node = tree.nodes.new('CompositorNodeMath')
    math_node.operation = 'ADD'
    math_node.use_clamp = True
    math_nodes.append(math_node)

viewer = tree.nodes.new('CompositorNodeViewer')

for id_mask in id_masks:
    links.new(render_layer.outputs["IndexOB"], id_mask.inputs[0])

if len(targets) == 1:
    links.new(id_masks[0].outputs[0], viewer.inputs[0])
else:
    links.new(id_masks[0].outputs[0], math_nodes[0].inputs[0])
    links.new(id_masks[1].outputs[0], math_nodes[0].inputs[1])

    for i in range(1, len(targets) -1):
        links.new(math_nodes[i-1].outputs[0], math_nodes[i].inputs[0])
        links.new(id_masks[i+1].outputs[0], math_nodes[i].inputs[1])

    links.new(math_nodes[-1].outputs[0], viewer.inputs[0])

bpy.context.scene.node_tree.nodes.active = viewer

def distance_between(obj1, obj2):
    pos1 = obj1.location
    pos2 = obj2.location
    return ( (pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2 ) ** 0.5

def check_and_reposition(objects, min_distance):
    # Randomize position and rotation
    for obj in objects:
        obj.location.x = random.uniform(-position_range, position_range)
        obj.location.y = random.uniform(-position_range, position_range)
        obj.rotation_euler.z = random.uniform(0, 2 * math.pi)
    
    # Check pairwise distances and reposition if needed
    def too_close():
        for i in range(len(objects)):
            for j in range(i+1, len(objects)):
                if distance_between(objects[i], objects[j]) < min_distance:
                    return True
        return False

    while too_close():
        for obj in objects:
            obj.location.x = random.uniform(-position_range, position_range)
            obj.location.y = random.uniform(-position_range, position_range)

for i in range(num_images):
    # Reset empty reference
    current_empty = None

    # Reposition targets ensuring minimal distances
    check_and_reposition(targets, min_distance)

    # Camera setup
    camera = bpy.data.objects['Camera']
    camera.constraints.clear()

    if random.random() < 0.3:
        # Control the camera to shoot overhead.
        x = random.uniform(-0.03, 0.03)
        y = random.uniform(-0.03, 0.03)
        z = random.uniform(6.9, 8)
        rot_x = random.uniform(0, math.pi * 0.002)
        rot_y = random.uniform(0, math.pi * 0.002)
        rot_z = random.uniform(0, math.pi * 2)

        camera.location = (x, y, z)
        camera.rotation_mode = 'ZYX'
        camera.rotation_euler = (rot_x, rot_y, rot_z)
    else:
        # Camera moves randomly on a ring and looks at center of all target objects
        radius = random.uniform(2, 5.5)
        angle = random.uniform(0, 2 * math.pi)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = random.uniform(4.2, 7.4)
        rot_x = 0
        rot_y = 0
        rot_z = 0

        # Calculate center of all targets + small noise
        center_x = sum(obj.location.x for obj in targets) / len(targets) + random.uniform(-0.15, 0.15)
        center_y = sum(obj.location.y for obj in targets) / len(targets) + random.uniform(-0.15, 0.15)
        center_z = sum(obj.location.z for obj in targets) / len(targets)

        camera.location = (x, y, z)
        camera.rotation_mode = 'ZYX'
        camera.rotation_euler = (0, 0, 0)

        # Create Empty target for TRACK_TO constraint
        current_empty = bpy.data.objects.new("Empty", None)
        bpy.context.collection.objects.link(current_empty)
        current_empty.location = (center_x, center_y, center_z)

        track_to_constraint = camera.constraints.new(type='TRACK_TO')
        track_to_constraint.target = current_empty
        track_to_constraint.track_axis = 'TRACK_NEGATIVE_Z'
        track_to_constraint.up_axis = 'UP_Y'

        bpy.context.view_layer.update()

        # Restore rotation (optional)
        camera.rotation_euler = (rot_x, rot_y, rot_z)

    # Randomize plane position and rotation
    plane_x = random.uniform(-2, 2)
    plane_y = random.uniform(-2, 2)
    plane_z = plane_object.location.z
    plane_object.location = (plane_x, plane_y, plane_z)
    plane_object.scale[0] = plane_object.scale[1] = random.uniform(4.1, 6.3)  # Randomize plane scale
    plane_rot_z = random.uniform(0, 2 * math.pi)
    plane_object.rotation_euler = (0, 0, plane_rot_z)

    # Light setup
    if light_object:
        if use_light == 1:
            light_random = random.random()
            if light_random < 0.4:
                light_object.data.energy = 0
            elif light_random < 0.7:
                light_object.data.type = 'POINT'
                light_object.data.energy = random.uniform(100, 1600)
                light_object.data.shadow_soft_size = random.uniform(0.5, 2)

                light_x = random.uniform(-6, 6)
                light_y = random.uniform(-6, 6)
                light_z = random.uniform(6, 10)
                light_object.location = (light_x, light_y, light_z)
            else:
                light_object.data.type = 'AREA'
                light_object.data.energy = random.uniform(100, 2000)
                light_object.data.size = random.uniform(1, 6)
                light_object.data.size_y = random.uniform(1, 6)

                rot_x_light = math.radians(random.uniform(-30, 30))
                rot_y_light = math.radians(random.uniform(-30, 30))
                light_object.rotation_euler = (rot_x_light, rot_y_light, 0)

                light_x = random.uniform(-5.5, 5.5)
                light_y = random.uniform(-5.5, 5.5)
                light_z = random.uniform(6, 10)
                light_object.location = (light_x, light_y, light_z)
        else:
            light_object.data.energy = 0
    else:
        print("Warning: Light object not found, skipping light setup")

    # Randomize world background strength
    if bpy.context.scene.world and bpy.context.scene.world.use_nodes:
        background_node = bpy.context.scene.world.node_tree.nodes.get("Background")
        if background_node:
            background_node.inputs[1].default_value = random.uniform(0.8, 1.7)
        else:
            print("Warning: World background node not found")

    # Set output file paths
    rgb_path = os.path.join(rgb_dir, f"{i + start_index}.png")
    mask_path = os.path.join(mask_dir, f"{i + start_index}.npy")

    bpy.context.scene.render.filepath = rgb_path

    bpy.ops.render.render(write_still=True)

    viewer_image = bpy.data.images.get("Viewer Node")
    if viewer_image:
        mask_array = np.array(viewer_image.pixels[:], dtype=np.int8)
        mask_array = mask_array.reshape((bpy.context.scene.render.resolution_y,
                                        bpy.context.scene.render.resolution_x, 4))
        mask_array = np.flipud(mask_array[:, :, 0])
        np.save(mask_path, mask_array)
        print(f"RGB image saved to: {rgb_path}")
        print(f"Mask saved to: {mask_path}")
    else:
        print("Error: Failed to get mask data from Viewer node")

    # Remove created Empty object to avoid clutter
    if current_empty and current_empty.name in bpy.data.objects:
        bpy.data.objects.remove(current_empty, do_unlink=True)

print("Task completed")



# Note: The below function 'get_object_center_2d' is not used in this script.
# It is kept here because it calculates the projection of a point in the scene 
# into the camera view, which can be used to output, for example, the center point 
# of the target object or the positions of all its corner points in the camera view.
def get_object_center_2d(obj):
    camera = bpy.context.scene.camera
    render = bpy.context.scene.render

    # Get the center of the bounding box (local coordinates)
    local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    # Convert to world coordinates
    global_bbox_center = obj.matrix_world @ local_bbox_center

    # Convert world coordinates to camera coordinates
    camera_matrix = camera.matrix_world.inverted()
    point_camera = camera_matrix @ global_bbox_center

    if camera.data.type == 'PERSP':
        x = point_camera.x / -point_camera.z
        y = point_camera.y / -point_camera.z
    else:
        x = point_camera.x
        y = point_camera.y

    render_scale = render.resolution_percentage / 100
    width = int(render.resolution_x * render_scale)
    height = int(render.resolution_y * render_scale)

    sensor = camera.data.sensor_width
    focal_length = camera.data.lens
    pixel_aspect = render.pixel_aspect_x / render.pixel_aspect_y

    if width >= height:
        f_x = (width * focal_length) / sensor
        f_y = f_x / pixel_aspect
    else:
        f_y = (height * focal_length) / sensor
        f_x = f_y * pixel_aspect

    c_x = width / 2
    c_y = height / 2

    u = c_x + (x * f_x)
    v = c_y - (y * f_y)  # Adjust Y coordinate calculation

    return np.array([u, v], dtype=np.float32)
