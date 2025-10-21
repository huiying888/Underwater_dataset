import os
import random
import math
import pygame
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models.networks import define_G

# =======================================================
# CONFIG
# =======================================================
IMG_WIDTH, IMG_HEIGHT = 512, 256
SAVE_DIR = "synthetic_dataset_pairs"
EDGE_COLOR = (0, 0, 0)
BG_COLOR = (255, 255, 255)
FISH_COLOR = (255, 0, 0)  # Red for fish
TERRAIN_COLOR = (0, 0, 255)  # Blue for terrain
NUM_IMAGES = 10  # number of images to generate
MODEL_NAME = "sonar_geometry_to_photo"
CHECKPOINT_DIR = f"./checkpoints/{MODEL_NAME}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
pygame.init()
surface = pygame.Surface((IMG_WIDTH, IMG_HEIGHT))

# =======================================================
# MODEL LOADING (Pix2Pix Generator)
# =======================================================
print(f"Loading Pix2Pix generator from {CHECKPOINT_DIR}/latest_net_G.pth ...")
netG = define_G(3, 3, 64, "unet_256", norm="batch", use_dropout=False, init_type="normal", init_gain=0.02)
netG.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "latest_net_G.pth"), map_location=DEVICE))
netG.to(DEVICE)
netG.eval()
print("✅ Model loaded successfully.")

# =======================================================
# IMAGE TRANSFORMS
# =======================================================
to_tensor = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

to_pil = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    transforms.ToPILImage()
])

# =======================================================
# SYNTHETIC GENERATION FUNCTIONS
# =======================================================
def draw_smooth_terrain_edge(surface):
    """Draw smooth terrain with gentle overall ups and downs."""
    points = []
    base_y = random.randint(int(IMG_HEIGHT * 0.8), int(IMG_HEIGHT * 0.9))
    
    # Smooth terrain with gentle overall variation
    for x in range(0, IMG_WIDTH + 1, 5):
        # Large gentle waves for overall shape
        wave = 20 * math.sin(x * 0.004) + 10 * math.sin(x * 0.012)
        # Small noise for texture
        noise = random.uniform(-2, 2)
        y = base_y + wave + noise
        points.append((x, int(y)))
    
    # Fill terrain area
    if len(points) > 1:
        terrain_polygon = points + [(IMG_WIDTH, IMG_HEIGHT), (0, IMG_HEIGHT)]
        pygame.draw.polygon(surface, TERRAIN_COLOR, terrain_polygon)
        pygame.draw.lines(surface, EDGE_COLOR, False, points, 2)
    
    return points



def draw_fish_arc_edges(surface, terrain_points):
    """Draw fish arches with realistic distribution patterns."""
    # Random fish scenarios
    scenario = random.choice(['none', 'few', 'scattered', 'grouped', 'many'])
    
    if scenario == 'none':
        return
    elif scenario == 'few':
        num_fish = random.randint(2, 6)
    elif scenario == 'scattered':
        num_fish = random.randint(8, 15)
    elif scenario == 'grouped':
        num_fish = random.randint(15, 25)
    else:  # many
        num_fish = random.randint(25, 35)
    
    terrain_x = [p[0] for p in terrain_points]
    terrain_y = [p[1] for p in terrain_points]
    
    # Create fish groups for grouped scenario
    if scenario == 'grouped':
        group_centers = [random.randint(80, IMG_WIDTH - 80) for _ in range(random.randint(1, 3))]
        fish_positions = []
        for center in group_centers:
            group_size = num_fish // len(group_centers)
            for _ in range(group_size):
                fish_x = center + random.randint(-40, 40)
                fish_positions.append(max(15, min(IMG_WIDTH - 15, fish_x)))
    else:
        fish_positions = [random.randint(15, IMG_WIDTH - 15) for _ in range(num_fish)]
    
    for fish_x in fish_positions:
        terrain_height = np.interp(fish_x, terrain_x, terrain_y)
        # Position fish higher above terrain
        fish_y = terrain_height - random.randint(20, 120)
        
        arc_width = random.randint(6, 18)
        arc_height = random.randint(3, 8)
        
        # Create fish arch polygon
        top_points = []
        for t in np.linspace(0, math.pi, 8):
            x = fish_x + (arc_width/2) * math.cos(t)
            y = fish_y - arc_height * math.sin(t)
            top_points.append((int(x), int(y)))
        
        bottom_points = []
        for t in np.linspace(math.pi, 0, 8):
            x = fish_x + (arc_width/2) * math.cos(t)
            y = fish_y - (arc_height-2) * math.sin(t)
            bottom_points.append((int(x), int(y)))
        
        # Fill fish arch
        if len(top_points) > 1 and len(bottom_points) > 1:
            fish_polygon = top_points + bottom_points
            pygame.draw.polygon(surface, FISH_COLOR, fish_polygon)
            pygame.draw.lines(surface, EDGE_COLOR, False, top_points, 1)
            pygame.draw.lines(surface, EDGE_COLOR, False, bottom_points, 1)

def draw_large_arches(surface, terrain_points):
    """Draw larger arches with reduced frequency."""
    # Reduce frequency - sometimes no large arches
    if random.random() < 0.3:  # 30% chance
        num_arches = 1
    else:
        return
    
    terrain_x = [p[0] for p in terrain_points]
    terrain_y = [p[1] for p in terrain_points]
    
    arch_x = random.randint(80, IMG_WIDTH - 80)
    terrain_height = np.interp(arch_x, terrain_x, terrain_y)
    
    arch_width = random.randint(30, 50)
    arch_height = random.randint(15, 25)
    
    # Create large arch polygon
    top_points = []
    for t in np.linspace(0, math.pi, 12):
        x = arch_x + (arch_width/2) * math.cos(t)
        y = terrain_height - arch_height * math.sin(t)
        top_points.append((int(x), int(y)))
    
    bottom_points = []
    for t in np.linspace(math.pi, 0, 12):
        x = arch_x + (arch_width/2) * math.cos(t)
        y = terrain_height - (arch_height-4) * math.sin(t)
        bottom_points.append((int(x), int(y)))
    
    # Fill large arch
    if len(top_points) > 1 and len(bottom_points) > 1:
        arch_polygon = top_points + bottom_points
        pygame.draw.polygon(surface, FISH_COLOR, arch_polygon)
        pygame.draw.lines(surface, EDGE_COLOR, False, top_points, 2)
        pygame.draw.lines(surface, EDGE_COLOR, False, bottom_points, 2)

def generate_domain_A():
    """Generate synthetic edge image matching reference style."""
    surface.fill(BG_COLOR)
    
    terrain_points = draw_smooth_terrain_edge(surface)
    draw_large_arches(surface, terrain_points)
    draw_fish_arc_edges(surface, terrain_points)
    
    raw_str = pygame.image.tostring(surface, "RGB")
    image = Image.frombytes("RGB", (IMG_WIDTH, IMG_HEIGHT), raw_str)
    return image

# =======================================================
# MAIN LOOP
# =======================================================
for i in range(NUM_IMAGES):
    # Generate domain A
    domainA_img = generate_domain_A()

    # Convert to tensor and run Pix2Pix model
    with torch.no_grad():
        input_tensor = to_tensor(domainA_img).unsqueeze(0).to(DEVICE)
        fake_B = netG(input_tensor)
        output_img = to_pil(fake_B[0].cpu())

    # Save A and B
    a_path = os.path.join(SAVE_DIR, f"{i:05d}_A.png")
    b_path = os.path.join(SAVE_DIR, f"{i:05d}_B.png")
    pair_path = os.path.join(SAVE_DIR, f"{i:05d}_pair.png")

    domainA_img.save(a_path)
    output_img.save(b_path)

    # Combine A|B for visualization
    combined = Image.new("RGB", (IMG_WIDTH*2, IMG_HEIGHT))
    combined.paste(domainA_img, (0, 0))
    combined.paste(output_img, (IMG_WIDTH, 0))
    combined.save(pair_path)

    if i % 50 == 0:
        print(f"Generated {i}/{NUM_IMAGES} synthetic (A,B) pairs...")

print(f"✅ Finished generating {NUM_IMAGES} synthetic sonar pairs at: {SAVE_DIR}")
pygame.quit()
