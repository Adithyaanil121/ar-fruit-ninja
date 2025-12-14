import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque
from network_manager import NetworkManager

# --- 1. CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720 
GRAVITY = 0.4
GAME_DURATION = 90
HIT_RADIUS = 30 # Increased for easier hits
MIN_SPEED = 16 
MAX_SPEED = 24 
HAND_LOCK_THRESHOLD = 150 

# Power Up Config
BLITZ_DURATION = 2    

# Colors
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GOLD = (0, 215, 255)
COLOR_PURPLE = (255, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_PINK = (147, 20, 255)

FRUIT_LIST = ['watermelon', 'apple', 'orange', 'mango', 'banana', 'pomegranate']

# --- 2. ASSET LOADING & HELPERS ---
def load_asset(path, scale=1.0):
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        if scale != 1.0:
            new_w = int(img.shape[1] * scale)
            new_h = int(img.shape[0] * scale)
            img = cv2.resize(img, (new_w, new_h))
        return img
    except:
        return None

def dim_image(image, factor=0.6):
    if image is None: return None
    b, g, r, a = cv2.split(image)
    return cv2.merge((
        np.clip(b.astype(np.float32) * factor, 0, 255).astype(np.uint8),
        np.clip(g.astype(np.float32) * factor, 0, 255).astype(np.uint8),
        np.clip(r.astype(np.float32) * factor, 0, 255).astype(np.uint8),
        a
    ))

def rotate_image(image, angle):
    if image is None: return None
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

def overlay_transparent(background, overlay, x, y, alpha_master=1.0):
    if overlay is None: return
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape
    x, y = int(x - ol_w / 2), int(y - ol_h / 2)
    y1, y2 = max(0, y), min(bg_h, y + ol_h)
    x1, x2 = max(0, x), min(bg_w, x + ol_w)
    y1o, y2o = max(0, -y), min(ol_h, bg_h - y)
    x1o, x2o = max(0, -x), min(ol_w, bg_w - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o: return
    
    alpha_s = (overlay[y1o:y2o, x1o:x2o, 3] / 255.0) * alpha_master
    alpha_l = 1.0 - alpha_s
    for c in range(3):
        background[y1:y2, x1:x2, c] = (alpha_s * overlay[y1o:y2o, x1o:x2o, c] +
                                       alpha_l * background[y1:y2, x1:x2, c])

# --- ASSET LOADING ---
fruit_assets = {}
img_bomb = load_asset('bomb.png', scale=0.8)
img_explosion = load_asset('explosion.png', scale=1.0)
img_ice = load_asset('ice_bottle.png', scale=0.8)
img_blitz = load_asset('colored_bomb.png', scale=0.8)
img_gold = load_asset('golden_pineapple.png', scale=0.8)

for fruit in FRUIT_LIST:
    raw_half_1 = load_asset(f'{fruit}_half_1.png', scale=0.8)
    raw_half_2 = load_asset(f'{fruit}_half_2.png', scale=0.8)
    fruit_assets[fruit] = {
        'whole': load_asset(f'{fruit}.png', scale=0.8),
        'half_1': dim_image(raw_half_1, factor=0.6),
        'half_2': dim_image(raw_half_2, factor=0.6),
        'splash': load_asset(f'{fruit}_splash.png', scale=0.8)
    }

# --- 3. CLASS DEFINITIONS ---

class Button:
    def __init__(self, x, y, w, h, text, color=COLOR_GREEN):
        self.rect = (x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50))
        self.is_hovered = False

    def draw(self, frame):
        x, y, w, h = self.rect
        c = self.hover_color if self.is_hovered else self.color
        cv2.rectangle(frame, (x, y), (x + w, y + h), c, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_WHITE, 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.text, font, 1.5, 3)[0]
        text_x = x + (w - text_size[0]) // 2
        text_y = y + (h + text_size[1]) // 2
        cv2.putText(frame, self.text, (text_x, text_y), font, 1.5, COLOR_BLACK, 3)

    def is_clicked(self, mouse_x, mouse_y, clicked_state):
        x, y, w, h = self.rect
        if x < mouse_x < x + w and y < mouse_y < y + h:
            self.is_hovered = True
            if clicked_state: return True
        else:
            self.is_hovered = False
        return False

class GameObject:
    def __init__(self, img, x, y, vx, vy, obj_type, fruit_name=None):
        self.img = img
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.type = obj_type 
        self.fruit_name = fruit_name
        self.active = True
        self.alpha = 1.0
        self.angle = 0
        self.angular_velocity = random.uniform(-10, 10)
        
        if img is not None:
            self.width, self.height = img.shape[1], img.shape[0]
            self.radius = max(self.width, self.height) // 2
        else:
            self.radius = 40

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += GRAVITY 
        self.angle += self.angular_velocity
        if self.y > HEIGHT + 100: self.active = False

    def draw(self, frame):
        if not self.active: return
        if self.img is not None:
            rotated_img = rotate_image(self.img, self.angle)
            overlay_transparent(frame, rotated_img, self.x, self.y, self.alpha)
        else:
            c = COLOR_GREEN
            if self.type == "bomb": c = COLOR_BLACK
            elif self.type == "ice": c = COLOR_CYAN
            elif self.type == "blitz": c = COLOR_PURPLE
            elif self.type == "gold": c = COLOR_GOLD
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, c, -1)
            cv2.circle(frame, (int(self.x), int(self.y)), self.radius, COLOR_WHITE, 2)
            label = self.type.upper() if self.type != "fruit" else ""
            if label:
                cv2.putText(frame, label, (int(self.x)-30, int(self.y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 2)

class Particle(GameObject):
    def __init__(self, img, x, y, vx, vy, lifetime=20, apply_gravity=True, spin=False, fade_speed=0.0):
        super().__init__(img, x, y, vx, vy, "particle")
        self.lifetime = lifetime
        self.age = 0
        self.apply_gravity = apply_gravity
        self.fade_speed = fade_speed
        if not spin: self.angular_velocity = 0

    def update(self):
        if self.apply_gravity: super().update()
        else:
            self.x += self.vx
            self.y += self.vy
        self.age += 1
        if self.lifetime and self.age > self.lifetime: self.active = False
        if self.fade_speed > 0:
            self.alpha -= self.fade_speed
            if self.alpha <= 0: self.active = False

# --- 4. PLAYER STATE CLASS ---
class PlayerState:
    def __init__(self, id, name, x_min, x_max, color):
        self.id = id
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.ui_color = color
        
        # Game Stats
        self.score = 0
        self.lives = 3
        self.is_alive = True
        
        # Power Ups
        self.blitz_end_time = 0
        self.popup_text = ""
        self.popup_timer = 0
        
        # Objects
        self.objects = []
        self.particles = []
        self.blade_points = deque(maxlen=15)
        self.finger_pos = None
        
        # Timers / Drops
        self.last_spawn = time.time()
        self.spawned_gold = False
        self.spawned_ice = False
        self.spawned_blitz = False

    def reset(self):
        self.score = 0
        self.lives = 3
        self.is_alive = True
        self.blitz_end_time = 0
        self.objects.clear()
        self.particles.clear()
        self.blade_points.clear()
        self.spawned_gold = False
        self.spawned_ice = False
        self.spawned_blitz = False
        self.last_spawn = time.time()

# --- 5. SETUP & CONFIGURATION ---

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mouse
mouse_pos = (0, 0)
mouse_clicked = False
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, mouse_clicked
    if event == cv2.EVENT_MOUSEMOVE: mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN: mouse_clicked = True

cv2.namedWindow("AR Fruit Ninja 4P")
cv2.setMouseCallback("AR Fruit Ninja 4P", mouse_callback)

# Init Players (4 Players split across width)
# P1: 0-25%, P2: 25-50%, P3: 50-75%, P4: 75-100%
q_width = WIDTH // 4
p1 = PlayerState(1, "P1", 0, q_width, COLOR_CYAN)
p2 = PlayerState(2, "P2", q_width, q_width*2, COLOR_GOLD)
p3 = PlayerState(3, "P3", q_width*2, q_width*3, COLOR_ORANGE)
p4 = PlayerState(4, "P4", q_width*3, WIDTH, COLOR_PINK)
players = [p1, p2, p3, p4]

# Shared Game Vars
state = "SETUP" # New initial state
game_start_time = 0
game_duration_mod = 0 
scores_saved = False 
game_counter = 1     
leaderboard = []
network = None
is_host = False

btn_host = Button(WIDTH//2 - 350, HEIGHT//2 - 50, 300, 100, "HOST GAME", COLOR_GREEN)
btn_client = Button(WIDTH//2 + 50, HEIGHT//2 - 50, 300, 100, "JOIN GAME", COLOR_CYAN)
btn_start = Button(WIDTH//2 - 150, HEIGHT//2 - 50, 300, 100, "START", COLOR_GREEN)
btn_restart = Button(WIDTH//2 - 150, HEIGHT - 150, 300, 100, "RESTART", COLOR_CYAN)

# Input for IP
input_ip = "localhost"
input_active = False

def start_new_game():
    global game_start_time, game_duration_mod, scores_saved, state
    for p in players: p.reset()
    game_start_time = time.time()
    game_duration_mod = 0
    scores_saved = False 
    state = "PLAYING"

def create_slice_particles(player, obj):
    downward_push = random.uniform(8, 15)
    
    if obj.type == "fruit":
        f_name = obj.fruit_name
        half1 = fruit_assets[f_name]['half_1']
        half2 = fruit_assets[f_name]['half_2']
        splash = fruit_assets[f_name]['splash']
        
        if splash is not None:
             player.particles.append(Particle(splash, obj.x, obj.y, obj.vx*0.5, 0, lifetime=None, apply_gravity=True, spin=False, fade_speed=0.05))
        if half1 is not None:
            p = Particle(half1, obj.x, obj.y, obj.vx + random.uniform(-8, -3), max(0, obj.vy) + downward_push, None, True, True, 0.01)
            p.angular_velocity = -20
            player.particles.append(p)
        if half2 is not None:
            p = Particle(half2, obj.x, obj.y, obj.vx + random.uniform(3, 8), max(0, obj.vy) + downward_push, None, True, True, 0.01)
            p.angular_velocity = 20
            player.particles.append(p)
    
    elif obj.type in ["ice", "blitz", "gold"]:
        color_img = img_explosion 
        if obj.type == "ice": color_img = dim_image(img_explosion, 0.8) 
        if color_img is not None:
             player.particles.append(Particle(color_img, obj.x, obj.y, 0, 0, 20, False, False, 0.1))

# --- 6. MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    current_click = mouse_clicked
    mouse_clicked = False 
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # --- HAND ASSIGNMENT ---
    # Local hands only
    local_hands = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hx, hy = int(hand_landmarks.landmark[8].x * WIDTH), int(hand_landmarks.landmark[8].y * HEIGHT)
            local_hands.append((hx, hy))

    # Assign local hands to players based on role
    # Host: P1 (Left), P2 (Right)
    # Client: P3 (Left), P4 (Right)
    
    my_p1 = p1 if is_host else p3
    my_p2 = p2 if is_host else p4
    
    my_p1.finger_pos = None
    my_p2.finger_pos = None
    
    # Simple assignment: Leftmost hand -> my_p1, Rightmost -> my_p2
    if len(local_hands) == 1:
        # If only one hand, assign based on screen side relative to center? 
        # Or just assign to P1/P3 for now.
        # Better: Assign to the player whose zone it is in?
        # Let's just assign to P1/P3 if x < WIDTH/2, else P2/P4
        hx, hy = local_hands[0]
        if hx < WIDTH // 2: my_p1.finger_pos = (hx, hy)
        else: my_p2.finger_pos = (hx, hy)
    elif len(local_hands) >= 2:
        # Sort by X
        local_hands.sort(key=lambda p: p[0])
        my_p1.finger_pos = local_hands[0]
        my_p2.finger_pos = local_hands[1]

    # --- NETWORK SYNC ---
    if network and network.connected:
        if is_host:
            # HOST: Receive P3/P4 inputs
            remote_data = network.get_latest_data()
            if remote_data:
                p3.finger_pos = remote_data.get('p3_pos')
                p4.finger_pos = remote_data.get('p4_pos')
            
            # HOST: Send Game State
            # We need to serialize objects and particles efficiently
            # For now, let's just send basic stats and let client simulate visuals if possible?
            # No, for sync, Host must dictate objects.
            # Simplified: Send object list (type, x, y, id).
            
            # Actually, sending full object state for 4 players might be too much for 60fps pickle.
            # Let's try sending just the essential data.
            
            # Pack data
            host_data = {
                'p1_pos': p1.finger_pos,
                'p2_pos': p2.finger_pos,
                'p1_score': p1.score, 'p2_score': p2.score,
                'p3_score': p3.score, 'p4_score': p4.score,
                'p1_lives': p1.lives, 'p2_lives': p2.lives,
                'p3_lives': p3.lives, 'p4_lives': p4.lives,
                'time': time.time(),
                'state': state
            }
            
            # Objects: We need to sync them. 
            # Let's send a simplified representation of objects for rendering
            # This is the heavy part.
            all_objects_data = []
            for p in players:
                p_objs = []
                for obj in p.objects:
                    p_objs.append({
                        'type': obj.type, 'x': obj.x, 'y': obj.y, 
                        'vx': obj.vx, 'vy': obj.vy, 'angle': obj.angle,
                        'fruit_name': obj.fruit_name
                    })
                all_objects_data.append(p_objs)
            
            host_data['objects'] = all_objects_data
            network.send_data(host_data)
            
        else:
            # CLIENT: Send P3/P4 inputs
            client_data = {
                'p3_pos': p3.finger_pos,
                'p4_pos': p4.finger_pos
            }
            network.send_data(client_data)
            
            # CLIENT: Receive Game State
            host_data = network.get_latest_data()
            if host_data:
                p1.finger_pos = host_data.get('p1_pos')
                p2.finger_pos = host_data.get('p2_pos')
                p1.score = host_data.get('p1_score', 0)
                p2.score = host_data.get('p2_score', 0)
                p3.score = host_data.get('p3_score', 0)
                p4.score = host_data.get('p4_score', 0)
                p1.lives = host_data.get('p1_lives', 3)
                p2.lives = host_data.get('p2_lives', 3)
                p3.lives = host_data.get('p3_lives', 3)
                p4.lives = host_data.get('p4_lives', 3)
                state = host_data.get('state', state)
                
                # Sync Objects (Recreate them on client for rendering)
                # This is a bit naive (recreating every frame), but ensures sync.
                # Optimization: Only update positions if ID matches? 
                # For now, clear and rebuild is easiest but might flicker.
                # Let's try to just update the list.
                if 'objects' in host_data:
                    for i, p_objs_data in enumerate(host_data['objects']):
                        players[i].objects.clear() # Clear old
                        for o_data in p_objs_data:
                            # Recreate object (visual only)
                            img = None
                            if o_data['type'] == 'fruit':
                                img = fruit_assets[o_data['fruit_name']]['whole']
                            elif o_data['type'] == 'bomb': img = img_bomb
                            elif o_data['type'] == 'ice': img = img_ice
                            elif o_data['type'] == 'blitz': img = img_blitz
                            elif o_data['type'] == 'gold': img = img_gold
                            
                            obj = GameObject(img, o_data['x'], o_data['y'], o_data['vx'], o_data['vy'], o_data['type'], o_data.get('fruit_name'))
                            obj.angle = o_data['angle']
                            players[i].objects.append(obj)

    # --- SETUP STATE ---
    if state == "SETUP":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        cv2.putText(frame, "LAN MULTIPLAYER SETUP", (WIDTH//2 - 300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_WHITE, 3)
        
        btn_host.draw(frame)
        btn_client.draw(frame)
        
        # IP Input for Client
        if not is_host:
            cv2.putText(frame, f"Host IP: {input_ip}", (WIDTH//2 - 150, HEIGHT//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)
            cv2.putText(frame, "(Type to edit)", (WIDTH//2 - 100, HEIGHT//2 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

        if btn_host.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
            is_host = True
            network = NetworkManager(is_server=True)
            # Start server in thread to not freeze UI
            threading.Thread(target=network.start_server, daemon=True).start()
            state = "WAITING"
            
        if btn_client.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
            is_host = False
            network = NetworkManager(is_server=False, server_ip=input_ip)
            if network.connect_to_server():
                state = "MENU" # Client goes to menu, waits for Host to start
            else:
                print("Failed to connect")

        # Handle IP typing (simple)
        # This requires cv2.waitKey to capture keys, which is at end of loop.
        # We'll handle it there.

    elif state == "WAITING":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        cv2.putText(frame, "WAITING FOR CLIENT...", (WIDTH//2 - 250, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YELLOW, 2)
        if network and network.connected:
            state = "MENU"

    # --- MENU STATE ---
    elif state == "MENU":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "4-PLAYER FRUIT NINJA", (WIDTH//2 - 350, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 4)
        
        # Draw Player Zones
        for i in range(1, 4):
            x = i * (WIDTH // 4)
            cv2.line(frame, (x, 250), (x, HEIGHT-200), COLOR_WHITE, 2)
            
        cv2.putText(frame, "P1", (WIDTH//8 - 20, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CYAN, 2)
        cv2.putText(frame, "P2", (3*WIDTH//8 - 20, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GOLD, 2)
        cv2.putText(frame, "P3", (5*WIDTH//8 - 20, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_ORANGE, 2)
        cv2.putText(frame, "P4", (7*WIDTH//8 - 20, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_PINK, 2)
        
        if is_host:
            btn_start.draw(frame)
            if btn_start.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
                start_new_game()
        else:
            cv2.putText(frame, "Waiting for Host to start...", (WIDTH//2 - 200, HEIGHT - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)

    # --- PLAYING STATE ---
    elif state == "PLAYING":
        current_time = time.time()
        
        # Only Host manages time and spawning
        if is_host:
            time_elapsed = current_time - game_start_time
            time_remaining = max(0, int((GAME_DURATION + game_duration_mod) - time_elapsed))
        else:
            # Client estimates or receives time? 
            # For now, let's rely on Host sending state, but we need time_remaining for UI
            # We can send time_remaining in packet.
            # Simplified: Client just renders what it has.
            time_remaining = 90 # Placeholder, should be synced
        
        # Draw Dividers
        for i in range(1, 4):
            cv2.line(frame, (i * (WIDTH // 4), 0), (i * (WIDTH // 4), HEIGHT), COLOR_BLACK, 2)

        # Update & Logic (HOST ONLY)
        if is_host:
            for p in players:
                if not p.is_alive: continue

                # Spawning Logic
                spawn_rate = 0.8 # Simplified
                if current_time - p.last_spawn > spawn_rate:
                    p.last_spawn = current_time
                    sx = random.randint(p.x_min + 20, p.x_max - 20)
                    sy = HEIGHT + 50
                    vx = random.randint(-3, 3) 
                    vy = random.randint(-MAX_SPEED, -MIN_SPEED)
                    
                    obj_type = "fruit"
                    f_name = random.choice(FRUIT_LIST)
                    img = fruit_assets[f_name]['whole']
                    
                    # Random drops logic (simplified for brevity)
                    if random.random() < 0.1: obj_type, img, f_name = "bomb", img_bomb, None
                    
                    obj = GameObject(img, sx, sy, vx, vy, obj_type, f_name)
                    if obj_type != "fruit": obj.angular_velocity = 15
                    p.objects.append(obj)

                # Updates & Collision
                for obj in p.objects: obj.update()
                p.objects = [o for o in p.objects if o.active]
                
                # Collision Check
                for obj in p.objects:
                    if not obj.active: continue
                    hit = False
                    if p.finger_pos:
                        dist = np.linalg.norm(np.array(p.finger_pos) - np.array([obj.x, obj.y]))
                        if dist < (obj.radius + HIT_RADIUS): hit = True
                    
                    if hit:
                        obj.active = False
                        # create_slice_particles(p, obj) # Host creates particles? 
                        # Particles are visual, maybe client creates them too?
                        # For now, no particles over network to save bandwidth.
                        
                        if obj.type == "fruit": p.score += 10
                        elif obj.type == "bomb":
                            p.lives -= 1
                            if p.lives <= 0: p.is_alive = False

        # Drawing (BOTH)
        for p in players:
            if not p.is_alive:
                cv2.putText(frame, "OUT", (p.x_min + 20, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 3)
                continue
                
            for obj in p.objects: obj.draw(frame)
            
            # Draw Blade
            if p.finger_pos:
                p.blade_points.append(p.finger_pos)
            
            if len(p.blade_points) > 1:
                pts = np.array(p.blade_points, np.int32)
                cv2.polylines(frame, [pts], False, p.ui_color, 4)

            # UI
            ui_x = p.x_min + 10
            cv2.putText(frame, f"{p.score}", (ui_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, p.ui_color, 2)
            cv2.putText(frame, "<3 " * p.lives, (ui_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

    # --- GAMEOVER STATE ---
    elif state == "GAMEOVER":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, "GAME OVER", (WIDTH//2 - 200, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_WHITE, 4)
        
        if is_host:
            btn_restart.draw(frame)
            if btn_restart.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
                start_new_game()

    # Draw Cursors
    if p1.finger_pos: cv2.circle(frame, p1.finger_pos, 10, p1.ui_color, -1)
    if p2.finger_pos: cv2.circle(frame, p2.finger_pos, 10, p2.ui_color, -1)
    if p3.finger_pos: cv2.circle(frame, p3.finger_pos, 10, p3.ui_color, -1)
    if p4.finger_pos: cv2.circle(frame, p4.finger_pos, 10, p4.ui_color, -1)

    cv2.imshow("AR Fruit Ninja 4P", frame)
    
    # Key Handling
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break # ESC
    elif state == "SETUP" and not is_host:
        # Simple IP typing
        if key == 8: input_ip = input_ip[:-1] # Backspace
        elif key != 255: input_ip += chr(key)

cap.release()
cv2.destroyAllWindows()