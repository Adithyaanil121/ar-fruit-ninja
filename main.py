import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import deque

# --- 1. CONFIGURATION ---
WIDTH, HEIGHT = 1280, 720 
GRAVITY = 0.4
GAME_DURATION = 30
HIT_RADIUS = 13
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
img_bomb = load_asset('assets/bomb.png', scale=0.8)
img_explosion = load_asset('assets/explosion.png', scale=1.0)
img_ice = load_asset('assets/ice_bottle.png', scale=0.8)
img_blitz = load_asset('assets/colored_bomb.png', scale=0.8)
img_gold = load_asset('assets/golden_pineapple.png', scale=0.8)

for fruit in FRUIT_LIST:
    raw_half_1 = load_asset(f'assets/{fruit}_half_1.png', scale=0.8)
    raw_half_2 = load_asset(f'assets/{fruit}_half_2.png', scale=0.8)
    fruit_assets[fruit] = {
        'whole': load_asset(f'assets/{fruit}.png', scale=0.8),
        'half_1': dim_image(raw_half_1, factor=0.6),
        'half_2': dim_image(raw_half_2, factor=0.6),
        'splash': load_asset(f'assets/{fruit}_splash.png', scale=0.8)
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
HALF_WIDTH = WIDTH // 2

# Mouse
mouse_pos = (0, 0)
mouse_clicked = False
def mouse_callback(event, x, y, flags, param):
    global mouse_pos, mouse_clicked
    if event == cv2.EVENT_MOUSEMOVE: mouse_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN: mouse_clicked = True

cv2.namedWindow("AR Fruit Ninja")
cv2.setMouseCallback("AR Fruit Ninja", mouse_callback)

# Init Players
p1 = PlayerState(1, "P1", 0, HALF_WIDTH, COLOR_CYAN)
p2 = PlayerState(2, "P2", HALF_WIDTH, WIDTH, COLOR_GOLD)
players = [p1, p2]

# Shared Game Vars
state = "MENU"
game_start_time = 0
game_duration_mod = 0 
scores_saved = False 
game_counter = 1     
leaderboard = []

btn_start = Button(WIDTH//2 - 150, HEIGHT//2 - 50, 300, 100, "START", COLOR_GREEN)
btn_restart = Button(WIDTH//2 - 150, HEIGHT - 150, 300, 100, "RESTART", COLOR_CYAN)

def start_new_game():
    """Resets everything to start a fresh game properly."""
    global game_start_time, game_duration_mod, scores_saved, state
    p1.reset()
    p2.reset()
    game_start_time = time.time()
    game_duration_mod = 0
    scores_saved = False # IMPORTANT: Reset this so new scores can be saved next time
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

def draw_leaderboard_multi(frame):
    overlay = frame.copy()
    # Draw a dark box in the bottom center
    cv2.rectangle(overlay, (WIDTH//2 - 250, 350), (WIDTH//2 + 250, 600), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, "TOP 5 SCORES", (WIDTH//2 - 120, 390), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_YELLOW, 2)
    
    # Sort and Display
    sorted_scores = sorted(leaderboard, key=lambda x: x['score'], reverse=True)[:5]
    
    for i, entry in enumerate(sorted_scores):
        text = f"#{i+1} {entry['label']} : {entry['score']}"
        cv2.putText(frame, text, (WIDTH//2 - 180, 430 + (i*35)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)

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
    p1.finger_pos = None
    p2.finger_pos = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            x_norm = hand_landmarks.landmark[0].x
            target_player = None
            if x_norm < 0.5: target_player = p1
            else: target_player = p2
            
            hx, hy = int(hand_landmarks.landmark[8].x * WIDTH), int(hand_landmarks.landmark[8].y * HEIGHT)
            target_player.finger_pos = (hx, hy)
            target_player.blade_points.append((hx, hy))
    else:
        p1.blade_points.clear()
        p2.blade_points.clear()

    # --- MENU STATE ---
    if state == "MENU":
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.putText(frame, "MULTIPLAYER FRUIT NINJA", (WIDTH//2 - 400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 4)
        cv2.line(frame, (HALF_WIDTH, 250), (HALF_WIDTH, HEIGHT-200), COLOR_WHITE, 5)
        cv2.putText(frame, "PLAYER 1", (WIDTH//4 - 100, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_CYAN, 2)
        cv2.putText(frame, "PLAYER 2", (3*WIDTH//4 - 100, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_GOLD, 2)
        
        btn_start.draw(frame)
        if btn_start.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
            start_new_game()

    # --- PLAYING STATE ---
    elif state == "PLAYING":
        current_time = time.time()
        time_elapsed = current_time - game_start_time
        time_remaining = max(0, int((GAME_DURATION + game_duration_mod) - time_elapsed))
        
        cv2.line(frame, (HALF_WIDTH, 0), (HALF_WIDTH, HEIGHT), COLOR_BLACK, 10)

        for p in players:
            if not p.is_alive and time_remaining > 0: 
                cv2.putText(frame, "OUT", (p.x_min + 200, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 3, COLOR_RED, 5)
                continue

            # Blitz Visuals
            is_blitz = current_time < p.blitz_end_time
            if is_blitz:
                overlay = frame.copy()
                cv2.rectangle(overlay, (p.x_min, 0), (p.x_max, HEIGHT), COLOR_PURPLE, -1)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Spawning Logic
            spawn_rate = 0.2 if is_blitz else 0.8
            if current_time - p.last_spawn > spawn_rate:
                p.last_spawn = current_time
                
                sx = random.randint(p.x_min + 50, p.x_max - 50)
                sy = HEIGHT + 50
                vx = random.randint(-5, 5) 
                vy = random.randint(-MAX_SPEED, -MIN_SPEED)
                
                obj_type = "fruit"
                f_name = random.choice(FRUIT_LIST)
                img = fruit_assets[f_name]['whole']
                rand_val = random.random()

                forced = False
                if time_remaining < 60 and not p.spawned_gold and rand_val < 0.2:
                    obj_type, img, f_name = "gold", img_gold, None; p.spawned_gold = True; forced = True
                elif time_remaining < 40 and not p.spawned_blitz and rand_val < 0.2:
                    obj_type, img, f_name = "blitz", img_blitz, None; p.spawned_blitz = True; forced = True
                elif time_remaining < 20 and not p.spawned_ice and rand_val < 0.2:
                    obj_type, img, f_name = "ice", img_ice, None; p.spawned_ice = True; forced = True

                if not forced:
                    if rand_val < 0.1: obj_type, img, f_name = "bomb", img_bomb, None
                    elif rand_val < 0.15: 
                        r = random.random()
                        if r < 0.33: obj_type, img, f_name = "ice", img_ice, None
                        elif r < 0.66: obj_type, img, f_name = "blitz", img_blitz, None
                        else: obj_type, img, f_name = "gold", img_gold, None
                
                obj = GameObject(img, sx, sy, vx, vy, obj_type, f_name)
                if obj_type != "fruit": obj.angular_velocity = 15
                p.objects.append(obj)

            # Updates & Collision
            for obj in p.objects: obj.update()
            p.objects = [o for o in p.objects if o.active]
            for part in p.particles: part.update()
            p.particles = [pt for pt in p.particles if pt.active]

            for obj in p.objects:
                if not obj.active: continue
                hit = False
                hit_source = "finger"
                
                if p.finger_pos:
                    dist = np.linalg.norm(np.array(p.finger_pos) - np.array([obj.x, obj.y]))
                    if dist < (obj.radius + HIT_RADIUS): hit = True
                
                if is_blitz and obj.y < HEIGHT - 50 and obj.type != "bomb":
                    hit = True; hit_source = "blitz"
                    center_x = p.x_min + (HALF_WIDTH // 2)
                    cv2.line(frame, (center_x, HEIGHT//2), (int(obj.x), int(obj.y)), COLOR_PURPLE, 3)

                if hit:
                    obj.active = False
                    create_slice_particles(p, obj)
                    
                    if obj.type == "fruit": p.score += 10
                    elif obj.type == "bomb":
                        if hit_source == "finger":
                            p.lives -= 1
                            p.particles.append(Particle(img_explosion, obj.x, obj.y, 0,0, 20, False, False, 0.05))
                            if p.lives <= 0: p.is_alive = False
                    elif obj.type == "ice":
                        game_duration_mod += 5
                        p.popup_text = "+5s TIME!"; p.popup_timer = 30
                    elif obj.type == "gold":
                        p.score += 100
                        p.popup_text = "+100 PTS!"; p.popup_timer = 30
                    elif obj.type == "blitz":
                        p.blitz_end_time = time.time() + BLITZ_DURATION
                        p.popup_text = "BLITZ!"; p.popup_timer = 30

            # Drawing
            for obj in p.objects: obj.draw(frame)
            for part in p.particles: part.draw(frame)
            
            if len(p.blade_points) > 1:
                pts = np.array(p.blade_points, np.int32)
                cv2.polylines(frame, [pts], False, p.ui_color, 4)

            ui_x = p.x_min + 20
            cv2.putText(frame, f"{p.name}: {p.score}", (ui_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, p.ui_color, 2)
            cv2.putText(frame, "<3 " * p.lives, (ui_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
            
            if p.popup_timer > 0:
                cv2.putText(frame, p.popup_text, (p.x_min + 100, HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 3)
                p.popup_timer -= 1

        mins, secs = divmod(time_remaining, 60)
        cv2.rectangle(frame, (WIDTH//2 - 60, 10), (WIDTH//2 + 60, 60), COLOR_BLACK, -1)
        cv2.putText(frame, f"{mins:02}:{secs:02}", (WIDTH//2 - 45, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 2)

        if time_remaining == 0 or (not p1.is_alive and not p2.is_alive):
            state = "GAMEOVER"

    # --- GAMEOVER STATE ---
    elif state == "GAMEOVER":
        
        # 1. Save Scores (Once per session)
        if not scores_saved:
            leaderboard.append({'label': f"Game {game_counter} (P1)", 'score': p1.score})
            leaderboard.append({'label': f"Game {game_counter} (P2)", 'score': p2.score})
            game_counter += 1
            scores_saved = True # Lock saving until restart triggers reset

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        winner_text = "DRAW!"
        color = COLOR_WHITE
        if p1.score > p2.score:
            winner_text = "PLAYER 1 WINS!"
            color = COLOR_CYAN
        elif p2.score > p1.score:
            winner_text = "PLAYER 2 WINS!"
            color = COLOR_GOLD
            
        cv2.putText(frame, winner_text, (WIDTH//2 - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        cv2.putText(frame, f"P1: {p1.score}", (WIDTH//4, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_CYAN, 3)
        cv2.putText(frame, f"P2: {p2.score}", (3*WIDTH//4 - 100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GOLD, 3)
        
        # 2. Draw Leaderboard
        draw_leaderboard_multi(frame)
        
        btn_restart.draw(frame)
        if btn_restart.is_clicked(mouse_pos[0], mouse_pos[1], current_click):
            start_new_game()

    if p1.finger_pos: cv2.circle(frame, p1.finger_pos, 10, COLOR_CYAN, -1)
    if p2.finger_pos: cv2.circle(frame, p2.finger_pos, 10, COLOR_GOLD, -1)

    cv2.imshow("AR Fruit Ninja", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
