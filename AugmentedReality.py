import cv2
import mediapipe as mp
import numpy as np
import math

class AugmentedReality():
    def __init__(self):
        self.height = 480
        self.width = 640

        self.button_width = 100
        self.button_height = 50
        self.button_offset = 40
        self.button_gap = 20
        self.button_colour_passive = (0, 255, 0)
        self.button_colour_active = (0, 0, 0)

        self.text_colour = (255, 255, 255)
        self.text_offset = 30

        self.mode = "Edit"

        self.font = cv2.FONT_HERSHEY_COMPLEX

        self.ESC_KEY = 27
        self.KEY_E = ord('e')
        self.KEY_E_UPPER = ord('E')
        self.KEY_I = ord('i')
        self.KEY_I_UPPER = ord('I')
        self.KEY_C = ord('c')
        self.KEY_C_UPPER = ord('C')
        self.KEY_U = ord('u')
        self.KEY_U_UPPER = ord('U')

        self.buttons = {
            "Edit": (
                    (self.width-self.button_width-self.button_offset, self.button_offset),
                    (self.width-self.button_offset, self.button_offset+self.button_height)
                    ),
            "Interact": 
                    (
                    (self.width-self.button_width-self.button_offset, self.button_offset+self.button_height+self.button_gap),
                    (self.width-self.button_offset, self.button_offset+2*self.button_height+self.button_gap)
                    )
        }

        self.drawing = False
        self.smoothness_val = 6
        self.temp_line = []
        self.current_obj_corners = []
        self.objects = []
        self.object_colour = (0, 0, 255)
        self.selected_colour = (255, 255, 255)

        self.pinch = None
        self.prev_angle = None
        self.move_multiplier = 1.2

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
    
    def draw_buttons(self):
        # cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
        for text, ((x1, y1), (x2, y2)) in self.buttons.items():
            cv2.rectangle(
                self.frame, 
                (x1, y1), 
                (x2, y2), 
                self.button_colour_passive, 
                -1 if self.mode == text else 1)
            cv2.putText(
                self.frame,
                text,
                (x1 + 10, y2 - 10), 
                self.font, 
                0.6,
                self.button_colour_active if self.mode == text else self.button_colour_passive, 
                2
            ) 

    def put_mode(self):
        cv2.putText(
            self.frame, f"Current Mode: {self.mode}", 
            (self.text_offset, self.height - self.text_offset),
            self.font, 
            0.7, 
            self.text_colour, 
            2
        )
        
    def draw_objects(self):
        for i, object_points in enumerate(self.objects):
            pts = np.array(object_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                self.frame, 
                [pts], 
                True, 
                self.object_colour if i != self.selected_obj else self.selected_colour, 
                2
            ) 

    def check_button_click(self, x, y):
        for name, ((x1, y1), (x2, y2)) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.mode = name
                print(f"Mode changed to: {self.mode}")
                return True
        return False

    def check_angle_difference(self):
        point_0 = self.temp_line[0]
        point_1 = self.temp_line[int(self.smoothness_val/2)-1]
        point_2 = self.temp_line[int(self.smoothness_val/2)]
        point_3 = self.temp_line[self.smoothness_val-1]

        dx = point_1[0] - point_0 [0]
        dy = point_1[1] - point_0 [1]

        angle_rad = math.atan2(-dy, dx)  # Negate dy because OpenCV y-axis increases downward
        old_angle_deg = math.degrees(angle_rad)

        dx = point_3[0] - point_2[0]
        dy = point_3[1] - point_2[1]

        angle_rad = math.atan2(-dy, dx)  # Negate dy because OpenCV y-axis increases downward
        new_angle_deg = math.degrees(angle_rad)

        # print(f"Current: {old_angle_deg}, new: {new_angle_deg}")
        diff = abs(old_angle_deg - new_angle_deg) % 360
        diff = min(diff, 360 - diff)
        return diff

    def keep_line(self, x, y):
        if(len(self.temp_line) != 0):
            if(self.temp_line[-1] == (x, y)):
                return

        self.temp_line.append((x,y))
        
        if(len(self.temp_line) > self.smoothness_val):
            self.temp_line.pop(0)
            if (self.check_angle_difference() > 30):
                self.temp_line = []
                self.current_obj_corners.append((x,y))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            is_button = self.check_button_click(x, y)
            if not is_button:
                self.drawing = True
                self.start_point = (x, y)
                self.current_obj_corners.append((x,y))
                # self.temp_line.append((x,y))
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_point = (x, y)
                self.keep_line(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.objects.append(self.current_obj_corners)
            # print(self.objects)
            self.drawing = False
            self.start_point = None
            self.temp_point = None
            self.temp_line = []
            self.current_obj_corners = []

    def wait_keyboard(self):
        key_num = cv2.waitKeyEx(1)
        match key_num:
            case self.ESC_KEY:
                return False
            case self.KEY_E| self.KEY_E_UPPER:
                self.mode = 'Edit'
            case self.KEY_I | self.KEY_I_UPPER:
                self.mode = 'Interact'
            case self.KEY_C | self.KEY_C_UPPER:
                self.objects = []
            case self.KEY_U | self.KEY_U_UPPER:
                if(len(self.objects) > 0):
                    self.objects.pop(-1)
        return True

    def selection_detection(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw hand landmarks
                # self.mp_drawing.draw_landmarks(self.frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extract coordinates (example: index finger tip)
                h, w, _ = self.frame.shape
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_base = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]

                self.tip_point = np.array([int(index_tip.x * w), int(index_tip.y * h)])
                base_point = np.array([int(index_base.x * w), int(index_base.y * h)])

                direction = self.tip_point - base_point
                direction = direction / np.linalg.norm(direction)
                length = 40
                self.end_point = self.tip_point + (direction * length).astype(int)

                dx = self.end_point[0] - self.tip_point[0]
                dy = self.end_point[1] - self.tip_point[1]
                angle_rad = math.atan2(-dy, dx)  # Negate dy because OpenCV y-axis increases downward
                self.direction_angle = math.degrees(angle_rad)

                # cv2.line(self.frame, tuple(self.tip_point), tuple(self.end_point), (255, 0, 255), 2)

    def move_object(self, move_x, move_y):
        # Rotation formula 
        # x' = cos(θ) * (x - cx) - sin(θ) * (y - cy) + cx 
        # y' = sin(θ) * (x - cx) + cos(θ) * (y - cy) + cy
        if self.selected_obj != -1:
            center = np.mean(self.objects[self.selected_obj], axis=0)
            print(f"Center: {center}")

            cos_a = math.cos(math.radians(self.pinch_direction_diff))
            sin_a = math.sin(math.radians(self.pinch_direction_diff))

            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a,  cos_a]
            ])

            translated = self.objects[self.selected_obj] - center
            rotated = translated.dot(rotation_matrix.T)
            self.objects[self.selected_obj] = rotated + center

            for i, corners in enumerate(self.objects[self.selected_obj]):
                self.objects[self.selected_obj][i] = (
                    corners[0] + move_x*self.move_multiplier, 
                    corners[1] + move_y*self.move_multiplier
                    )
        # cv2.putText(self.frame, "Pinch!", (self.tip_point[0], self.tip_point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def pinch_detection(self):
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                h, w, _ = self.frame.shape
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                distance = np.linalg.norm(self.tip_point - np.array([thumb_x, thumb_y]))

                if distance < 20:
                    self.pinch = True

                    move_x = thumb_x - self.prev_thumb_x
                    move_y = thumb_y - self.prev_thumb_y

                    if self.prev_angle is None:
                        self.prev_angle = self.direction_angle
                    diff = self.prev_angle - self.direction_angle 
                    self.pinch_direction_diff = min(diff, 360 - diff)
                    self.prev_angle = self.direction_angle
                    if(abs(self.pinch_direction_diff) < 1.0):
                        self.pinch_direction_diff = 0
                    print(f"Direction diff: {self.pinch_direction_diff}")

                    self.move_object(move_x, move_y)
                else:
                    self.pinch = False
                    self.prev_angle = None
                
                self.prev_thumb_x = thumb_x
                self.prev_thumb_y = thumb_y


    def check_collisions(self):
        if(self.pinch):
            return

        self.tip_selected, self.direction_selected = -1, -1
        
        if self.end_point is not None and self.tip_point is not None:
            for i, obj in enumerate(self.objects):
                contour = np.array(obj, dtype=np.int32).reshape((-1, 1, 2))

                # A bug, points should be int
                direction_result = cv2.pointPolygonTest(contour, (int(self.end_point[0]), int(self.end_point[1])), measureDist=False)
                tip_result = cv2.pointPolygonTest(contour, (int(self.tip_point[0]), int(self.tip_point[1])), measureDist=False)

                # did it this way so finger has the priority
                if tip_result >= 0:
                    self.tip_selected = i
                    continue      

                if direction_result >= 0:
                    self.direction_selected = i
        
        self.selected_obj = self.tip_selected if self.tip_selected != -1 else self.direction_selected

    def stabilize_objects(self):
        if(abs(self.avg_motion[0]) < 1):
            self.avg_motion[0] = 0
        if(abs(self.avg_motion[1]) < 1):
            self.avg_motion[1] = 0

        for i, obj in enumerate(self.objects):
            for j, corners in enumerate(obj):
                self.objects[i][j] = (corners[0] + self.avg_motion[0], corners[1] + self.avg_motion[1])

        # print(f"Camera shift: {self.avg_motion}")

    def mask_hand(self):
        # Step 1: Create full mask (1 = valid, 0 = masked out)
        self.mask = np.ones_like(self.gray, dtype=np.uint8)

        # Step 2: If hand detected, create mask on hand area
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                h, w = self.gray.shape
                points = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    points.append((x, y))
                cv2.fillPoly(self.mask, [np.array(points, dtype=np.int32)], 0)

    def detect_camera_movement(self):
        # New points for object stabilization
        if self.bg_pts is not None and self.old_pts is not None:
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(self.old_gray, self.gray, self.old_pts, None)

            if new_pts is not None:
                good_old = self.old_pts[status == 1]
                self.good_new = new_pts[status == 1]

                # Estimate average shift (camera motion)
                if len(good_old) > 0 and len(self.good_new) > 0:
                    motion_vectors = self.good_new - good_old
                    self.avg_motion = np.mean(motion_vectors, axis=0)
                else:
                    self.avg_motion = np.array([0.0, 0.0])  # Or None if you prefer
        else:
            self.old_pts = self.bg_pts


    def start(self):
        self.cap = cv2.VideoCapture(1)
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )


        if not self.cap.isOpened():
            print("Cannot open webcam")
            return

        cv2.namedWindow("AR App")
        cv2.setMouseCallback("AR App", self.mouse_callback)

        ret, old_frame = self.cap.read()

        # Find features to track camera movement
        old_frame = cv2.resize(old_frame, (self.width, self.height))
        self.old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        self.old_pts = cv2.goodFeaturesToTrack(self.old_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

        while self.wait_keyboard():
            self.tip_point = None
            self.end_point = None

            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Resize for consistency
            self.frame = cv2.resize(self.frame, (self.width, self.height))

            # Flip the image
            self.frame = cv2.flip(self.frame, 1)
            # RGB for Hand tracking gray for camera movement
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            self.results = self.hands.process(rgb)

            self.mask_hand()
            self.bg_pts = cv2.goodFeaturesToTrack(self.gray, mask=self.mask, maxCorners=100, qualityLevel=0.3, minDistance=7)

            self.detect_camera_movement()
            self.stabilize_objects()

            # Update old values
            self.old_gray = self.gray.copy()
            self.old_pts = self.good_new.reshape(-1, 1, 2)

            self.draw_buttons()
            self.put_mode()

            self.selection_detection()
            self.check_collisions()
            self.pinch_detection()

            self.draw_objects()

            cv2.imshow("AR App", self.frame)
        
        self.exit()
    
    def exit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        exit()

if __name__ == "__main__":
    ar = AugmentedReality()
    ar.start()