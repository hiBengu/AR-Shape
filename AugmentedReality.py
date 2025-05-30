import cv2
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
        for object_points in self.objects:
            pts = np.array(object_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                self.frame, 
                [pts], 
                True, 
                self.object_colour, 
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
                print(self.current_obj_corners)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            is_button = self.check_button_click(x, y)
            if not is_button:
                self.drawing = True
                self.start_point = (x, y)
                self.current_obj_corners.append((x,y))
                print(self.current_obj_corners)
                self.keep_line(x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.temp_point = (x, y)
                self.keep_line(x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                self.objects.append(self.current_obj_corners)
            print(self.objects)
            self.drawing = False
            self.start_point = None
            self.temp_point = None
            self.temp_line = []
            self.current_obj_corners = []

    def waitKeyboard(self):
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

    def start(self):
        self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("Cannot open webcam")
            return

        cv2.namedWindow("AR App")
        cv2.setMouseCallback("AR App", self.mouse_callback)

        while self.waitKeyboard():
            ret, self.frame = self.cap.read()
            if not ret:
                break

            # Resize for consistency
            self.frame = cv2.resize(self.frame, (self.width, self.height))

            self.draw_buttons()
            self.put_mode()
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