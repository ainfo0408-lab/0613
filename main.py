import cv2
import mediapipe as mp
import time
import os
import sys
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from PIL import ImageFont, ImageDraw, Image

# 상태 정의
STATE_MENU = "MENU"
STATE_LEARNING = "LEARNING"
STATE_TRANSLATING = "TRANSLATING"

# 글로벌 변수
current_state = STATE_MENU
db = None
ksl_data = []
firebase_connected = False
selected_word_idx = 0
correct_start_time = None # 정답 유지 시작 시간
cap = None

# 한글 출력 헬퍼 함수
def draw_hangul_text(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 윈도우 기본 맑은 고딕 폰트 사용
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    if not os.path.exists(font_path):
        # 폰트가 없는 경우 맑은 고딕 굵은체 또는 기본 폰트 사용
        font_path = "C:\\Windows\\Fonts\\malgunbd.ttf"
        if not os.path.exists(font_path):
            font_path = "malgun.ttf"  # 현재 폴더 로컬 폰트 시도
            
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 텍스트 줄바꿈 함수
def wrap_text(text, limit=20):
    lines = []
    current_line = ""
    for char in text:
        current_line += char
        if len(current_line) >= limit:
            lines.append(current_line)
            current_line = ""
    if current_line:
        lines.append(current_line)
    return lines

# Firebase 데이터 연동 초기화
def init_firebase():
    global db, ksl_data, firebase_connected
    KEY_PATH = "serviceAccountKey.json"
    
    if not os.path.exists(KEY_PATH):
        print("Firebase 인증 키 'serviceAccountKey.json' 파일이 존재하지 않습니다.")
        return False
        
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(KEY_PATH)
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        
        # 데이터 조회
        docs = db.collection("ksl_dictionary").stream()
        ksl_data = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            ksl_data.append(data)
            
        # 단어 정렬 순서 정의 (안녕 -> 감사 -> 사랑 -> 나 -> 너 -> 1 -> 2 -> 3)
        order_map = {
            "ksl_hello": 0, "ksl_thankyou": 1, "ksl_iloveyou": 2,
            "ksl_me": 3, "ksl_you": 4,
            "ksl_one": 5, "ksl_two": 6, "ksl_three": 7
        }
        ksl_data.sort(key=lambda x: order_map.get(x["id"], 99))
        
        firebase_connected = len(ksl_data) > 0
        return firebase_connected
    except Exception as e:
        print(f"Firebase 연동 오류: {e}")
        return False

# 이미지 렌더링 헬퍼 함수
def draw_image_on_canvas(canvas, img_path, x, y, size=(400, 300)):
    if not os.path.exists(img_path):
        cv2.rectangle(canvas, (x, y), (x + size[0], y + size[1]), (60, 60, 60), -1)
        canvas = draw_hangul_text(canvas, "이미지 파일 없음", (x + 110, y + 130), 20, (255, 255, 255))
        return canvas
        
    img = cv2.imread(img_path)
    if img is not None:
        img_resized = cv2.resize(img, size)
        canvas[y:y+size[1], x:x+size[0]] = img_resized
    return canvas

# 손가락 개폐 상태 감지 함수
def get_finger_states(hand_landmarks):
    landmarks = hand_landmarks.landmark
    
    # 4개 손가락 (검지, 중지, 약지, 소지) 개폐 판정 (Tip.y < PIP.y 이면 Open)
    index_open = landmarks[8].y < landmarks[6].y
    middle_open = landmarks[12].y < landmarks[10].y
    ring_open = landmarks[16].y < landmarks[14].y
    pinky_open = landmarks[20].y < landmarks[18].y
    
    # 엄지 개폐 판정 (엄지 끝 4번과 기저 관절 2번의 수평 거리가 기준 이상 벌어지면 Open)
    thumb_open = abs(landmarks[4].x - landmarks[2].x) > 0.045
    
    return thumb_open, index_open, middle_open, ring_open, pinky_open

# 수형 판정 규칙 알고리즘
def check_gesture(word_id, multi_hand_landmarks):
    if not multi_hand_landmarks:
        return False
        
    hand_count = len(multi_hand_landmarks)
    
    # 1. 감사합니다 (ksl_thankyou)
    # 양손이 모두 감지되어야 하고, 양손의 거리(손목 포인트 0번 기준)가 가까워야 하며, 손가락들이 펴진 상태여야 함 (Tapping)
    if word_id == "ksl_thankyou":
        if hand_count < 2:
            return False
        h1 = multi_hand_landmarks[0].landmark
        h2 = multi_hand_landmarks[1].landmark
        # 두 손가락 기저/손목 거리 측정
        dist = np.sqrt((h1[0].x - h2[0].x)**2 + (h1[0].y - h2[0].y)**2)
        
        t1, i1, m1, r1, p1 = get_finger_states(multi_hand_landmarks[0])
        t2, i2, m2, r2, p2 = get_finger_states(multi_hand_landmarks[1])
        
        # 양손 다 완전히 웅크린 손이 아니고 펴진 상태이며 거리가 가까울 때
        both_open = (i1 and m1) and (i2 and m2)
        return both_open and dist < 0.22

    # 단일 손 판정 규칙 (2개 손 중 하나라도 규칙을 만족하면 정답)
    for hand_landmarks in multi_hand_landmarks:
        t, i, m, r, p = get_finger_states(hand_landmarks)
        h = hand_landmarks.landmark
        
        if word_id == "ksl_hello":  # 안녕하세요 (모든 손가락 펼침)
            if t and i and m and r and p:
                return True
                
        elif word_id == "ksl_iloveyou":  # 사랑합니다 (엄지, 검지, 소지 펼침 / 중지, 약지 접음)
            if t and i and not m and not r and p:
                return True
                
        elif word_id == "ksl_me":  # 나 (검지만 펼쳐 아래를 가리킴 / 중지, 약지, 소지 접음)
            pointing_down = h[8].y > h[6].y
            if pointing_down and not m and not r and not p:
                return True
                
        elif word_id == "ksl_you":  # 너 (검지만 펼쳐 위/앞을 가리킴 / 중지, 약지, 소지 접음)
            pointing_up = h[8].y < h[6].y
            if pointing_up and not m and not r and not p:
                return True
                
        elif word_id == "ksl_one":  # 숫자 1 (검지만 펼침, 나머지 접음)
            if i and not m and not r and not p:
                return True
                
        elif word_id == "ksl_two":  # 숫자 2 (검지, 중지 펼침, 나머지 접음)
            if i and m and not r and not p:
                return True
                
        elif word_id == "ksl_three":  # 숫자 3 (엄지, 검지, 중지 펼침, 나머지 접음)
            if t and i and m and not r and not p:
                return True
                
    return False

# 사용자 관절 그리기 함수
def draw_custom_skeleton(frame, hand_landmarks):
    h, w, _ = frame.shape
    landmarks_px = []
    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        landmarks_px.append((cx, cy))

    # 1. 손바닥 및 손등 부분 그리기 - 파란색
    palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
    for start, end in palm_connections:
        cv2.line(frame, landmarks_px[start], landmarks_px[end], (255, 0, 0), 3)

    # 2. 손가락 마디(뼈대) 선 그리기 - 초록색
    finger_connections = [
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10), (10, 11), (11, 12),
        (13, 14), (14, 15), (15, 16),
        (17, 18), (18, 19), (19, 20)
    ]
    for start, end in finger_connections:
        cv2.line(frame, landmarks_px[start], landmarks_px[end], (0, 255, 0), 3)

    # 3. 손가락 관절 부분 그리기 - 빨간 점
    for cx, cy in landmarks_px:
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), cv2.FILLED)

def main():
    global current_state, selected_word_idx, firebase_connected, ksl_data, correct_start_time, cap
    
    # 윈도우 크기 정의
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    cv2.namedWindow("Korean Sign Language Learning & Translation", cv2.WINDOW_AUTOSIZE)
    
    # Firebase 데이터 로드
    init_firebase()
    
    # 웹캠 상시 구동 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 감지할 수 없습니다. 카메라 설정을 확인해 주세요.")
        
    # 미디어파이프 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    while True:
        # 웹캠 프레임 항상 읽기 (버퍼 정체 방지)
        success, frame = False, None
        if cap is not None and cap.isOpened():
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1) # 좌우 반전
                
        # 기본 백그라운드 캔버스 생성 (Charcoal Dark)
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[:, :] = (30, 30, 30)
        
        # ----------------- 1. 메인 메뉴 상태 -----------------
        if current_state == STATE_MENU:
            cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 120), (45, 45, 45), -1)
            canvas = draw_hangul_text(canvas, "한국수어 학습 및 실시간 번역 프로그램", (320, 35), 36, (255, 255, 255))
            
            # 수어 학습 메뉴 카드
            card1_x1, card1_y1 = 150, 200
            card1_x2, card1_y2 = 550, 520
            cv2.rectangle(canvas, (card1_x1, card1_y1), (card1_x2, card1_y2), (60, 100, 60), -1)
            cv2.rectangle(canvas, (card1_x1, card1_y1), (card1_x2, card1_y2), (100, 255, 100), 3)
            canvas = draw_hangul_text(canvas, "1. 수어 학습 모드", (card1_x1 + 60, card1_y1 + 80), 30, (255, 255, 255))
            canvas = draw_hangul_text(canvas, "수형 이미지와 카메라 연동을", (card1_x1 + 40, card1_y1 + 160), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "통해 실시간으로 동작 피드백을", (card1_x1 + 40, card1_y1 + 200), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "받으며 수어를 학습합니다.", (card1_x1 + 40, card1_y1 + 240), 18, (200, 200, 200))
            
            # 실시간 번역 메뉴 카드
            card2_x1, card2_y1 = 730, 200
            card2_x2, card2_y2 = 1130, 520
            cv2.rectangle(canvas, (card2_x1, card2_y1), (card2_x2, card2_y2), (100, 60, 60), -1)
            cv2.rectangle(canvas, (card2_x1, card2_y1), (card2_x2, card2_y2), (255, 100, 100), 3)
            canvas = draw_hangul_text(canvas, "2. 실시간 번역 모드", (card2_x1 + 60, card2_y1 + 80), 30, (255, 255, 255))
            canvas = draw_hangul_text(canvas, "카메라로 입력되는 손 모양을", (card2_x1 + 40, card2_y1 + 160), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "실시간으로 인식해 수어로", (card2_x1 + 40, card2_y1 + 200), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "번역해 줍니다 (준비중)", (card2_x1 + 40, card2_y1 + 240), 18, (255, 150, 150))
            
            canvas = draw_hangul_text(canvas, "키보드의 숫자 키 [1] 또는 [2]를 누르세요.  (종료: [Q])", (350, 600), 22, (255, 255, 100))
            
        # ----------------- 2. 수어 학습 상태 -----------------
        elif current_state == STATE_LEARNING:
            if not firebase_connected:
                canvas = draw_hangul_text(canvas, "오류: Firebase 수어 데이터를 불러오지 못했습니다.", (100, 250), 28, (100, 100, 255))
                canvas = draw_hangul_text(canvas, "serviceAccountKey.json 키가 올바른지 확인해 주세요.", (100, 320), 22, (200, 200, 200))
                canvas = draw_hangul_text(canvas, "[M] 키를 눌러 메인 메뉴로 이동", (100, 420), 20, (255, 255, 100))
            else:
                # 2.1 좌측 영역 (단어 목록)
                cv2.rectangle(canvas, (0, 0), (350, WINDOW_HEIGHT), (40, 40, 40), -1)
                cv2.line(canvas, (350, 0), (350, WINDOW_HEIGHT), (100, 100, 100), 2)
                
                canvas = draw_hangul_text(canvas, "수어 단어 목록", (25, 30), 24, (255, 255, 255))
                canvas = draw_hangul_text(canvas, "단어 선택: 숫자 키 (1~8)", (25, 75), 14, (180, 180, 180))
                
                for idx, item in enumerate(ksl_data):
                    y_pos = 130 + idx * 55
                    if idx == selected_word_idx:
                        cv2.rectangle(canvas, (15, y_pos - 10), (335, y_pos + 35), (80, 120, 80), -1)
                        cv2.rectangle(canvas, (15, y_pos - 10), (335, y_pos + 35), (100, 255, 100), 1)
                        color = (255, 255, 255)
                    else:
                        color = (160, 160, 160)
                    canvas = draw_hangul_text(canvas, f"[{idx + 1}] {item['word']}", (30, y_pos), 18, color)
                
                canvas = draw_hangul_text(canvas, "[M] 메인 메뉴로 이동", (25, 660), 16, (255, 255, 100))
                
                # 2.2 중앙 영역 (상세 설명 및 안내 이미지)
                curr_word = ksl_data[selected_word_idx]
                canvas = draw_hangul_text(canvas, f"단어: {curr_word['word']}", (375, 30), 30, (255, 255, 255))
                canvas = draw_hangul_text(canvas, f"카테고리: {curr_word['category']}", (375, 80), 16, (150, 255, 150))
                
                # 설명글 출력
                canvas = draw_hangul_text(canvas, "동작 설명:", (375, 120), 18, (200, 200, 200))
                desc_lines = wrap_text(curr_word['description'], limit=21)
                for line_idx, line in enumerate(desc_lines):
                    canvas = draw_hangul_text(canvas, line, (375, 155 + line_idx * 28), 16, (230, 230, 230))
                    
                # 안내 이미지 그리기
                img_name = curr_word['id'] + ".png"
                img_path = os.path.join("images", img_name)
                canvas = draw_hangul_text(canvas, "수형 가이드 이미지:", (375, 345), 18, (200, 200, 200))
                canvas = draw_image_on_canvas(canvas, img_path, 375, 385, size=(390, 290))
                
                # 2.3 우측 영역 (웹캠 피드 및 판정)
                canvas = draw_hangul_text(canvas, "내 카메라 피드:", (800, 30), 20, (255, 255, 255))
                
                # 웹캠 데이터 처리 및 그리기
                is_correct = False
                cam_feed_resized = np.zeros((315, 420, 3), dtype=np.uint8)
                
                if success and frame is not None:
                    # 미디어파이프 처리를 위해 RGB 변환
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False
                    results = hands.process(rgb_frame)
                    rgb_frame.flags.writeable = True
                    
                    # 손 감지 및 골격 그리기
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            draw_custom_skeleton(frame, hand_landmarks)
                            
                        # 수형 매칭 판단 실행
                        is_correct = check_gesture(curr_word['id'], results.multi_hand_landmarks)
                    
                    # 피드 크기 조정 (420x315)
                    cam_feed_resized = cv2.resize(frame, (420, 315))
                    
                # 웹캠 화면 캔버스에 붙이기
                canvas[80:80+315, 800:800+420] = cam_feed_resized
                cv2.rectangle(canvas, (800, 80), (1220, 395), (150, 150, 150), 2) # 테두리
                
                # 판정 결과 영역 렌더링
                canvas = draw_hangul_text(canvas, "판정 결과:", (800, 420), 18, (200, 200, 200))
                
                # 정답 판단 박스
                box_x1, box_y1 = 800, 455
                box_x2, box_y2 = 1220, 675
                
                if is_correct:
                    # 정답 유지 시간 체크
                    if correct_start_time is None:
                        correct_start_time = time.time()
                    
                    elapsed = time.time() - correct_start_time
                    
                    # 밝은 초록색 카드
                    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (40, 120, 40), -1)
                    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 255, 100), 2)
                    canvas = draw_hangul_text(canvas, "★ 정답 ★", (box_x1 + 140, box_y1 + 30), 28, (255, 255, 255))
                    canvas = draw_hangul_text(canvas, "참 잘했습니다! 동작 유지 중...", (box_x1 + 75, box_y1 + 80), 18, (230, 255, 230))
                    
                    # 다음 단계로 넘어가기 위한 게이지 바 그리기
                    bar_w = int(min(elapsed / 1.5, 1.0) * 360)
                    cv2.rectangle(canvas, (box_x1 + 30, box_y2 - 35), (box_x1 + 390, box_y2 - 20), (60, 80, 60), -1)
                    cv2.rectangle(canvas, (box_x1 + 30, box_y2 - 35), (box_x1 + 30 + bar_w, box_y2 - 20), (255, 255, 255), -1)
                    
                    # 1.5초 이상 유지 시 자동 다음 단어 전환
                    if elapsed >= 1.5:
                        selected_word_idx = (selected_word_idx + 1) % len(ksl_data)
                        correct_start_time = None
                else:
                    # 정답이 아닐 경우 유지시간 타이머 리셋
                    correct_start_time = None
                    
                    # 오렌지색 카드 (연습 유도)
                    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (45, 75, 180), -1) # BGR
                    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (100, 150, 255), 2)
                    canvas = draw_hangul_text(canvas, "다시 시도해주세요", (box_x1 + 105, box_y1 + 45), 26, (255, 255, 255))
                    canvas = draw_hangul_text(canvas, "왼쪽 가이드를 보고 수형을 맞춰보세요.", (box_x1 + 50, box_y1 + 110), 16, (210, 210, 210))
                    
        # ----------------- 3. 실시간 수어 번역 상태 (Placeholder) -----------------
        elif current_state == STATE_TRANSLATING:
            cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 120), (45, 45, 45), -1)
            canvas = draw_hangul_text(canvas, "실시간 수어 번역 모드 (준비 중)", (360, 35), 36, (100, 100, 255))
            
            canvas = draw_hangul_text(canvas, "본 기능은 다음 개발 단계에서 구현될 예정입니다.", (150, 250), 26, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "현재 학습 모드를 완벽히 완성하는 것에 중점을 두고 있습니다.", (150, 320), 20, (160, 160, 160))
            canvas = draw_hangul_text(canvas, "기존 손 인식 랜드마크 및 뼈대 오버레이 기능과 연동하여", (150, 380), 20, (160, 160, 160))
            canvas = draw_hangul_text(canvas, "실시간 수어 번역이 제공될 것입니다.", (150, 430), 20, (160, 160, 160))
            
            canvas = draw_hangul_text(canvas, "[M] 또는 [ESC] 키를 누르면 메인 메뉴로 돌아갑니다.", (150, 560), 22, (255, 255, 100))
            
        # 전체 캔버스 화면 출력
        cv2.imshow("Korean Sign Language Learning & Translation", canvas)
        
        # 키 입력 대기
        key = cv2.waitKey(20) & 0xFF
        
        # 'q' 또는 'Q' 누르면 프로그램 종료
        if key == ord('q') or key == ord('Q'):
            break
            
        # 상태 전환 조작
        if current_state == STATE_MENU:
            if key == ord('1'):
                if not firebase_connected:
                    init_firebase()
                current_state = STATE_LEARNING
                selected_word_idx = 0
                correct_start_time = None
            elif key == ord('2'):
                current_state = STATE_TRANSLATING
                
        elif current_state == STATE_LEARNING:
            if key == ord('m') or key == ord('M') or key == 27: # 'M' 또는 ESC
                current_state = STATE_MENU
                correct_start_time = None
            # 1~8 숫자 키 선택
            elif ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                if idx < len(ksl_data):
                    selected_word_idx = idx
                    correct_start_time = None
                    
        elif current_state == STATE_TRANSLATING:
            if key == ord('m') or key == ord('M') or key == 27: # 'M' 또는 ESC
                current_state = STATE_MENU
                
    # 리소스 정리
    if cap is not None and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
