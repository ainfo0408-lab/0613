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
def draw_image_on_canvas(canvas, img_path, x, y, size=(400, 400)):
    if not os.path.exists(img_path):
        cv2.rectangle(canvas, (x, y), (x + size[0], y + size[1]), (60, 60, 60), -1)
        canvas = draw_hangul_text(canvas, "이미지 파일 없음", (x + 130, y + 185), 20, (255, 255, 255))
        return canvas
        
    img = cv2.imread(img_path)
    if img is not None:
        img_resized = cv2.resize(img, size)
        canvas[y:y+size[1], x:x+size[0]] = img_resized
    return canvas

def main():
    global current_state, selected_word_idx, firebase_connected, ksl_data
    
    # 윈도우 크기 정의
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    cv2.namedWindow("Korean Sign Language Learning & Translation", cv2.WINDOW_AUTOSIZE)
    
    # Firebase 데이터로드 시도
    init_firebase()
    
    while True:
        # 기본 캔버스 생성 (어두운 백그라운드)
        canvas = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        canvas[:, :] = (30, 30, 30) # Dark Charcoal
        
        # 1. 메인 메뉴 상태
        if current_state == STATE_MENU:
            # 타이틀 영역
            cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 120), (45, 45, 45), -1)
            canvas = draw_hangul_text(canvas, "한국수어 학습 및 실시간 번역 프로그램", (320, 35), 36, (255, 255, 255))
            
            # 메뉴 카드 디자인 (수어 학습)
            card1_x1, card1_y1 = 150, 200
            card1_x2, card1_y2 = 550, 520
            cv2.rectangle(canvas, (card1_x1, card1_y1), (card1_x2, card1_y2), (60, 100, 60), -1)
            cv2.rectangle(canvas, (card1_x1, card1_y1), (card1_x2, card1_y2), (100, 255, 100), 3) # 테두리
            canvas = draw_hangul_text(canvas, "1. 수어 학습 모드", (card1_x1 + 60, card1_y1 + 80), 30, (255, 255, 255))
            canvas = draw_hangul_text(canvas, "Firebase 수어 사전을 토대로", (card1_x1 + 40, card1_y1 + 160), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "수형 이미지와 설명을 보며", (card1_x1 + 40, card1_y1 + 200), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "수어를 쉽게 배웁니다.", (card1_x1 + 40, card1_y1 + 240), 18, (200, 200, 200))
            
            # 메뉴 카드 디자인 (실시간 번역)
            card2_x1, card2_y1 = 730, 200
            card2_x2, card2_y2 = 1130, 520
            cv2.rectangle(canvas, (card2_x1, card2_y1), (card2_x2, card2_y2), (100, 60, 60), -1)
            cv2.rectangle(canvas, (card2_x1, card2_y1), (card2_x2, card2_y2), (255, 100, 100), 3) # 테두리
            canvas = draw_hangul_text(canvas, "2. 실시간 번역 모드", (card2_x1 + 60, card2_y1 + 80), 30, (255, 255, 255))
            canvas = draw_hangul_text(canvas, "카메라로 입력되는 손 모양을", (card2_x1 + 40, card2_y1 + 160), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "실시간으로 인식해 수어로", (card2_x1 + 40, card2_y1 + 200), 18, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "번역해 줍니다 (준비중)", (card2_x1 + 40, card2_y1 + 240), 18, (255, 150, 150))
            
            # 하단 안내 메시지
            canvas = draw_hangul_text(canvas, "키보드의 숫자 키 [1] 또는 [2]를 누르세요.  (종료: [Q])", (350, 600), 22, (255, 255, 100))
            
        # 2. 수어 학습 상태
        elif current_state == STATE_LEARNING:
            if not firebase_connected:
                # 연결 실패 화면
                canvas = draw_hangul_text(canvas, "오류: Firebase 수어 데이터를 불러오지 못했습니다.", (100, 250), 28, (100, 100, 255))
                canvas = draw_hangul_text(canvas, "serviceAccountKey.json 키가 올바른지 확인해 주세요.", (100, 320), 22, (200, 200, 200))
                canvas = draw_hangul_text(canvas, "[M] 키를 눌러 메인 메뉴로 이동", (100, 420), 20, (255, 255, 100))
            else:
                # 좌측 영역: 단어 목록 구성
                cv2.rectangle(canvas, (0, 0), (450, WINDOW_HEIGHT), (40, 40, 40), -1)
                cv2.line(canvas, (450, 0), (450, WINDOW_HEIGHT), (100, 100, 100), 2)
                
                canvas = draw_hangul_text(canvas, "수어 단어 목록", (30, 30), 26, (255, 255, 255))
                canvas = draw_hangul_text(canvas, "배울 단어의 숫자 키를 누르세요.", (30, 80), 16, (200, 200, 200))
                
                for idx, item in enumerate(ksl_data):
                    y_pos = 140 + idx * 60
                    # 선택된 단어 하이라이트
                    if idx == selected_word_idx:
                        cv2.rectangle(canvas, (20, y_pos - 10), (430, y_pos + 40), (80, 120, 80), -1)
                        cv2.rectangle(canvas, (20, y_pos - 10), (430, y_pos + 40), (100, 255, 100), 1)
                        color = (255, 255, 255)
                    else:
                        color = (180, 180, 180)
                        
                    canvas = draw_hangul_text(
                        canvas, 
                        f"[{idx + 1}] {item['word']} ({item['category']})", 
                        (40, y_pos), 
                        20, 
                        color
                    )
                
                canvas = draw_hangul_text(canvas, "[M] 메인 메뉴로 돌아가기", (30, 650), 18, (255, 255, 100))
                
                # 우측 영역: 학습 단어 상세 정보 표시
                curr_word = ksl_data[selected_word_idx]
                canvas = draw_hangul_text(canvas, f"단어명: {curr_word['word']}", (500, 40), 36, (255, 255, 255))
                canvas = draw_hangul_text(canvas, f"분류: {curr_word['category']}", (500, 95), 18, (150, 255, 150))
                
                # 설명글 출력 (줄바꿈 처리)
                canvas = draw_hangul_text(canvas, "수어 동작 설명:", (500, 140), 20, (200, 200, 200))
                desc_lines = wrap_text(curr_word['description'], limit=32)
                for line_idx, line in enumerate(desc_lines):
                    canvas = draw_hangul_text(canvas, line, (500, 180 + line_idx * 30), 18, (230, 230, 230))
                    
                # 이미지 파일명 도출 및 렌더링
                img_name = curr_word['id'] + ".png"
                img_path = os.path.join("images", img_name)
                
                # 동작 예시 이미지 영역 렌더링 (하단 우측 배치)
                canvas = draw_hangul_text(canvas, "동작 시각 가이드:", (500, 360), 20, (200, 200, 200))
                canvas = draw_image_on_canvas(canvas, img_path, 500, 400, size=(280, 280))
                
        # 3. 실시간 수어 번역 상태 (Placeholder)
        elif current_state == STATE_TRANSLATING:
            cv2.rectangle(canvas, (0, 0), (WINDOW_WIDTH, 120), (45, 45, 45), -1)
            canvas = draw_hangul_text(canvas, "실시간 수어 번역 모드 (준비 중)", (360, 35), 36, (100, 100, 255))
            
            canvas = draw_hangul_text(canvas, "본 기능은 다음 개발 단계에서 구현될 예정입니다.", (150, 250), 26, (200, 200, 200))
            canvas = draw_hangul_text(canvas, "현재 학습 모드를 완벽히 완성하는 것에 중점을 두고 있습니다.", (150, 320), 20, (160, 160, 160))
            canvas = draw_hangul_text(canvas, "기존 손 인식 랜드마크 및 뼈대 오버레이 기능과 연동하여", (150, 380), 20, (160, 160, 160))
            canvas = draw_hangul_text(canvas, "실시간 수어 번역이 제공될 것입니다.", (150, 430), 20, (160, 160, 160))
            
            canvas = draw_hangul_text(canvas, "[M] 또는 [ESC] 키를 누르면 메인 메뉴로 돌아갑니다.", (150, 560), 22, (255, 255, 100))
            
        # 화면 출력
        cv2.imshow("Korean Sign Language Learning & Translation", canvas)
        
        # 키 입력 대기
        key = cv2.waitKey(30) & 0xFF
        
        # 'q' 또는 'Q' 누르면 프로그램 종료 (메뉴 상태에서만 작동하도록 안전장치 또는 전체 적용)
        if key == ord('q') or key == ord('Q'):
            break
            
        # 상태 전환 조작
        if current_state == STATE_MENU:
            if key == ord('1'):
                # 학습 모드로 진입하기 전에 한번 더 데이터 로드 시도 (실패 시 복구용)
                if not firebase_connected:
                    init_firebase()
                current_state = STATE_LEARNING
                selected_word_idx = 0
            elif key == ord('2'):
                current_state = STATE_TRANSLATING
                
        elif current_state == STATE_LEARNING:
            if key == ord('m') or key == ord('M') or key == 27: # 'M' 또는 ESC
                current_state = STATE_MENU
            # 1~8 숫자 키 선택
            elif ord('1') <= key <= ord('8'):
                idx = key - ord('1')
                if idx < len(ksl_data):
                    selected_word_idx = idx
                    
        elif current_state == STATE_TRANSLATING:
            if key == ord('m') or key == ord('M') or key == 27: # 'M' 또는 ESC
                current_state = STATE_MENU
                
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
