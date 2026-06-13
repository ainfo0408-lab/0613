import cv2
import mediapipe as mp
import time

def main():
    # 1. MediaPipe Hands 초기화
    mp_hands = mp.solutions.hands
    
    # max_num_hands=2: 최대 두 손까지 인식
    # min_detection_confidence: 감지 신뢰도 임계값
    # min_tracking_confidence: 추적 신뢰도 임계값
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # 2. 웹캠 캡처 초기화 (기본 카메라: 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: 웹캠을 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")
        return

    print("웹캠이 정상적으로 연결되었습니다. 'Q' 키를 누르면 종료됩니다.")

    # FPS 계산용 변수
    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("카메라 프레임을 읽어올 수 없습니다. 무시하고 계속 진행합니다.")
            continue

        # 거울 모드로 프레임 좌우 반전
        frame = cv2.flip(frame, 1)

        # 미디어파이프 처리를 위해 BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 성능 최적화를 위해 읽기 전용으로 설정
        rgb_frame.flags.writeable = False
        results = hands.process(rgb_frame)

        # 다시 그리기 가능하도록 설정
        rgb_frame.flags.writeable = True

        # 손이 감지된 경우 화면에 관절과 연결선 그리기
        hand_count = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                # 랜드마크를 이미지 상의 픽셀 좌표로 변환
                landmarks_px = []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_px.append((cx, cy))

                # 1. 손바닥 및 손등 부분 그리기 - 파란색 (Blue: 255, 0, 0)
                palm_connections = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)]
                for start, end in palm_connections:
                    cv2.line(frame, landmarks_px[start], landmarks_px[end], (255, 0, 0), 3)

                # 2. 손가락 마디(뼈대) 선 그리기 - 초록색 (Green: 0, 255, 0)
                finger_connections = [
                    (1, 2), (2, 3), (3, 4),      # 엄지
                    (5, 6), (6, 7), (7, 8),      # 검지
                    (9, 10), (10, 11), (11, 12), # 중지
                    (13, 14), (14, 15), (15, 16),# 약지
                    (17, 18), (18, 19), (19, 20) # 소지
                ]
                for start, end in finger_connections:
                    cv2.line(frame, landmarks_px[start], landmarks_px[end], (0, 255, 0), 3)

                # 3. 손가락 관절 부분 그리기 - 빨간 점 (Red: 0, 0, 255)
                for cx, cy in landmarks_px:
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), cv2.FILLED)

        # FPS 계산
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 화면에 FPS 및 정보 표시
        cv2.putText(
            frame, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            frame, 
            f"Hands Detected: {hand_count}", 
            (10, 70), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )

        # 결과 화면 출력
        cv2.imshow('Hand Tracking & AR Skeleton', frame)

        # 'Q' 또는 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()
