import os
import sys
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def seed_data():
    KEY_PATH = "serviceAccountKey.json"

    # 1. 인증 JSON 파일이 존재하는지 확인
    if not os.path.exists(KEY_PATH):
        print("=" * 60)
        print("오류: 'serviceAccountKey.json' 파일을 찾을 수 없습니다!")
        print("Firebase Console에서 생성한 비공개 키 JSON 파일을 현재 폴더에 복사해 주세요.")
        print("경로 예시: c:\\Users\\user\\Desktop\\teacher_hackerton\\0613\\serviceAccountKey.json")
        print("=" * 60)
        sys.exit(1)

    print("Firebase 인증 키 발견! Firestore 연결을 시도합니다...")

    # 2. Firebase Admin SDK 초기화
    try:
        cred = credentials.Certificate(KEY_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase 연결 성공!")
    except Exception as e:
        print(f"Firebase 초기화 오류: {e}")
        sys.exit(1)

    # 3. 적재할 한국수어 샘플 데이터정의
    # 수형 정보(손모양 매핑용) 및 동작 설명 포함
    ksl_samples = [
        # 인사 카테고리
        {
            "id": "ksl_hello",
            "word": "안녕하세요",
            "category": "인사",
            "description": "오른손 주먹을 쥐고 가슴 앞으로 가져온 후 아래로 내리며 손바닥을 폅니다. (또는 두 손을 모아 가볍게 아래로 숙임)",
            "hand_signs": ["fist_chest", "open_palm_down"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733917/visual_video.mp4"
        },
        {
            "id": "ksl_thankyou",
            "word": "감사합니다",
            "category": "인사",
            "description": "왼손 손등을 오른손 바닥으로 가볍게 두 번 두드립니다.",
            "hand_signs": ["left_hand_flat", "right_hand_tap_twice"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733918/visual_video.mp4"
        },
        {
            "id": "ksl_iloveyou",
            "word": "사랑합니다",
            "category": "인사",
            "description": "오른손의 엄지, 검지, 새끼손가락을 펴고(I Love You 수형) 가슴 앞으로 내밉니다.",
            "hand_signs": ["thumb_index_pinky_extended"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733919/visual_video.mp4"
        },
        # 대명사 카테고리
        {
            "id": "ksl_me",
            "word": "나",
            "category": "대명사",
            "description": "오른손 검지손가락으로 자신의 가슴을 가리킵니다.",
            "hand_signs": ["index_pointing_chest"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733920/visual_video.mp4"
        },
        {
            "id": "ksl_you",
            "word": "너",
            "category": "대명사",
            "description": "오른손 검지손가락으로 상대방(앞방향)을 가리킵니다.",
            "hand_signs": ["index_pointing_forward"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733921/visual_video.mp4"
        },
        # 숫자 카테고리
        {
            "id": "ksl_one",
            "word": "일(1)",
            "category": "숫자",
            "description": "오른손 검지만 위로 폅니다.",
            "hand_signs": ["index_finger_up"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733922/visual_video.mp4"
        },
        {
            "id": "ksl_two",
            "word": "이(2)",
            "category": "숫자",
            "description": "오른손 검지와 중지를 위로 폅니다.",
            "hand_signs": ["index_middle_finger_up"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733923/visual_video.mp4"
        },
        {
            "id": "ksl_three",
            "word": "삼(3)",
            "category": "숫자",
            "description": "오른손 엄지, 검지, 중지를 폅니다.",
            "hand_signs": ["thumb_index_middle_finger_up"],
            "video_url": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20200820/733924/visual_video.mp4"
        }
    ]

    # 4. Firestore에 데이터 적재
    print("Firestore에 데이터 적재를 시작합니다...")
    collection_ref = db.collection("ksl_dictionary")

    for sample in ksl_samples:
        doc_id = sample["id"]
        doc_data = {
            "word": sample["word"],
            "category": sample["category"],
            "description": sample["description"],
            "hand_signs": sample["hand_signs"],
            "video_url": sample["video_url"]
        }
        
        # set()을 사용해 문서를 덮어쓰거나 새로 생성
        collection_ref.document(doc_id).set(doc_data)
        print(f"적재 완료: [{sample['category']}] {sample['word']} (문서 ID: {doc_id})")

    print("\n모든 샘플 데이터가 성공적으로 Firestore에 등록되었습니다!")

if __name__ == "__main__":
    seed_data()
