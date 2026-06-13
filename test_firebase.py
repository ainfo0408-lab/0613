import os
import sys
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def fetch_data():
    KEY_PATH = "serviceAccountKey.json"

    if not os.path.exists(KEY_PATH):
        print("오류: 'serviceAccountKey.json' 파일이 없습니다. 먼저 배치해 주세요.")
        sys.exit(1)

    # Firebase 초기화
    try:
        cred = credentials.Certificate(KEY_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
    except Exception as e:
        print(f"Firebase 연결 오류: {e}")
        sys.exit(1)

    print("Firestore에서 데이터를 가져옵니다...\n")

    try:
        # ksl_dictionary 컬렉션 내의 모든 문서 조회
        docs = db.collection("ksl_dictionary").stream()
        
        # 카테고리별로 정리를 위해 딕셔너리 준비
        categorized_data = {}
        
        for doc in docs:
            data = doc.to_dict()
            category = data.get("category", "미분류")
            word = data.get("word", "")
            description = data.get("description", "")
            hand_signs = data.get("hand_signs", [])
            
            if category not in categorized_data:
                categorized_data[category] = []
                
            categorized_data[category].append({
                "id": doc.id,
                "word": word,
                "description": description,
                "hand_signs": hand_signs
            })

        # 결과 예쁘게 출력
        if not categorized_data:
            print("데이터베이스에 저장된 수어 정보가 없습니다. seed_firebase.py를 먼저 실행해 주세요.")
            return

        for category, items in categorized_data.items():
            print(f"[카테고리: {category}]")
            print("=" * 60)
            for item in items:
                print(f"  - 단어명: {item['word']} (ID: {item['id']})")
                print(f"    - 설명: {item['description']}")
                print(f"    - 필요 수형: {', '.join(item['hand_signs'])}")
                print("-" * 60)
            print()

    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")

if __name__ == "__main__":
    fetch_data()
