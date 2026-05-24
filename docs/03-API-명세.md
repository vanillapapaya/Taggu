# API 명세

## 베이스 URL
```
개발: http://localhost:8000/api/v1
운영: https://api.yoink.app/v1
```

## 인증
- JWT 기반
- 검색/피드 조회: 인증 불필요 (비로그인 허용)
- 좋아요/업로드/프로필: 인증 필요
- NSFW 콘텐츠: 성인인증 추가 필요 (장기)

---

## 1. 검색 API

### `GET /search`
자연어 텍스트로 이미지 검색

**Query Parameters**:
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| q | string | Y | 검색 쿼리 ("뒤돌아보면서 웃는 남자") |
| limit | int | N | 결과 수 (기본 20, 최대 50) |
| offset | int | N | 페이지네이션 오프셋 |
| type | string | N | 콘텐츠 타입 필터: meme, illustration, all (기본 all) |
| source | string | N | 소스 필터: twitter, safebooru, manual, all (기본 all) |

**Response** (`200 OK`):
```json
{
    "results": [
        {
            "id": "uuid",
            "thumbnail_url": "https://r2.yoink.app/thumbnails/...",
            "source_url": "https://twitter.com/...",
            "source": "twitter",
            "tags": ["meme", "reaction"],
            "caption": "뒤를 돌아보며 웃고 있는 남성",
            "score": 0.87,
            "creator": {
                "id": "uuid",
                "display_name": "작가명",
                "twitter_handle": "@handle"
            },
            "like_count": 42,
            "is_liked": false
        }
    ],
    "total": 150,
    "has_more": true
}
```

**검색 내부 로직**:
```
1. 쿼리 텍스트 → CLIP 텍스트 인코더 → clip_query_vector
2. 쿼리 텍스트 → e5-large 인코더 → text_query_vector
3. Qdrant clip_vectors 검색 (clip_query_vector, top 50) → results_a
4. Qdrant text_vectors 검색 (text_query_vector, top 50) → results_b
5. 가중 합산: final_score = 0.4 * clip_score + 0.6 * text_score
6. 중복 제거, 상위 limit개 반환
```

---

## 2. 피드 API

### `GET /feed`
개인화 추천 피드

**Query Parameters**:
| 파라미터 | 타입 | 필수 | 설명 |
|---------|------|------|------|
| limit | int | N | 결과 수 (기본 20) |
| cursor | string | N | 페이지네이션 커서 |

**Response** (`200 OK`):
```json
{
    "items": [
        {
            "type": "image",
            "data": {
                "id": "uuid",
                "thumbnail_url": "...",
                "source_url": "...",
                "tags": ["meme"],
                "creator": { ... },
                "like_count": 42
            }
        },
        {
            "type": "ad",
            "data": {
                "ad_id": "...",
                "ad_unit": "interstitial",
                "label": "광고"
            }
        }
    ],
    "next_cursor": "abc123"
}
```

**피드 내부 로직**:
```
비로그인 사용자:
  → 인기순 (like_count 기반) + 최신순 혼합

로그인 사용자:
  1. user_interactions에서 최근 좋아요/검색 이미지 30개 추출
  2. 해당 이미지들의 CLIP 벡터 평균 → user_profile_vector
  3. Qdrant 검색 (user_profile_vector, filter: NOT IN 이미 본 이미지)
  4. 추천 70% + 랜덤 30% 혼합
  5. 5개마다 광고 슬롯 삽입
```

### `GET /feed/trending`
인기 트렌딩 피드 (비개인화)

**Query Parameters**: limit, cursor

---

## 3. 이미지 API

### `GET /images/{id}`
이미지 상세 정보

**Response** (`200 OK`):
```json
{
    "id": "uuid",
    "thumbnail_url": "...",
    "source_url": "...",
    "source": "twitter",
    "tags": ["meme", "reaction", "funny"],
    "caption": "...",
    "creator": {
        "id": "uuid",
        "display_name": "...",
        "twitter_handle": "...",
        "is_verified": true,
        "profile_image": "..."
    },
    "like_count": 42,
    "view_count": 1200,
    "is_liked": false,
    "similar_images": [
        { "id": "uuid", "thumbnail_url": "...", "score": 0.82 }
    ]
}
```

### `POST /images/upload`
이미지 업로드 (인증 필요)

**Request** (multipart/form-data):
| 필드 | 타입 | 필수 | 설명 |
|-----|------|------|------|
| image | file | Y* | 이미지 파일 (최대 10MB) |
| image_url | string | Y* | 또는 이미지 URL (*둘 중 하나) |
| description | string | N | 이미지 설명 |
| tags | string[] | N | 태그 |
| source_url | string | N | 원본 출처 URL |

**Response** (`201 Created`):
```json
{
    "id": "uuid",
    "status": "processing",
    "message": "이미지가 업로드되었습니다. 처리 완료까지 약 30초 소요됩니다."
}
```

---

## 4. 인터랙션 API

### `POST /images/{id}/like`
좋아요 토글 (인증 필요)

**Response** (`200 OK`):
```json
{
    "is_liked": true,
    "like_count": 43
}
```

### `POST /interactions/view`
조회 기록 (피드 추천용, 배치 전송)

**Request**:
```json
{
    "views": [
        { "image_id": "uuid", "duration_ms": 2500 },
        { "image_id": "uuid", "duration_ms": 800 }
    ]
}
```

---

## 5. 유저 API

### `POST /auth/login`
OAuth 로그인 (Google, Twitter)

### `GET /users/me`
내 프로필 조회

### `GET /users/me/likes`
내 좋아요 목록

---

## 6. 크리에이터 API

### `GET /creators/{id}`
크리에이터 프로필 조회

**Response**:
```json
{
    "id": "uuid",
    "display_name": "작가명",
    "bio": "일러스트레이터",
    "twitter_handle": "@handle",
    "pixiv_id": "12345678",
    "is_verified": true,
    "image_count": 48,
    "total_likes": 1250,
    "images": [ ... ]
}
```

### `POST /creators/claim`
크리에이터 본인 인증 요청 (OAuth 연동)

---

## 7. 관리자 API

### `POST /admin/collect`
수집 작업 트리거

### `GET /admin/stats`
서비스 통계 (이미지 수, DAU, 검색 수 등)

### `DELETE /admin/images/{id}`
이미지 삭제 (옵트아웃 대응)

### `POST /admin/moderate`
콘텐츠 모더레이션 (NSFW 재분류 등)

---

## Rate Limiting

| 대상 | 제한 |
|------|------|
| 비인증 검색 | 30회/분 |
| 인증 검색 | 60회/분 |
| 피드 | 120회/분 |
| 업로드 | 10회/시간 |
| 좋아요 | 60회/분 |

## 에러 응답
```json
{
    "error": {
        "code": "RATE_LIMITED",
        "message": "요청이 너무 많습니다. 잠시 후 다시 시도해주세요.",
        "retry_after": 30
    }
}
```
