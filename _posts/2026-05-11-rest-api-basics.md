---
title: REST API를 설계할 때 먼저 정할 것들
description: 리소스 중심 URL, HTTP 메서드의 안전성·멱등성, 상태 코드, 오류 응답 형식까지 REST API 계약을 정할 때 먼저 결정해야 할 것들을 RFC 근거와 함께 정리합니다.
date: 2026-05-11 22:57:00 +0900
categories: [개발, Web]
tags: [REST, API, HTTP]
---

프론트엔드와 백엔드가 함께 작업할 때 API 계약을 먼저 정해 두면 구현과 테스트가 수월해집니다. URL에는 리소스를 드러내고, HTTP 메서드에는 수행할 동작을 담는 방식부터 시작할 수 있습니다.

먼저 용어 두 개를 짚고 갑니다.

- **API 계약(API contract)**: 클라이언트와 서버가 "이 주소로 이런 요청을 보내면 이런 응답이 온다"고 미리 합의한 약속입니다. 계약이 먼저 있으면 프론트엔드는 서버가 완성되기 전에도 가짜 응답(mock)으로 화면을 만들 수 있습니다.
- **REST(REpresentational State Transfer)**: Roy Fielding이 2000년 박사 논문에서 제안한 웹 아키텍처 스타일입니다. 서버가 가진 데이터를 **리소스**로 보고, 각 리소스에 URL을 붙인 뒤, 표준 HTTP 메서드로 그 리소스를 다루는 방식입니다.

> 실무에서 "REST API"라고 부르는 것은 대부분 *HTTP + JSON + 리소스 중심 URL* 정도의 의미입니다. Fielding이 정의한 REST의 제약 조건(예: 응답 안에 다음 행동 링크를 담는 HATEOAS)까지 모두 지키는 API는 드뭅니다. 면접에서 "REST를 완전히 지켰나요?"라는 질문이 나오면 이 차이를 구분해 답할 수 있으면 좋습니다.
{: .prompt-info }

## 리소스 중심 URL

게시글 리소스를 예로 들면 다음처럼 표현할 수 있습니다.

| 요청 | 의미 |
| --- | --- |
| `GET /api/posts` | 게시글 목록 조회 |
| `GET /api/posts/42` | 게시글 한 건 조회 |
| `POST /api/posts` | 게시글 생성 |
| `PATCH /api/posts/42` | 게시글 일부 수정 |
| `DELETE /api/posts/42` | 게시글 삭제 |

URL을 동사로 늘리기보다 리소스와 표준 HTTP 메서드 조합으로 의미를 전달합니다. 예를 들어 `POST /api/deletePost?id=42`처럼 동작을 URL에 넣으면, 메서드마다 URL이 하나씩 늘어나고 캐시·로그·권한 규칙도 URL마다 따로 관리해야 합니다. 단, 실제 API에서는 인증과 권한 확인을 각 요청에 맞게 적용해야 합니다.

## 메서드를 고르는 기준: 안전성과 멱등성

메서드를 고를 때는 "느낌"이 아니라 HTTP 명세가 정한 두 성질을 기준으로 삼습니다.

- **안전(safe)**: 요청이 서버의 상태를 바꾸지 않는다는 뜻입니다. `GET`은 안전하므로 브라우저나 검색 엔진이 마음대로 미리 불러와도 문제가 없어야 합니다.
- **멱등(idempotent)**: 같은 요청을 한 번 보내든 여러 번 보내든 서버에 남는 결과가 같다는 뜻입니다. 네트워크가 끊겨 응답을 못 받았을 때 **안심하고 재시도할 수 있는지**가 여기서 갈립니다.

| 메서드 | 안전 | 멱등 | 용도 |
| --- | --- | --- | --- |
| `GET` | O | O | 조회 |
| `PUT` | X | O | 리소스 **전체**를 보낸 내용으로 교체 |
| `DELETE` | X | O | 삭제 (두 번째 요청은 404일 수 있지만 서버 상태는 같음) |
| `POST` | X | X | 생성, 또는 다른 메서드로 표현하기 어려운 처리 |
| `PATCH` | X | 보장 안 됨 | 리소스 **일부**만 수정 |

`PATCH`가 멱등이 아닐 수 있는 이유는 `{"op": "increment", "field": "views"}`처럼 "현재 값에서 1 더하기" 같은 변경도 표현할 수 있기 때문입니다. 반대로 `{"title": "새 제목"}`처럼 값을 덮어쓰는 PATCH는 사실상 멱등으로 동작합니다.

<details markdown="1">
<summary>더 알아보기: POST를 안전하게 재시도하려면</summary>

결제나 주문 생성처럼 `POST`인데 중복 실행되면 안 되는 요청은, 클라이언트가 요청마다 고유한 키를 헤더(관례적으로 `Idempotency-Key`)에 담아 보내고 서버가 같은 키의 요청을 한 번만 처리하는 방식을 많이 씁니다. Stripe 같은 결제 API가 대표적입니다. 이 헤더는 IETF HTTPAPI 워킹그룹에서 표준화 초안(draft-ietf-httpapi-idempotency-key-header)으로 논의됐지만, 2025년 10월의 07판을 마지막으로 2026년 4월에 만료되었고 RFC로 발행되지는 않았습니다(2026년 10월 확인). 표준은 아니지만 관례로 널리 쓰이는 헤더라고 보면 됩니다.

- 근거: [RFC 9110 §9.2 Common Method Properties](https://www.rfc-editor.org/rfc/rfc9110#section-9.2)
</details>

## 응답 형식과 상태 코드

클라이언트가 성공과 실패를 예측할 수 있도록 응답 형식과 상태 코드를 일관되게 유지합니다.

```json
{
  "data": {
    "id": 42,
    "title": "REST API 설계"
  }
}
```

상태 코드는 아래 정도만 정확히 구분해도 대부분의 상황을 다룰 수 있습니다.

| 코드 | 언제 쓰나 | 함께 챙길 것 |
| --- | --- | --- |
| `200 OK` | 조회·수정 성공 | 응답 본문 |
| `201 Created` | 새 리소스 생성 | 새 리소스 주소를 `Location` 헤더에 (예: `Location: /api/posts/43`) |
| `204 No Content` | 성공했지만 돌려줄 본문이 없음 (삭제 등) | 본문을 비워 둠 |
| `400 Bad Request` | 요청 형식이 잘못됨 | 어떤 필드가 왜 잘못됐는지 |
| `401 Unauthorized` | **인증**되지 않음 (로그인 안 함, 토큰 만료) | `WWW-Authenticate` 헤더 |
| `403 Forbidden` | 인증은 됐지만 **권한**이 없음 | |
| `404 Not Found` | 리소스가 없음 | |
| `409 Conflict` | 현재 상태와 충돌 (이미 존재하는 아이디 등) | |

`401`은 이름이 "Unauthorized(권한 없음)"이지만 실제 의미는 "누구인지 확인되지 않음(unauthenticated)"입니다. 권한 문제는 `403`입니다. 이름 때문에 둘을 바꿔 쓰는 실수가 흔합니다.

오류 응답에도 안정적인 오류 코드와 사람이 읽을 수 있는 메시지를 제공하면 클라이언트가 대응하기 쉽습니다. 형식을 새로 만들기보다 표준인 **Problem Details**([RFC 9457](https://www.rfc-editor.org/rfc/rfc9457), 이전 RFC 7807을 대체)를 따르는 것도 방법입니다.

```json
{
  "type": "https://example.com/problems/invalid-title",
  "title": "제목이 비어 있습니다",
  "status": 400,
  "detail": "title 필드는 1자 이상이어야 합니다."
}
```

이때 응답의 `Content-Type`은 `application/problem+json`으로 보냅니다.

## 문서화와 검증

요청·응답 예시, 필수 필드, 오류 상황을 OpenAPI 같은 형식으로 문서화하면 프론트엔드와 백엔드가 같은 계약을 공유할 수 있습니다. **OpenAPI**는 API의 주소, 파라미터, 응답 구조를 YAML이나 JSON으로 적는 표준 명세이고, 이 파일 하나로 문서 페이지(Swagger UI)와 클라이언트 코드를 자동으로 만들 수 있습니다.

페이지네이션, 정렬, 필터의 이름과 기본값도 함께 정하고, 정상·오류 응답을 모두 테스트하는 것이 좋습니다. 예를 들면 다음처럼 쿼리 파라미터 규칙을 미리 정해 둡니다.

```text
GET /api/posts?page=2&size=20&sort=-createdAt&tag=python
```

<details markdown="1">
<summary>더 알아보기: page 방식 페이지네이션의 한계</summary>

`page`/`size`(offset) 방식은 단순하지만, 뒤쪽 페이지로 갈수록 DB가 앞의 행을 모두 세고 버려야 해서 느려지고, 조회 도중 새 글이 추가되면 같은 글이 두 페이지에 걸쳐 보이기도 합니다. 피드처럼 데이터가 계속 쌓이는 화면에서는 "마지막으로 본 글의 id 이후 20개"처럼 기준점을 넘기는 **커서(cursor) 방식**을 많이 씁니다.

```text
GET /api/posts?cursor=eyJpZCI6NDJ9&size=20
```
</details>

> (여기에 실제 프로젝트에서 API 계약을 정하며 겪은 일 — 예: 상태 코드나 필드 이름을 두고 프론트엔드와 맞춘 경험 — 을 적으면 글의 설득력이 크게 올라갑니다.)
{: .prompt-tip }

## 참고 자료

- [RFC 9110 – HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110)
- [RFC 9457 – Problem Details for HTTP APIs](https://www.rfc-editor.org/rfc/rfc9457)
- [IETF Datatracker – The Idempotency-Key HTTP Header Field](https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/)
- [Roy Fielding, *Architectural Styles and the Design of Network-based Software Architectures* (2000), 5장](https://ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html)
