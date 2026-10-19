---
icon: fas fa-info-circle
order: 4
---

<style>
  .about-hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem;
    margin-bottom: 2rem;
    border: 1px solid var(--card-border-color, rgba(0, 0, 0, 0.1));
    border-radius: 1rem;
    background: var(--card-bg, transparent);
  }
  .about-hero img {
    width: 7rem;
    height: 7rem;
    flex-shrink: 0;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid var(--card-border-color, rgba(0, 0, 0, 0.1));
  }
  .about-hero h2 {
    margin: 0 0 0.35rem;
    border: 0;
  }
  .about-hero p {
    margin: 0 0 0.9rem;
    color: var(--text-muted-color, #6c757d);
  }
  .about-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .about-links a {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border: 1px solid var(--btn-border-color, rgba(0, 0, 0, 0.15));
    border-radius: 999px;
    font-size: 0.9rem;
    color: var(--text-color, inherit);
    text-decoration: none;
    border-bottom: 1px solid var(--btn-border-color, rgba(0, 0, 0, 0.15)) !important;
    transition: background-color 0.15s, color 0.15s;
  }
  .about-links a:hover {
    background: var(--link-color, #0056b2);
    border-color: var(--link-color, #0056b2) !important;
    color: #fff;
  }
  .about-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr));
    gap: 0.9rem;
    margin: 1rem 0 2rem;
  }
  .about-card {
    display: block;
    padding: 1.1rem 1.2rem;
    border: 1px solid var(--card-border-color, rgba(0, 0, 0, 0.1));
    border-radius: 0.75rem;
    background: var(--card-bg, transparent);
    color: var(--text-color, inherit);
    text-decoration: none;
    border-bottom: 1px solid var(--card-border-color, rgba(0, 0, 0, 0.1)) !important;
  }
  a.about-card {
    transition: transform 0.15s, border-color 0.15s;
  }
  a.about-card:hover {
    transform: translateY(-2px);
    border-color: var(--link-color, #0056b2) !important;
    color: var(--text-color, inherit);
  }
  .about-card i {
    font-size: 1.3rem;
    color: var(--link-color, #0056b2);
    margin-bottom: 0.6rem;
  }
  .about-card strong {
    display: block;
    margin-bottom: 0.25rem;
    color: var(--heading-color, inherit);
  }
  .about-card span {
    font-size: 0.88rem;
    color: var(--text-muted-color, #6c757d);
  }
  .about-card .about-count {
    float: right;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--link-color, #0056b2);
  }
  @media (max-width: 576px) {
    .about-hero {
      flex-direction: column;
      text-align: center;
    }
    .about-links {
      justify-content: center;
    }
  }
</style>

<div class="about-hero">
  <img src="{{ site.avatar | relative_url }}" alt="양경동 프로필 사진">
  <div>
    <h2>안녕하세요, 양경동입니다</h2>
    <p>경량 AI 모델과 온디바이스 추론, 로봇 학습을 공부합니다.</p>
    <div class="about-links">
      <a href="https://mzui3.notion.site/AI-555b32aeb4c24c189c211ddb6d7ffc1c?source=copy_link" target="_blank" rel="noopener"><i class="fas fa-briefcase"></i> Portfolio</a>
      <a href="https://github.com/MZUI3" target="_blank" rel="noopener"><i class="fab fa-github"></i> GitHub</a>
      <a href="mailto:mzui2033@gmail.com"><i class="fas fa-envelope"></i> Email</a>
    </div>
  </div>
</div>

이 블로그에는 연구실 세미나에서 발표하며 읽은 AI 논문과 개발하면서 익힌 기초를 정리합니다. 논문 리뷰에는 원 논문 링크를 함께 남기고, 코드는 직접 돌려 본 결과까지 적으려고 합니다.

## 관심 분야

<div class="about-grid">
  <div class="about-card">
    <i class="fas fa-eye"></i>
    <strong>Vision Transformer · 경량화</strong>
    <span>ViT 구조와 Pruning, Quantization, BNN 같은 모델 압축</span>
  </div>
  <div class="about-card">
    <i class="fas fa-file-lines"></i>
    <strong>VLM 기반 OCR · VQA</strong>
    <span>이미지 속 글자를 읽고, 이미지에 대한 질문에 답하는 모델</span>
  </div>
  <div class="about-card">
    <i class="fas fa-code"></i>
    <strong>웹 서비스 개발</strong>
    <span>Vue.js 프론트엔드와 Spring 백엔드</span>
  </div>
  <div class="about-card">
    <i class="fas fa-robot"></i>
    <strong>강화학습 · 로봇 학습</strong>
    <span>탐험, 모방학습, Offline RL, Diffusion 기반 정책</span>
  </div>
</div>

## 프로젝트

<div class="about-grid">
  <a class="about-card" href="https://app.notion.com/p/1c7ccb42fc1c452281959d52db412a87" target="_blank" rel="noopener">
    <i class="fas fa-microchip"></i>
    <span class="about-count">2023</span>
    <strong>TinyML Design Contest</strong>
    <span>STM32(RAM 16KB)에서 동작하는 심실 부정맥 탐지 1D CNN · 세계 Top 11</span>
  </a>
  <a class="about-card" href="https://app.notion.com/p/3997f0db48dc80afad07d3046b15851d" target="_blank" rel="noopener">
    <i class="fas fa-layer-group"></i>
    <span class="about-count">2023</span>
    <strong>ExMobileViT</strong>
    <span>1저자 · 파라미터 +5%로 ImageNet Top-1 +0.68%p</span>
  </a>
  <a class="about-card" href="https://app.notion.com/p/3997f0db48dc80e6bb52ce1269641a7a" target="_blank" rel="noopener">
    <i class="fas fa-cubes"></i>
    <span class="about-count">2023</span>
    <strong>ResidualViT</strong>
    <span>Tiny 버전 0.89M으로 MobileViT-S 대비 CIFAR-100 +7.0%p</span>
  </a>
  <a class="about-card" href="https://app.notion.com/p/37c7f0db48dc803d8221c436ff755bbd" target="_blank" rel="noopener">
    <i class="fas fa-pen-nib"></i>
    <span class="about-count">2023</span>
    <strong>캐리커처 로봇팔</strong>
    <span>생성형 AI 결과를 G-Code로 바꿔 로봇팔이 그림 · 공학 페스티벌 최우수상</span>
  </a>
</div>

## 블로그에서 다루는 것

<div class="about-grid">
{%- assign about_categories = "Neural Network|fas fa-network-wired|CNN, 경량화, Object Detection, Audio;Reinforcement Learning|fas fa-chess-knight|탐험, 모방학습, Offline RL, 계층적 RL, 멀티에이전트;Transformer|fas fa-shapes|Attention, Vision Transformer, LLM;Generative AI|fas fa-wand-magic-sparkles|Diffusion, Image Synthesis;개발|fas fa-terminal|Web, Python, DB, CI/CD, Robotics" | split: ";" -%}
{%- for item in about_categories -%}
  {%- assign parts = item | split: "|" -%}
  {%- assign cat_name = parts[0] -%}
  {%- assign cat_posts = site.categories[cat_name] -%}
  {%- assign cat_slug = cat_name | slugify | url_encode -%}
  <a class="about-card" href="{{ '/categories/' | append: cat_slug | append: '/' | relative_url }}">
    <i class="{{ parts[1] }}"></i>
    <span class="about-count">{{ cat_posts.size | default: 0 }}편</span>
    <strong>{{ cat_name }}</strong>
    <span>{{ parts[2] }}</span>
  </a>
{%- endfor -%}
</div>
