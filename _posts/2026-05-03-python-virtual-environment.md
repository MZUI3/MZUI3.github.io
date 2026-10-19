---
title: Python 가상환경으로 프로젝트 의존성 분리하기
description: Python 표준 venv로 프로젝트별 가상환경을 만들고 활성화하는 방법, 활성화가 실제로 하는 일, requirements.txt 관리와 Windows에서 흔한 실행 정책 오류까지 정리합니다.
date: 2026-05-03 16:49:00 +0900
categories: [개발, Python]
tags: [Python, venv, pip]
---

프로젝트마다 필요한 패키지 버전이 다를 수 있으므로, 전역 Python 환경에 모든 패키지를 설치하는 대신 프로젝트별 가상환경을 사용하는 편이 안전합니다. Python 표준 라이브러리의 `venv`로 별도 환경을 만들 수 있습니다.

**가상환경(virtual environment)**이란 프로젝트 폴더 안에 Python 실행 파일과 패키지 설치 공간을 따로 두어, 그 프로젝트에서 설치한 패키지가 다른 프로젝트나 시스템 전체에 영향을 주지 않게 하는 독립된 실행 환경입니다.

가상환경 없이 작업하면 이런 문제가 생깁니다. 프로젝트 A는 `numpy 1.26`에서만 동작하고 프로젝트 B는 `numpy 2.x`가 필요하다고 해 봅시다. 전역 환경에는 numpy를 한 버전만 설치할 수 있으므로, B를 위해 업그레이드하는 순간 A가 깨집니다. 가상환경은 각 프로젝트에 자기만의 `site-packages`(패키지가 설치되는 폴더)를 주어 이 충돌을 없앱니다.

> 이 글의 명령은 Python 3.3 이상에 기본 포함된 `venv` 기준이고, 아래 출력 예시는 Windows 11, Python 3.14에서 실행해 확인했습니다.
{: .prompt-info }

## 가상환경 만들기

프로젝트 폴더에서 다음 명령을 실행합니다. 마지막 `.venv`는 만들 폴더 이름입니다.

```bash
python -m venv .venv
```

Windows에서 `python` 명령이 없다고 나오면 Python 런처를 써서 `py -m venv .venv`를 실행합니다.

Windows PowerShell에서는 아래 명령으로 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS와 Linux에서는 다음과 같이 활성화합니다.

```bash
source .venv/bin/activate
```

활성화되면 프롬프트 앞에 `(.venv)`가 붙습니다. 작업을 마치고 원래 환경으로 돌아갈 때는 `deactivate`를 입력합니다.

<details markdown="1">
<summary>PowerShell에서 "이 시스템에서 스크립트를 실행할 수 없으므로…" 오류가 날 때</summary>

Windows는 기본 실행 정책 때문에 `Activate.ps1` 같은 스크립트 실행을 막을 수 있습니다. 현재 PowerShell 창에서만 잠시 허용하려면 다음을 실행한 뒤 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

`-Scope Process`는 이 창을 닫으면 원래대로 돌아가므로 시스템 설정을 영구히 바꾸지 않습니다. 매번 입력하기 번거롭다면 `-Scope CurrentUser`로 현재 사용자에게만 적용할 수도 있습니다.

- 근거: [about_Execution_Policies (Microsoft Learn)](https://learn.microsoft.com/powershell/module/microsoft.powershell.core/about/about_execution_policies)
</details>

## 활성화는 실제로 무엇을 하나

활성화 스크립트가 하는 핵심 일은 **환경 변수 `PATH`의 맨 앞에 `.venv/Scripts`(macOS·Linux는 `.venv/bin`)를 끼워 넣는 것**입니다. 그래서 터미널에서 `python`이나 `pip`를 입력하면 시스템 Python보다 가상환경 안의 실행 파일이 먼저 찾아집니다.

가상환경 안의 Python은 시작할 때 `.venv/pyvenv.cfg` 파일을 읽고, 자신이 가상환경 안에 있음을 알아차려 패키지를 `.venv` 안의 `site-packages`에서 찾습니다. 직접 확인해 볼 수 있습니다.

```bash
python -c "import sys; print(sys.prefix); print(sys.prefix != sys.base_prefix)"
```

```text
C:\projects\myapp\.venv
True
```

`sys.prefix`(현재 환경의 위치)와 `sys.base_prefix`(원래 설치된 Python의 위치)가 다르면, 즉 `True`가 나오면 가상환경 안에서 실행 중입니다.

이 원리 때문에 **활성화는 필수가 아닙니다.** `.venv\Scripts\python.exe app.py`(macOS·Linux는 `.venv/bin/python app.py`)처럼 가상환경의 Python을 경로로 직접 실행해도 똑같이 동작합니다. 크론 작업이나 배포 스크립트에서는 이 방식을 많이 씁니다.

## 패키지 설치와 requirements.txt

활성화한 뒤 패키지를 설치하면 해당 프로젝트 환경에만 설치됩니다.

```bash
python -m pip install requests
python -m pip freeze > requirements.txt
```

`pip install` 대신 `python -m pip install`을 쓰는 이유는, 지금 선택된 `python`에 딸린 pip를 확실히 쓰기 위해서입니다. PATH가 꼬여 있으면 `pip`와 `python`이 서로 다른 설치본을 가리킬 수 있습니다.

`pip freeze`는 현재 환경에 설치된 패키지를 `이름==버전` 형식으로 출력합니다.

```text
certifi==2026.7.22
charset-normalizer==3.5.2
idna==3.20
requests==2.34.2
urllib3==2.8.0
```

(2026년 10월에 Python 3.14로 실행한 결과입니다. 버전은 설치 시점에 따라 다릅니다.) `requests` 하나만 설치했는데 다섯 줄이 나오는 이유는, requests가 의존하는 패키지(전이 의존성)까지 함께 설치되기 때문입니다.

다른 개발자는 같은 환경에서 `python -m pip install -r requirements.txt`를 실행해 의존성을 설치할 수 있습니다. 실제 서비스에서는 재현 가능한 설치를 위해 패키지 버전을 검토하고 고정하는 것이 중요합니다.

<details markdown="1">
<summary>더 알아보기: pip freeze의 한계와 다른 도구들</summary>

`pip freeze`는 "내가 직접 고른 패키지"와 "따라 들어온 패키지"를 구분하지 않고 모두 적습니다. 그래서 나중에 requests를 지워도 certifi 같은 줄이 그대로 남아 목록이 점점 지저분해집니다.

이를 보완하려고 직접 쓰는 패키지만 적은 파일(`pyproject.toml`, `requirements.in` 등)과, 전이 의존성까지 정확한 버전으로 고정한 잠금 파일(lock file)을 나눠 관리하는 도구들이 있습니다.

| 도구 | 특징 |
| --- | --- |
| pip-tools | `requirements.in` → `pip-compile` → 고정된 `requirements.txt` |
| Poetry | `pyproject.toml` + `poetry.lock`, 가상환경도 함께 관리 |
| uv | Rust로 만든 빠른 설치기, `uv venv`, `uv pip`, `uv.lock` 지원 |

팀 프로젝트라면 이런 도구 중 하나를 정해 쓰는 것을 추천합니다. 이 도구들도 내부적으로는 결국 같은 가상환경 구조를 사용합니다.
</details>

## 저장소에 포함하지 않기

가상환경 폴더는 크고 운영체제와 Python 버전에 영향을 받으므로 보통 Git에 올리지 않습니다. 가상환경 안의 실행 파일과 스크립트는 만든 컴퓨터의 절대 경로를 기억하고 있어서, 폴더를 다른 위치나 다른 컴퓨터로 복사하면 제대로 동작하지 않습니다.

`.gitignore`에 `.venv/`를 추가하고 `requirements.txt`처럼 필요한 의존성 목록만 관리합니다.

```gitignore
.venv/
```

참고로 Python 3.13부터는 `venv`가 가상환경 폴더 안에 모든 파일을 무시하는 `.gitignore`를 자동으로 만들어 줍니다. 그래도 팀원마다 Python 버전이 다를 수 있으니 프로젝트 `.gitignore`에도 적어 두는 편이 안전합니다.

## 참고 자료

- [venv — Creation of virtual environments (Python 공식 문서)](https://docs.python.org/3/library/venv.html)
- [Installing packages using pip and virtual environments (Python Packaging User Guide)](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
