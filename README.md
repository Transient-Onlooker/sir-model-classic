# SIRVQ-D 격자 기반 전염병 확산 시뮬레이터

2차원 격자의 개체 상태 변화를 이용해 감염 확산과 방역 정책을 실험하는 Python 프로젝트입니다. 저장소 이름은 `sir-model-classic`이지만, **단순한 고전 SIR 연립미분방정식 풀이가 아닙니다.** 확률적 개체 전이·인구 이동·정책 이벤트를 갖춘 격자 기반 시뮬레이터입니다.

## 모델과 주요 기능

- 개체 상태: **S**(감염 가능), **I**(감염), **R**(회복), **V**(접종), **Q**(격리), **D**(사망)
- NumPy 2차원 격자와 주변 개체 상호작용을 이용한 감염 확산
- 방역 단계, 정책 대응 지연, 백신 접종 및 면역 감소
- 지역별 감염 밀집 탐지·봉쇄, 시나리오 조건에 따른 인구 이동
- Tkinter 설정 화면, Matplotlib 타임라인·재생 컨트롤 및 CSV/PNG 결과 저장
- `--batch` 비대화형 실행, `--no-history` 기록 최소화 모드

## 실행

Python 3 환경에서 저장소 루트를 기준으로 다음을 실행합니다.

```bash
python -m pip install -r "소스코드/requirements.txt"
python "소스코드/sir_model.py"
```

비대화형 실행은 다음과 같습니다.

```bash
python "소스코드/sir_model.py" --batch
```

- 진입점: [`소스코드/sir_model.py`](소스코드/sir_model.py)
- 시나리오: [`소스코드/config.json`](소스코드/config.json), `소스코드/example_config_*.json`
- 파라미터와 화면 사용법: [기존 상세 안내](소스코드/README.txt)
- 구현 이력: [개발 로그](소스코드/development_log.md)

GUI에는 Tkinter를 지원하는 실행 환경이 필요합니다. `--batch`에서도 코드 경로에 따라 시각화 라이브러리가 필요할 수 있습니다.

## 모델 해석의 한계

이 프로젝트의 상태 전이·방역 임계값·국가별 시나리오 값은 학습과 가상 실험을 위한 **모형 가정**입니다. 실제 국가의 정책·역학자료를 검증한 예측 모델로 해석하면 안 됩니다. 특히 확산률과 치사율을 연동하는 UI는 선택한 시뮬레이션 제약일 뿐, 두 값 사이의 보편적인 역학 법칙이 아닙니다. 난수 사용으로 결과는 실행마다 달라질 수 있습니다.

## 이어지는 프로젝트

- [sir-model-differential-equations](https://github.com/Transient-Onlooker/sir-model-differential-equations): 격자 모델과 별도로, SciPy ODE 기반 구획 모델을 실험한 Python GUI
- [sir-model-diff-web](https://github.com/Transient-Onlooker/sir-model-diff-web): 브라우저의 TypeScript/RK4 기반 구획 모델
