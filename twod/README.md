# s2_2d_package

2차원 정사각 격자에서 S² 스핀장의 유효장/결합 디코더를 검증하기 위한 코드와 의뢰서입니다. 작업 지시는 `BRIEF.md`에 있습니다.

```
s2spin/            핵심 라이브러리 (numpy)
  vmf.py           vMF 샘플러, Langevin 함수, 안정적인 log A(κ)
  model.py         해밀토니안(J_μ = J0 I + [D]× + Γ, h, Q), DMI 규약(interfacial/bulk), 0⊕1⊕2 분해
  mc.py            체커보드 열욕(정확한 조건부 vMF) + over-relaxation, 체인 벡터화
  estimators.py    26-매개변수 score matching(선형해, 잭나이프 오차), 의사우도(닫힌 형태), 매개변수 묶기
  observables.py   횡상관, Berg–Lüscher 위상전하, 자기상관 시간
  analytic.py      격자 가우시안 G(r), κ_ind 예측, Mermin–Wagner, Moriya 기대 방향, 평탄 방향 주석
  torch_losses.py  Stage 2용 PyTorch 손실 (작성 환경에서 미실행 — tests/test_torch_losses.py로 먼저 검증)
scripts/           stage1a_gaussian.py, stage1b_dmi.py, stage1c_skx.py, stage2_make_ensembles.py  (--quick 지원)
tests/             pytest -q tests/   (약 2초; torch 테스트는 torch가 있을 때만)
reference_1d/      1차원 사슬 검증에 쓴 원본 코드
```

실행: 저장소 루트에서 `pip install numpy scipy pytest` 후 `python -m pytest -q tests/`, 이어서 `cd scripts && python stage1a_gaussian.py --quick`.
