# 불균형 범주 분류를 위한 동적 샘플링 스케줄러

## 1. Overview

#### 1) Background

분류 데이터셋은 의도되지 않는 한 클래스 불균형을 포함하는 경우가 많으며 이는 다음과 같은 문제를 일으킬 수 있다.

- 모델이 다수 클래스에 편향되어 다른 클래스를 무시하는 경향이 있다.
- 모델이 전체적으로 높은 정확도(Accuracy)를 달성할 수 있지만, 소수 클래스에 대한 정확도와 재현율이 낮아질 수 있다.
- 소수 클래스에 대한 충분한 학습 샘플이 부족하여 모델이 적절한 결정 경계를 학습하지 못할 수 있다.

##### Related Works in Batch Sampling

| 샘플링 기준                                | 샘플링 방법    | 특징                                                         |
| ------------------------------------------ | -------------- | ------------------------------------------------------------ |
| 샘플 개수                                  | Over Sampling  | 저빈도 레이블 샘플을 중복 사용<br />➔ 특정 샘플에 대한 과적합 위헙 |
|                                            | Under Sampling | 고빈도 레이블 중 일부만 사용<br />➔ 정보 손실 발생           |
| 샘플 난이도<br />(손실값 기준)             | Easy Sample    | 손실값이 작은 샘플 위주<br />이상치와 노이즈에 강건<br />작은 손실값 ➔ 느린 학습 속도 |
|                                            | Hard Sample    | 손실값이 높은 샘플 위주<br />과적합 발생 가능성 높음<br />큰 손실값 ➔ 빠른 학습 속도 |
| 샘플의 불확실성<br />(예측 결과 이력 기준) | Online Batch   | 가장 최근의 손실값 기반<br />학습 속도 향상                  |
|                                            | Active Bias    | 일반화 오류 감소<br />학습 수렴 속도 저하                    |
|                                            | Recency Bias   | Active Bias의 개선으로 학습 속도 가속화<br />저빈도 레이블에 대한 개선 어려움 |

#### 2) Project Goal

> 1. 저빈도 레이블의 학습량을 늘릴 것
> 2. 고정된 샘플링 확률에 의한 편향 문제를 완화시킬 것

#### 3) Proposed Method

- **샘플링 확률 계산 방법**: 저빈도 클래스의 샘플링 확률을 높이기 위한 방법을 제안한다.
    - 전체 데이터에서 각 클래스의 샘플 수를 기반으로 선택 확률을 계산한다.
    - 저빈도 레이블의 선택 확률을 높이고 고빈도 레이블의 선택 확률을 낮출 수 있도록 선택 확률 스코어를 정의한다.

- **배치 샘플링 확률 변경**: 학습 중에 배치 샘플링을 변경하여 편향 문제를 완화하기 위한 스케줄링 방법을 제안한다.
    - 샘플의 선택 확률은 변경되지 않기 때문에 새로운 편향이 발생할 수 있다.
    - 학습률 스케줄링 방법을 차용하여 미니배치 샘플링 선택 방법을 전환하는 스케줄러를 정의한다.
    - 정의된 순서에 따라 Sequential Sampling 방법과 제안 샘플링 방법을 선택하여 사용한다.


#### 4) Structure

| 파일/디렉토리           | 설명                                                         |
| ----------------------- | ------------------------------------------------------------ |
| **data/**               | 실험에 사용할 데이터가 저장되는 디렉토리                     |
| **src/agent/**          | 모델을 선언하고 학습과 평가를 수행하는 코드가 저장되는 디렉토리 |
| **src/models/**         | 모델 아키텍처 정의, 손실 함수, 평가 지표 등을 모델 파일에서 정의 |
| **src/datamodule/**     | 데이터셋에 관련된 코드                                       |
| **configs/config.yaml** | 전역 config 설정 파일                                        |
| **config/task/**        | 실험별 config를 정의한 yaml 파일이 저장되는 디렉토리         |

## 3. Getting Started

#### 1) Set Env.

- hydra-core==1.2.0
- transformers==4.18.0
- torch==1.10.1

```bash
pip install -r requirements/requirements_hydra.txt
pip install -r requirements/requirements_transformers.txt
```

#### 2) Train

- `configs/task/sst2/sst2_bert_cls_dsampling_schedule_cyclic.yaml` 기준 학습

```bash
python run.py --multirun \
              task=sst2/sst2_bert_cls_dsampling_schedule_cyclic \
              agent.cycle=2,3,4,5 \
              optimizer.lr=1e-5 \
              seed=42 \
              agent.batch_weight_gamma=1
```

#### 3) Predict

- 학습된 모델 `checkpoint/sst2_cls-bert_cls-dsampling_scheduler_cyclic_3_g1_v1/exp_lr1e-05_ms96_mt0_seed42_/trained_model/`에 대한 예측 파일 생성 예시

```bash
python run.py mode=predict \
              task=sst2/sst2_bert_cls_dsampling_schedule_cyclic \
              model.path=checkpoint/sst2_cls-bert_cls-dsampling_scheduler_cyclic_3_g1_v1/exp_lr1e-05_ms96_mt0_seed42_/trained_model/ \
              seed=42 \
              agent.batch_weight_gamma=1 \
              predict_file_path=predictions.jsonl
```

## About

- [ **특허** ] 10-2458360, “불균형 데이터에 대한 딥러닝 분류 모델 성능을
	향상시키기 위한 레이블 기반 샘플 추출 장치 및 그 방법”, 2022.10.
- [ **논문** ] “불균형 범주 분류를 위한 동적 샘플링 스케줄러”, 제 33 회 한글 및 한국어 정보처리 학술대회 논문집 (2021): 221-226.

