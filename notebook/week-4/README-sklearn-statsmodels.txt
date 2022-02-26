# Scikit-Learn 사용법 요약

Scikit-learn은 머신러닝에 사용되는 지도/비지도 학습 알고리즘을 제공하는 파이썬 라이브러리다. 
내부 구조를 살펴보면 NumPy, pandas, Matplotlib과 같이 이미 널리 쓰이는 기술을 기반으로 한다고 한다.
다양한 머신러닝 알고리즘을 구현한 파이썬 라이브러리
심플하고 일관성 있는 API, 유용한 온라인 문서, 풍부한 예제
머신러닝을 위한 쉽고 효율적인 개발 라이브러리 제공
다양한 머신러닝 관련 알고리즘과 개발을 위한 프레임워크와 API 제공
많은 사람들이 사용하며 다양한 환경에서 검증된 라이브러리


scikit-learn 주요 모듈

datasets: 예제 데이터셋
preprocessing: 다양한 전처리 기능 제공 (변환, 정규화, 스케일링 )
feature_selection: Feature을 선택할 수 있는 기능
feature_extraction: Feature 추출에 사용
decomposition: 차원 축소 관련 알고리즘 (PCA, NMF, Truncated SVD 등)
metrics: 분류, 회귀, 클러스터링, Pairwise에 대한 다양한 성능 측정 방법 제공( Acc, Prec, Rec, ROC-AUC, RMSE 등)
pipeline: Feature 처리 등의 변환과 ML 학습 예측을 묶어서 실행할 수 있는 유틸리티
linear_model: 선형회귀, ridge, lasso, logistic 등 회귀 관련 알고리즘과 SGD 알고리즘 제공
svm: svm
neighbors: K-NN 등
naive_bayes: Gaussian NB, 다항분포 NB 등
cluster: k-means, higherarchical cluster, DBScan 등

estimator API

일관성: 모든 객체는 일관된 문서를 갖춘 제한된 메서드 집합에서 비롯된 공통 인터페이스 공유
검사(inspection): 모든 지정된 파라미터 값은 공개 속성으로 노출
구성: 많은 머신러닝 작업은 기본 알고리즘의 시퀀스로 나타낼 수 있으며, Scikit-Learn은 가능한 곳이라면 어디서든 이 방식을 사용
합리적인 기본값: 모델이 사용자 지정 파라미터를 필요로 할 때 라이브러리가 적절한 기본값을 정의

https://makeit.tistory.com/132

예제 데이터 세트 및 온라인 데이터셋
fetch_california_housing()
fetch_covtype()
fetch_20newsgroups()
fetch_olivetti_faces()
fetch_lfw_people()
fetch_lfw_paris()
fetch_rcv1()
fetch_mldaata()

예제 데이터 세트 구조

일반적으로 딕셔너리 형태로 구성
data: 특징 데이터 세트
target: 분류용은 레이블 값, 회귀용은 숫자 결과값 데이터
target_names: 개별 레이블의 이름 (분류용)
feature_names: 특징 이름
DESCR: 데이터 세트에 대한 설명과 각 특징 설명

기본 예제

```
import numpy as np  # numpy 패키지 가져오기
import matplotlib.pyplot as plt # 시각화 패키지 가져오기

## 2.데이터 가져오기
import pandas as pd # csv -> dataframe으로 전환
from sklearn import datasets # python 저장 데이터 가져오기

## 3.데이터 전처리
from sklearn.preprocessing import StandardScaler # 연속변수의 표준화
from sklearn.preprocessing import LabelEncoder # 범주형 변수 수치화

# 4. 훈련/검증용 데이터 분리
from sklearn.model_selection import train_test_split 

## 5.분류모델구축
from sklearn.tree import DecisionTreeClassifier # 결정 트리
from sklearn.naive_bayes import GaussianNB # 나이브 베이즈
from sklearn.neighbors import KNeighborsClassifier # K-최근접 이웃
from sklearn.ensemble import RandomForestClassifier # 랜덤 포레스트
from sklearn.ensemble import BaggingClassifier # 앙상블
from sklearn.linear_model import Perceptron # 퍼셉트론
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델
from sklearn.svm import SVC # 서포트 벡터 머신(SVM)
from sklearn.neural_network import MLPClassifier # 다층인공신경망

## 6.모델검정
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score # 정확도, 민감도 등
from sklearn.metrics import roc_curve, auc # ROC 곡선 그리기

## 7.최적화
from sklearn.model_selection import cross_validate # 교차타당도
from sklearn.pipeline import make_pipeline # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV # 하이퍼파라미터 튜닝
```


# Statsmodels (https://www.statsmodels.org/stable/api.html)

statsmodels.api
    - Regression
        - OLS
            - methods:
                - fit
                - fit_regularized
                - from_formula
                - get_distribution
                - hessian
                - hessian_factor
                - information
                - initialize
                - loglike
                - predict
                - score
            - properties
                - df_model
                - df_resid
                - endog_names
                - exdog_names
        - WLS
        - GLS
        - GLSAR
        - RecurseveLS
        - RollingOLS
        - RollingWLS

    - Imputation
        - BayeseGaussMI
        - MI
        - MICE
        - MICEData
