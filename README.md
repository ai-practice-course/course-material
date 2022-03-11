### Course

이번주에는 수요예측 방법 중에 시계열 자체가 뿐만 아니라, 기타 meta 정보를 활용하는 인과관계 모형들에 대해 알아봅니다.
인과형 예측모형에는
  - 통계적 모형
  - ML 모형
  - DL 모형
들이 있고, 간단하게 각각의 모형들에 대해 짚고 넘어가는 시간을 가지도록 하겠습니다.


Dataset candidate:

> 데이터는 frequency가 높거나, 종목의 개수가 높은 것들이 좋을 것 같음

  - Walmart Sales Data
    - 참고 할만한 다양한 notebook들이 존재
    - 여기에 20년 이후 출시된 모형을 적용시켜 성능 향상이 있는지 실험
    - 참고 사이트
      * [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy)
      * [캐글코리아](https://m.facebook.com/groups/230717130993727/posts/%EA%B9%80%EC%98%81%EB%AF%BC-posted-in-%EC%BA%90%EA%B8%80-%EC%BD%94%EB%A6%AC%EC%95%84-(Kaggle-Korea)/948649505867149/)

  - 암호화폐 가격 예측
    - 다양한 예측 시도가 이루어지고 있음
    - 암호화폐의 움직임과 동시에 여러가지 메타정보를 포함하는 형태의 연구들이 존재함
    - 다양한 데이터를 수집하고 활용하는 연습이 동시에 가능해질 것으로 보임
    - 참고 사이트
      * [암호화폐 가격 예측을 위한 딥러닝 앙상블 모델 개발 연구] (http://www.riss.kr/search/detail/DetailView.do?p_mat_type=be54d9b8bc7cdb09&control_no=8070829862e45fa2ffe0bdc3ef48d419&outLink=K)
      * [예일대 경제학자 “암호화폐 가격에 영향을 주는 지표 보고서”](https://www.coinpress.co.kr/2018/08/11/8374/)


  - [공공데이터 포털](https://www.data.go.kr/index.do)
    - 다양한 데이터가 있음
    - 수요 데이터는 주로 전력 데이터가 있었음
    - 부동산 데이터는 너무 sparse한 데이터로 우리 과정에 적합하지는 않은 것 같음
    - 이 중에서 하나 골라 보는 것도 좋을 것 같음
      * [국가중점데이터](https://www.data.go.kr/tcs/eds/selectCoreDataListView.do)
      * [부산시 실시간 교통정보] (https://www.data.go.kr/tcs/eds/selectCoreDataView.do?coreDataInsttCode=6260890&coreDataSn=1&searchCondition1=coreDataNm&searchKeyword1=&searchOrder=INSTT_NM_ASC)
      * [RFID 기반 음식물쓰레기정보](https://www.data.go.kr/tcs/eds/selectCoreDataView.do?coreDataInsttCode=B552584&coreDataSn=1&searchCondition1=coreDataNm&searchKeyword1=&searchOrder=INSTT_NM_ASC)
      * [한국공항공사](https://www.airport.co.kr/www/cms/frFlightStatsCon/timeSeriesStats.do?MENU_ID=1250)



### 참고자료
- [git 기본](https://sabarada.tistory.com/75)