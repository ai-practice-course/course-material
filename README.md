## Course

이 과정에서는 현업에서 바로 적용해 볼 수 있는 데이터 분석 실무에 대해서 다룹니다.
그동안 학교에서 배웠던 지식 위주의 수업보다는 실제로 현업을 진행하면서 겪을 수 있는 많은 문제들에 대해서 직접 다루고자 합니다.
교실에서 배웠던 많은 지식들을 유기적으로 연결시키고, 실무에서 현업 담당자와 업무를 진행하는 과정에서 발생할 수 있는 어려움들에 대해서 다루고, 주의사항과 적절한 해법에 대해서도 같이 다루도록 합니다.

Data science 이론과 실무, 분석과 data engineering이 적절하게 조화된 수업이 되는 것을 목표로 향후 14주 동안 아래와 같이 구성할 예정입니다.



### **Part I** (Week 1 - 3)

Part Overview: 분석을 위한 환경 구축에 대해서 자세히 설명합니다. 환경 격리는 opensource에 기반하는 최근 data 분석에서는 업무 효율성 제고를 위해 필수불가결한 표준으로 자리잡고 있습니다. 이러한 작업에는 docker라는 container system이 표준으로 활용되고 있고, 보통 data science를 위한 docker image들이 이미 공개되어 있어 이를 분석에 바로 활용할 수 있습니다. 

하지만 금융권의 경우 완전하게 구축되어 있는 docker image를 만들어야 망분리 환경 하에서 작업을 진행할 수 있습니다. 이런 경우에는 본인이 원하는 package들이 모두 구성되어 있는 본인만의 docker image를 구축할 수 있어야 합니다. 이를 위해 이번 part에서는 분석 과정에서 필요한 engineering background를 데이터 분석가의 관점에서 설명하고자 합니다. 

자칫 이러한 engineering은 데이터분석의 역할이 아니라고 생각할 수 있습니다. 이런 질문에 대답하기 위해서 본격적인 환경 구성에 앞서 data science, 이보다 넓은 data analytic의 큰 트렌드에 대해서 먼저 짚고 넘어가고자 합니다.

Part Target: 이 이 파트를 통해 분석 과정에서 필요한 linux, python, git, 그리고 docker의 기본 개념과 활용법에 대해 이해하고, 본인만의 분석 환경을 만들 수 있습니다.

- Week 1
  - Target: Data science의 시작과 최신 흐름에 대해서 이해하고, data scientist의 역할이 세분화 되면서도, 기술적 coverage가 높아질 수 밖에 없는 구조에 대해서 이해합니다.

- Week 2
  - Target: Docker에 대해서 자세하게 이해하고, linux 환경에서 docker를 사용하는 방법에 대해서 학습합니다. 이미 공개되어 있는 image에서 출발, 데이터 분석 관점에서 유용한 python package를 포함한 환경을 만들어가는 과정과 팁에 대해서 학습합니다.
  - 과정 중간 중간, datascience의 data handling, modeling, visualization 과정에서 필요한 python, node 관련 지식들을 정리하는 시간을 갖도록 합니다. 

- Week 3
    - Target: Docker-compose에 대해서 이해하고, docker-compose를 활용해서 train용 image와 inference용 image를 직접 만들어 봅니다. 이후 변경된 이미지를 commit하여 본인만의 분석 환경을 발전시켜나갈 수 있는 역량을 키워봅니다.
    - MSA를 구현하기 위해서는 API 통신을 할 수 있는 web server가 필요합니다. 우리 과정에서 사용함은 물론, Data science에 있어서 가장 dominent한 언어인 python에 기반한 경량 webserver인 FastAPI와 WAS에 대한 전반적인 개론을 이야기 합니다.

### **Part II - Part III** (Week 4 - 6)

Part Overview: 실제 데이터를 활용해서 분석을 진행해 봅니다. 총 9주에 걸쳐 분석 과제를 진행합니다. 
모두 각자가 원하는 프로젝트가 있기는 하겠습니다만, 데이터 수급, 과제별 분석 난이도 상이에 따른 여러가지 문제점들을 미연에 방지하기 위해, 데이터는 ****를 사용하도록 합니다. 

회사에서 결국 필요한 건 business 의사결정입니다. 의사결정은 결국 판단의 문제이고, 모형으로 이야기하면 supervised learning을 통한 판별기 개발입니다. 이러한 형태의 분석이 회사에서 이루어지는 데이터 분석의 8할을 차지하고 있다고 해도 과언이 아닙니다. 
보통 회사에서 수집하는 OSS(Operation Support System) data들은 보통 Label 혹은 Label로 간주할 수 있는 데이터들을 가지고 있고 , 본 수업에서 사용될 데이터에도 label이 포함되어 있습니다. 현장에서 가장 잘 적용할 수 있는 방법론을 들이 존재하고 판별기를 구축해 보면 좋겠습니다.

다만 분석의 목표를 결정하고, 분석을 위한 research question을 구체화하는 과정은 본인의 관심사에 따라 자유롭게 정하시면 되겠습니다.
Research question을 설정하는 과정에서 현업 담당자와의 커뮤니케이션 상황을 가정하여, role play를 통해 communication 과정을 같이 경험해 보고자 합니다. 


### **Part II**


Part Target: 주어진 데이터에서 모호한 현업의 요청 사항을 바탕으로 research question을 구체화하는 과정에 대해서 배우고, 도출된 research question에 대한 답을 찾기 위한 분석을 진행할 수 있다.

- Week 4
    - Target: 분석 대상 데이터를 불러오는 방법과, 가장 기본적으로 파악해야 할 데이터를 느끼는 방법에 대해서 이해합니다. 이는 주로 간단한 EDA를 통해 얻어집니다. EDA를 위한 기초적인 library 조작법에 대해 익힙니다.
    - Lecture note:
        - Jupyter lab: DS의 기본
        - Pandas, Numpy에 대한 이해
        - Graph library에 대한 이해 - Interactive plot
    
    
- Week 5
    - Target: 현업의 요청사항은 언제나 blur합니다. 추상적입니다. Biz 언어에 근거합니다. 이를 데이터로 증명 가능한 가설들로 치환을 해야 합니다. 이러한 과정들에 대해 이해하고, 데이터로 풀어낼 수 있는 문제로 shaping해 낼 수 있습니다. 최소 2개 이상의 데이터로 증명 가능한 가설들을 도출해 봅니다.
    - Lecture note
        - 희망인가 가설인가: "매출을 확대하고 싶어요.", "고객을 만족도를 높이고 싶어요." 
        - Modeling vs data summary



- Week 6
    - Target: 명확하게 합의된 Metric 설정의 중요성과 현업 담당자와 metric을 합의하는 과정들에 대해서 알아봅니다. 

    - Lecture note:
        - 현업은 과학적 metric에 대한 이해가 떨어진다.
        - A/B Testing이 뭔가? 심지어는 Thompson sampling도 아는 듯 했지만 아니더라..
        - 메트릭의 부재. 호기심 천국으로의 지름길
        - Benefit에 대한 논란




### **Part III** (Week 7 - 9)

판별기를 만드는 경우, 가장 중요한 과정은 EDA 과정입니다. 이 과정을 통해 모형에 사용할 feature를 결정하는 것이죠. Feature를 결정한 후에 가장 전통적인 판별기로 GLM, SVM이 있습니다. 여기에 대해서 자세히 알아봅니다.

- Week 7
    - Target: 가설을 증명하기 위한 본격적인 EDA를 시작합니다. 모델링의 경우 다양한 파생변수에 대한 idea를 내보고, data handling을 토해 파생변수들을 만들어 봅니다. 현대의 data science는 일반론을 찾는 과정이 아닙니다. 예측력을 높이는 과정입니다. 상상하는 만큼 더 많은 파생변수들을 도출할 수 있고, 모형은 좀더 예측력이 올라갑니다. 상상은 또한 경험이 바탕이 되어야 할 수 있습니다. 그래서 현업의 경험이 중요한 것입니다.

    - Leture note:
        - Simpson's paradox
        - Divergence plot, violin plots
        - t-test, multiple comparison
        - 데이터 제작 오류



- Week 8
    - Target: Logistic regression에서 배울 수 있는 여러가지 통계적인 개념과 회귀진단에 대해서 알아보도록 하겠습니다.
    
    - Lecture note:
        - Logistic regression
        - odds, relative risk
        - Residual plots
        - Interaction vs correlation between varaibles
        - Confusion matrix - Precision, Recall etc.
        - Lift
        - 예측치는 probability로 해석이 가능한가?


- Week 9
    - Target: SVM에 대해서 배워보도록 합니다.

    - Lecture note:
        - SVM
        - Kernel methods



### **Part IV** (Week 10 - 12)

Machine learning 기법들에 대해서 정리하고, 이에 근거한 modeling을 진행해 봅니다.

- Week 10:
    - Target:

    - Lecture note:
        - Tree vs Regression
        - SVM
        - Bootstraping method
        - Bagging vs Boosting
        - Random Forest
        
        
        
- Week 11:
    - Target:
    
    - Lecture note:
        - XG Boost vs Gradient  vs Catboost


### **Part V** (Week 13 - 14)

- Week: 13-14