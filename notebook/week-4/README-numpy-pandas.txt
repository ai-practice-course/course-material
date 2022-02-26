- 이번 시간에는 python을 활용해서 데이터를 핸들링하는 방법들에 대해 실습해 봅니다.
    - Python은 데이터 분석과 모델 서빙을 동시에 할 수 있는 가장 popular한 대안이다.
    - 기존의 R이 통계 community에서 발전한 툴이라고 한다면, python은 cs에서 데이터분석이 본격화되면서 부터 R의 대안으로 자리를 잡아간다.
    - Tensorflow가 세상에 모습을 드러냈던 17년 파이썬은 그당시에 데이터 분석의 차세대 언어로 비교되던, scalar, julia, r을 넘어서서 가장 강력한 데이터분석의 툴의 자리에 오릅니다.
    - 여전히 데이터 분석에 꼭 필요한 시각화의 편의성은 R이 ggplot과 R이 제공하는 독특한 chain 문법에 힘입어 더욱 좋다라는 평가를 얻고 있다.
    - 하지만, R의 시각화가 데이터 분석에 특화되어 있고, shiny 등의 library를 활용하여 구현한 대쉬보드들이 production level에서 사용하기에는 모자라고, interactive한 plot을 그리기에도 부족한 특징을 지닌다.
    - 반면 python은 시각화의 편의성은 좀 떨어지지만, 점점 편해지고 있고, 완성도에서도 production level에 준하는 결과를 도출하는 library도 점점 발전하고 있다.
    - 이외에 커뮤니티도 R의 커뮤니티를 압도하고 있어, 현재 존재하는 가장 powerful한 데이터분석을 위한 언어라고 생각이 된다.

    - Python으로 개발하는 것과는 다르게 데이터 분석을 하기 위해서는 interactive한 환경이 중요하다. 개발을 용이하게 하기 위한 IDE가 있는 것과 마찬가지로 데이터 분석 영역에서도 데이터분석에 특화되어 있는 일종의 IDE가 존재한다.
    - Pycharm, Jupyter, Spyder,등이 있다. (https://ichi.pro/ko/4-deiteo-gwahagjaleul-wihan-choegoui-python-ide-225095697317027)
        - 스파이더: 
            - Scientific Python 개발 환경 또는 Spyder 는 무료 오픈 소스 Python IDE입니다. 
            - Anaconda를 설치하는 경우 설치 가능한 소프트웨어 중 하나는 Spyder입니다.
            - Spyder는 데이터 과학을 위해 특별히 제작 된 IDE입니다. 
            - 이전에 RStudio IDE에서 왔고 Python에서 이와 유사한 것을 찾으려고한다면 Spyder IDE를 사용하고 싶습니다.
        - 토니:
            -  Thonny는 프로그래밍 학습 및 교육을 위해 개발 된 IDE입니다. The University of Tartu에서 개발되었으며 Bitbucket 저장소에서 무료로 다운로드 할 수 있습니다 .
            - Thonny는 어시스턴트 페이지의 도움으로 단계별 문장과 표현을 보여줄 수 있기 때문에 초보자에게 매우 좋습니다. Thonny는 또한 지역 변수가있는 새 창이 열리고 코드가 기본 코드와 별도로 표시되는 멋진 편집 기능을 제공합니다.
            3. 원자
        - Atom:
            -GitHub에서 개발 한 오픈 소스 IDE입니다. 개발자들은 Atom을 통해 사용자가 타사 패키지와 테마를 설치할 수 있기 때문에이를 "21 세기를위한 해킹 가능한 텍스트 편집기"라고 부릅니다.
            - 원하는대로 편집기의 기능과 모양을 사용자 정의 할 수 있으므로 매우 다재다능한 IDE입니다. 또한 Atom은 초보자에게 친숙하고 숙련 된 개발자를 위해 고도로 사용자 정의 할 수 있습니다.
        - PyCharm은:
            - 유명한 Java IDE 인 IntelliJ IDEA를 개발 한 JetBrain이 만든 IDE입니다.
            - PyCharm은 이미 다른 IDE를 사용한 개발 또는 프로그래밍 언어 경험이있는 사용자에게 적합합니다. 
            - Anaconda 배포판과 마찬가지로 Pycharm은 NumPy 및 Matplotlib와 같은 도구 및 라이브러리를 통합하여 다양한 뷰어 및 대화 형 플롯으로 작업 할 수 있습니다. 
            - 또한 앞서 논의한 Atom IDE와 마찬가지로 PyCharm은 버전 제어 통합 및 디버깅 기능을 제공합니다.
        - Visual Studio Code:
            - MS에서 제작한 오픈 소스 IDE
            - Atom과 같은 해킹 가능한 텍스트 편집기로 데이터 분석을 위한 jupyter notebook을 위한 plug-in을 제공하여, 데이터 분석 영역에서도 최적화된 개발용 IDE
            - 거대 community의 support를 바탕으로 다양한 언어를 지원하여, production level의 code를 build 하는 데 최적의 IDE



    - 우리는 docker image를 이미 구축해 뒀기 때문에 (혹은 국민대의 분석 플랫폼이 있기 때문에), 약속된 url로 접속을 하기만 하면 됨//ichi.pro/ko/4

    - 이제 환경이 준비되었으므로, 데이터 분석을 하기 위해 library가 필요함
        - 데이터 핸들링을 위한 library -> numpy, Pandas
        - 데이터 분석을 위한 library -> sklearn, statsmodel, tensorflow, pytorch, caffe, mxnet
        - 시각화를 위한 library -> matplotlib, bokeh

    - 위의 라이브러리 외에도 수많은 라이브러리가 존재하나, 위의 라이브러리는 모든 분석에서 사용되는 핵심 중의 핵심이라고 생각하면 됨


    - Numpy: (ref: https://laboputer.github.io/machine-learning/2020/04/25/numpy-quickstart/#:~:text=%EB%84%98%ED%8C%8C%EC%9D%B4(Numpy)%EB%8A%94%20Python,%EB%B9%A0%EB%A5%B8%20%EC%86%8D%EB%8F%84%EB%A1%9C%20%EC%88%98%ED%96%89%EB%90%A9%EB%8B%88%EB%8B%A4.)
        - NumPy는 행렬이나 일반적으로 대규모 다차원 배열을 쉽게 처리 할 수 있도록 지원하는 파이썬의 라이브러리입니다.
        - 프로그래밍 하기 어려운 C, C++, FORTRAN 등의 언어에 비하면, NumPy로는 편리하게 수치해석을 실행할 수 있습니다. 
        - 게다가 Numpy 내부 상당부분은 C나 포트란으로 작성되어 실행 속도도 빠른 편입니다. 기본적으로 array라는 자료를 생성하고 이를 바탕으로 색인, 처리, 연산 등을 하는 기능을 수행합니다.
        - Numpy에서 오브젝트는 동차(Homogeneous) 다차원 배열이라고 하는데 Homogeneous하다
        - Numpy에서 배열은 ndarray 또는 array라고도 부릅니다. Numpy.array와 Python.array는 다릅니다
        - 행벡터 기반
                ndarray.shape : 배열의 각 축(axis)의 크기
                ndarray.ndim : 축의 개수(Dimension)
                ndarray.dtype : 각 요소(Element)의 타입
                ndarray.itemsize : 각 요소(Element)의 타입의 bytes 크기
                ndarray.size : 전체 요소(Element)의 개수
        - np.array()를 이용하여 Python에서 사용하는 Tuple(튜플)이나 List(리스트)를 입력으로 numpy.ndarray를 만들 수 있습니다.
        ```
        a = np.array([2,3,4])
        print(a)
        # [2 3 4]
        print(a.dtype)
        # int64

        b = np.array([1.2, 3.5, 5.1])
        print(b.dtype)
        # float64
        ```
        - dtype = complex으로 복소수 값도 생성할 수 있습니다.
        - np.zeros(), np.ones(), np.empty()
        - np.arange() 와 np.linspace()를 이용하여 연속적인 데이터도 쉽게 생성
        - np.ndarray.reshape()을 통해 데이터는 그대로 유지한 채 차원을 쉽게 변경해줍니다.
        - 2차원 배열을 행렬이라고 생각했을 때 행렬의 여러가지 곱셈이 있습니다.

            - * : 각각의 원소끼리 곱셈 (Elementwise product, Hadamard product)
            - @ : 행렬 곱셈 (Matrix product)
            - .dot() : 행렬 내적 (dot product)
        - aggregation
            - .sum(): 모든 요소의 합
            - .min(): 모든 요소 중 최소값
            - .max(): 모든 요소 중 최대값
            - .argmax(): 모든 요소 중 최대값의 인덱스
            - .cumsum(): 모든 요소의 누적합
        - index 번호를 가지고 함수 정의 생성
            -   ```   
                    def f(x,y):
                        return 10*x+y
        
                    b = np.fromfunction(f, (5,4), dtype=int)
                    print(b)
                ````
        - ...은 차원이 너무 많을 때 실수를 줄여줄 수 있습니다. 만약 x가 5차원이라고 할 때 아래 처럼 표현할 수 있습니다.
        - 다차원의 배열을 for문을 적용하면 axis=0 기준으로 적용됩니다. 만약 다차원 배열의 모든 원소를 for문 적용하고 싶은 경우 .reshape()을 해도 되지만, .flat을 이용할 수 있습니다.
        - Shape 변경 (Shape Manipulation)
            - .ravel()은 1차원으로, 
            - .reshape()는 지정한 차원으로, 
            - .T는 전치(Transpose) 변환을 할 수 있습니다. 하지만 데이터 원본은 변경시키지 않고 복사하여 연산한 결과가 return 됩니다.
                ```
                a = np.random.random((3,4))
                print(a.reshape(2, 6))
                a.T
                a.ravel().shape
                a.flatten().shape
                ```
        - 데이터 쌓기 - Stacking together different arrays
            - np.vstack() 와 np.hstack()를 통해 데이터를 합칠 수 있습니다.\
            - np.vstack(): axis=0 기준으로 쌓음
            - np.hstack(): axis=1 기준으로 쌓음
            ```
            a = np.floor(10 * np.random.random((2,3,4)))
            b = np.floor(10 * np.random.random((2,3,4)))
            np.vstack((a, b)).shape
            np.hstack((a, b)).shape
            ```

        - 데이터 쪼개기 - Splitting one array into several smaller ones
            - np.hsplit()을 통해 숫자1개가 들어갈 경우 X개로 등분, 리스트로 넣을 경우 axis=1 기준 인덱스로 데이터를 분할할 수 있습니다.
    
    
        - 데이터 복사가
            - No Copy at All
                - 아래와 같이 np.array를 변수에 넣는다고 해서 복사가 되지 않습니다. 레퍼런스를 참조할 뿐입니다. 
                - id()를 통해 주소값을 반환해서 확인할 수 있습니다.
                    ```
                    a = np.array([[ 0,  1,  2,  3],
                                [ 4,  5,  6,  7],
                                [ 8,  9, 10, 11]])
                    b = a
                    print(b is a)
                    print(id(a))
                    print(id(b))
                    ```
            - View or Shallow Copy
                - view()를 통해 Shallow Copy를 할 수 있습니다. 
                - Shallow Copy, Deep Copy의 개념을 이해하고 있다면 이것으로 이해하실 수 있을 것입니다. 
                - Numpy 관점에서 쉽게 설명드리면 view()는 실제로 데이터가 복사된다기 보다는 데이터 각각의 참조값이 복사됩니다. 
                - c와 a의 참조값은 다르지만 각각의 데이터 참조값이 복사됐다는 의미입니다. 
                - 따라서 a와 c는 다르지만 c[0, 4]는 a[0, 4]는 같은 참조값을 보고 있어 a가 변경되는 것을 확인할 수 있습니다. 
                - 마찬가지로 s에 a를 슬라이싱하여 데이터를 가져가도 s를 변경하면 a가 변경됩니다.
                    ```
                    c = a.view()
                    print(c is a)
                    c = c.reshape((2, 6))
                    print(a.shape)
                    c[0, 4] = 1234
                    print(a)
                    s = a[ : , 1:3]
                    s[:] = 10
                    print(a)
                    ```
            - 깊은복사 - Deep copy
                - .copy()를 이용하면 Deep Copy를 할 수 있습니다. 
                - 즉 a와 d의 참조값 뿐만 아니라 a의 각각의 데이터 전부가 새로운 객체로 생성됩니다.
                - Python의 del 키워드를 이용하면 메모리를 반환할 수 있습니다.
                    ```
                    d = a.copy()
                    print(d is a)
                    d[0,0] = 9999
                    print(a)
                    a = np.arange(int(1e8))
                    b = a[:100].copy()
                    del a 
                    print(a)
                    ```
    

            - 브로드캐스팅 (Broadcasting rules)
                - Numpy에서 Broadcasting(브로드캐스팅)은 반드시 이해하고 있어야 하는 개념이어서 그림과 함께 설명하겠습니다. 
                - 브로드 캐스팅은 단순하게 편리성을 위해 Shape가 다른 np.narray 끼리 연산을 지원해주기 위함입니다. 
                - 데이터 계산 시 자주 등장하는 상황인데, 이것이 없다면 Shape를 맞춰야하는 번거로움이 생기게 되는데 이 개념을 이해하면 잘 활용하실 수 있습니다. 
                - 웬만하면 Shape를 같게 맞춘 후에 연산하는 것이 바람직하지만 이 글에서도 알게 모르게 사용하고 있었습니다.


                    ```
                    print(np.arange(4) * 2)
                    print(np.ones((3,4)) * np.arange(4))
                    print(np.arange(3).reshape((3,1)) * np.arange(3))
                    ```

            - 인덱스 배열로 인덱싱하기 - Indexing with Arrays of Indices
                - 인덱스를 가진 배열로 인덱싱을 할 수 있습니다.
                    ```
                    a = np.arange(12)**2
                    print(a)

                    i = np.array([1, 1, 3, 8, 5])
                    print(a[i])

                    j = np.array([[3, 4], [9, 7]])
                    print(a[j])
                    ```

            - Bool로 인덱싱하기 - Indexing with Boolean Arrays
                - Bool 타입을 가진 값들로도 인덱싱이 가능합니다.
                - Mandelbrot Set이라고 하는 프랙탈 모형이 있습니다. 이 값은 복소수의 집합으로 정의된 것인데 이런 것들도 구현이 가능함을 볼 수 있습니다.

                    ````
                    import numpy as np
                    import matplotlib.pyplot as plt
                    def mandelbrot( h,w, maxit=20 ):
                        """Returns an image of the Mandelbrot fractal of size (h,w)."""
                        y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
                        c = x+y*1j
                        z = c
                        divtime = maxit + np.zeros(z.shape, dtype=int)

                        for i in range(maxit):
                            z = z**2 + c
                            diverge = z*np.conj(z) > 2**2         # who is diverging
                            div_now = diverge & (divtime==maxit)  # who is diverging now
                            divtime[div_now] = i                  # note when
                            z[diverge] = 2                        # avoid diverging too much

                        return divtime

                    plt.imshow(mandelbrot(400,400))
                    ```

            -선형대수
                ```
                a = np.array([[1.0, 2.0], [3.0, 4.0]])
                print(a)

                a.transpose()
                np.linalg.inv(a)
                
                # unit 2x2 matrix; "eye" represents "I"
                u = np.eye(2) 
                
                # matrix product
                j = np.array([[0.0, -1.0], [1.0, 0.0]])
                j @ j
                
                # trace
                np.trace(u) 
                y = np.array([[5.], [7.]])
                np.linalg.solve(a, y)
                
                np.linalg.eig(j)
                ```





    - Pandas: Pandas
        - 판다스(Pandas)는 Python에서 DB처럼 테이블 형식의 데이터를 쉽게 처리할 수 있는 라이브러리 입니다. 
        - 데이터가 테이블 형식(DB Table, csv 등)으로 이루어진 경우가 많아 데이터 분석 시 자주 사용하게 될 Python 패키지입니다.
        - Object Creation
            - Series: 1차원 데이터와 각 데이터의 위치정보를 담는 인덱스로 구성
            - DataFrame: 2차원 데이터와 인덱스, 컬럼으로 구성
            - DataFrame에서 하나의 컬럼만 가지고 있는 것이 Series입니다.
            - 딕셔너리 형식으로도 DataFrame을 만들 수 있습니다
                ```
                s = pd.Series([1,3,5,np.nan,6,8])
                dates = pd.date_range('20130101', periods=6)
                df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
                df2 = pd.DataFrame({'A':1.,
                   'B':pd.Timestamp('20130102'),
                   'C':pd.Series(1,index=list(range(4)),dtype='float32'),
                   'D':np.array([3]*4,dtype='int32'),
                   'E':pd.Categorical(["test","train","test","train"]),
                   'F':'foo'})
                df2.dtypes
                ```
        - Data view
            - DataFrame은 head(), tail()의 함수로 처음과 끝의 일부 데이터만 살짝 볼 수 있습니다. 
            - 데이터가 큰 경우에 데이터가 어떤식으로 구성되어 있는지 확인할 때 자주 사용합니다.
                ```
                df.index
                df.columns
                df.values
                ```
            - DataFrame의 to_numpy()를 이용하여 인덱스와 컬럼을 제외한 2차원 데이터만을 numpy의 형식으로 반환해줍니다. 사실은 .values와 동일합니다.
                ```
                df.to_numpy()
                ```
            - DataFrame의 desribe()를 통해 각 컬럼의 통계적인 수치를 요약
                ```
                df.describe()
                ```
            - DataFrame의 .T 속성은 values를 Transpose한 결과를 보여줍니다. Transpose는 인덱스를 컬럼으로, 컬럼을 인덱스로 변경하여 보여주는 것입니다.
            - DataFrame의 sort_index()를 통해 인덱스 또는 컬럼의 이름으로 정렬을 할 수도 있습니다.
                - axis: 축 기준 정보 (0: 인덱스 기준, 1: 컬럼 기준)
                - ascending: 정렬 방식 (false : 내림차순, true: 오름차순)
        
            - DataFrame의 sort_values() 를 이용하여 value 값 기준으로 정렬할 수도 있습니다.
                - by: 데이터 정렬에 기준이 되는 컬럼

        - Data Selection
            - 데이터 가져오기 - 컬럼을 기준으로 데이터를 가져올 수 있습니다
            - []을 이용하여 특정 범위의 행을 슬라이싱할 수 있습니다.
                ```
                df[0:3]
                df['20130102':'20130104']
                ```
            - 이름으로 데이터 가져오기 - Selection by label
                - 이름(Label)로 가져오는 것은 DataFrame의 .loc 속성을 이용합니다.
                - .loc은 2차원으로 구성되어 있습니다. .loc[인덱스명, 컬럼명] 형식으로 접근가능 합니다. 만약 인덱스명만 입력하면 행의 값으로 결과가 나옵니다. 또한 인덱스명, 컬럼명을 선택할때 리스트 형식으로 멀티인덱싱이 가능합니다.
                - 인덱스명, 컬럼명을 하나씩 선택하면 스칼라값을 가져올 수 있습니다.
                    ```
                    df.loc[dates[0]]
                    df.loc[:,['A','B']]
                    df.loc['20130102':'20130104',['A','B']]
                    df.loc['20130102',['A','B']]
                    df.loc[dates[0],'A']
                    ```

            - 인덱스로 데이터 가져오기 - Selection by Position
                - 여기서 말하는 인덱스는 위치(숫자) 정보를 말합니다.
                - DataFrame의 .iloc 속성을 이용합니다.
                - .iloc도 2차원 형태로 구성되어 있어 1번째 인덱스는 행의 번호를, 2번째 인덱스는 컬럼의 번호를 의미합니다. 마찬가지로 멀티인덱싱도 가능합니다.
                    ```
                    df.iloc[3:5,0:2]
                    df.iloc[[1,2,4],[0,2]]
                    df.iloc[1:3,:]
                    df.iloc[:,1:3]
                    df.iat[1,1]
                    ```

            - 조건으로 가져오기 - Boolean Indexing
                - 하나의 컬럼의 값에 따라 행들을 선택할 수 있습니다.
                    ```
                    df[df['A'] > 0]
                    df[df > 0]
                    df2 = df.copy()
                    # 새로운 컬럼 E에 값을 넣습니다.
                    df2['E'] = ['one','one', 'two','three','four','three']
                    df2[df2['E'].isin(['two','four'])]
                    ```

            - 데이터 변경하기 - Setting
                - 데이터 선택하기와 같은 속성 at, iat, loc, iloc 등을 그대로 사용하면 됩니다.
                - 조건문(where)으로 선택하여 데이터를 변경할 수도 있습니다.
                    ```
                    s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102',periods=6))
                    df['F'] = s1
                    df.at[dates[0],'A'] = 0
                    df.iat[0,1] = 0
                    df.loc[:,'D'] = np.array([5] * len(df))
                    ```

            - 결측 데이터 (Missing Data)
                - 데이터를 다루다보면 값이 없는 경우가 자주 생깁니다. 데이터가 없는 것을 결측 데이터라고 합니다. 
                - 판다스에서는 이러한 값이 NaN 으로 표현됩니다. 기본적으로 결측 데이터가 있는 경우에는 연산에 포함되지 않습니다.
                - DataFrame의 dropna()를 통해 결측데이터를 삭제(drop)할 수 있습니다. how='any'는 값들 중 하나라도 NaN인 경우 삭제입니다. how='all'은 전체가 NaN인 경우 삭제입니다.
                - DataFrame의 fillna()를 통해 결측데이터에 값을 넣을 수도 있습니다.
                - pd.isnull()을 통해 결측데이터 여부를 Boolean으로 가져올 수 있습니다.
                ```
                    df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
                    df1.loc[dates[0]:dates[1],'E'] = 1
                    df1.dropna(how='any')
                    df1.fillna(value=5)
                    pd.isnull(df1)
                    ```

            - 데이터 연산 (Operations)
                - 일반적으로 결측데이터는 빼고 계산됩니다.
                - 만약 다른 차원의 오브젝트들 간 연산이 필요한 경우 축만 맞춰진다면 자동으로 연산을 수행합니다.
                - 데이터에 대해 정의된 함수들이나 lamdba 식을 이용하여 새로운 함수도 적용할 수 있습니다
                    ```
                    df.mean()
                    df.mean(1)
                    # 데이터 시프트 연산 (2개씩 밀립니다.)
                    s = pd.Series([1,3,5,np.nan,6,8], index=dates).shift(2)
                    df.sub(s, axis='index')
                    df.apply(np.cumsum)
                    df.apply(lambda x: x.max() - x.min())

                    s = pd.Series(np.random.randint(0, 7, size=10))
                    s.value_counts()

                    ```
                - 문자열 처리 - String Methods
                    - Series에서 문자열 관련된 함수들은 .str 속성에 포함되어 있습니다.
                    - str.lower()를 통해 문자를 소문자로 변경할 수 있습니다.
                    ```
                        s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
                        s.str.lower()
                    ```

            - 데이터 합치기 (Merge)
                - 판다스는 Series와 DataFrame 간에 쉽게 데이터를 합칠 수 있도록 join과 merge와 같은 연산을 제공합니다.
                - concat()을 이용하여 이어붙이는 연산(Concatenating)을 할 수 있습니다.
                    ```
                    df = pd.DataFrame(np.random.randn(10, 4))
                    pieces = [df[:3], df[3:7], df[7:]]
                    pd.concat(pieces)
                    ```
                - SQL에서 자주 사용하는 join 연산도 제공됩니다.
                    ```
                    left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
                    
                    right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
                    
                    pd.merge(left, right, on='key')
                    
                    ```
            - 그룹화 (Grouping)
                - group by에 관련된 내용은 아래와 같은 과정을 말합니다.
                - Spltting : 특정 기준으로 데이터 나누기
                - applying : 각 그룹에 함수를 독립적으로 적용시키는 것
                - Combining : 결과를 데이터 구조로 저장하는 것

            - 데이터 구조 변경하기 (Reshaping)
                - Stack
                - DataFrame의 stack()은 모든 데이터들을 인덱스 레벨로 변형합니다. 이를 압축(compresses)한다고 표현합니다.
                ```
                tuples = list(zip(*[['bar', 'bar', 'baz', 'baz', 'foo', 'foo', 'qux', 'qux'],
                ['one', 'two', 'one', 'two', 'one', 'two', 'one', 'two']]))
                # 각 리스트에서 첫번째를 'first'로 두번째를 'second'로 멀티인덱스를 만듭니다.
                index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
                # 두 개의 컬럼을 생성하고 랜덤값을 부여합니다.
                df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
                stacked = df2.stack()
                ````
                - unstack() 을 통해 “stacked”된 DataFrame 이나 Series 를 원래 형태로 되돌릴 수 있습니다. 되돌리는(압축 해제) 것의 레벨을 정할 수 있습니다.





    - Target: 분석 대상 데이터를 불러오는 방법과, 가장 기본적으로 파악해야 할 데이터를 느끼는 방법에 대해서 이해합니다. 이는 주로 간단한 EDA를 통해 얻어집니다. EDA를 위한 기초적인 library 조작법에 대해 익힙니다.
    - Lecture note:
        - Jupyter lab: DS의 기본
        - Pandas, Numpy에 대한 이해
        - Graph library에 대한 이해 - Interactive plot