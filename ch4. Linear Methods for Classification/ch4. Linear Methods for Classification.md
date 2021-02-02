# Linear Methods for Classification

## 4.1. Introduction

해당 절에서는 linear method를 분류문제에 적용한 기법들에 대해서 알아보려고 한다. 해당 절에서 예측해야하는 변수 $G(x)$는 이산적인 $g$개의 변수를 가지고, input space를 classification에 따라 타겟 변수의 값별로 나눠보려고 한다.

분류해야하는 타겟 값에는 K개의 class가 있으며, k번째 타겟을 위한 선형 모델의 값은 $\hat{f}_k(x)=\hat{\beta}_{k0}+\hat{\beta}^T_{k}x$이다. class k와 l의 결정 경계는 $\hat{f}_k(x)=\hat{f}_l(x)$가 같아지는 지점들의 집합이고, 이는 $\{x:(\hat{\beta}_{k0}-\hat{\beta}_{l0})+(\hat{\beta}_{k}-\hat{\beta}_{l})^Tx=0\}$와 같이 나타낼 수 있다. 이로 인해, input space는 classification 된 공간으로 나눠질 수 있는 것이다. 

이러한 회귀적인 접근법은 각 클래스를 위해 모델된 discriminant functions $\delta_k(x)$로 각 discriminant function의 최대값을 가지고 x를 각 클래스로 분류한다. 사후 확률 $Pr(G=k|X=x)$로 모델링하는 모델 또한 위의 접근법에 속한다. $\delta_k(x)$ 또는 $Pr(G=k|X=x)$가 x에 대해 선형적이라면, 결정 경계 또한 선형이 될 것이다. 

따라서, 결정경계를 선형으로 하기 위해서는 $\delta_k(x)$ 또는 $Pr(G=k|X=x)$이 선형이 되도록 단조변환을 거쳐야 할 것이다. 

<div align="center">
<img src="imgs/eq_4_1.png" />
</div>

이를 $log[p/(1-p)]$에 대해 단조변환하게 되면, 

<div align="center">
<img src="imgs/eq_4_2.png" />
</div>

위의 식에서 결정경계는 4.2.의 우변의 수식을 0로 하는 값을 찾는 것이며, 이를 초평면에는 $\{x|\beta_0+\beta^Tx\}$로 정의한다. 이를 계산하는 방식에는 아래의 방법이 있다. 

* Linear Dicriminant Analysis
* Linear Logistic Regression

이 둘의 가장 중요한 차이는 훈련데이터셋에 linear function을 학습하는 방법이다. 

## 4.2. Linear Regression of and Indicator Matrix

결과 값으로 나온는 class들을 'indicator variable'이라고 하자. 만약 $g$가 K개의 클래스를 갖는다면, $k=1,...,K; Y_k$의 indicator가 있고 만약 특정 Row의 target G=k이면 1 아닌 경우에는 0의 값을 가진다. 이를 벡터 Y로 하여 $(Y_1,...,Y_K)$로 나타내면, Y는 N개의 train data를 가지는 경우에 0과 1로 구성된 NXK 행렬로 나타난다. 

<div align="center">
<img src="imgs/eq_4_3.png" />
</div>

Y가 K개수의 class만큼 늘었기 때문에, 추정해야하는 coefficieint Beta의 값 또한 (p+1) X K 행렬로 나타난다. 

* 4.3.의 식으로 추정한 estimator에 따른 수식은 $\hat{f}(X)^T = (1,x^T)\hat{B}$
* 각 row 별로 k 클래스의 $\hat{f}(X)^T$ 중에 가장 큰 값을 예측값이라고 하면, 예측한 결과값을 아래와 같이 나타낼 수 있다. 

    $\hat{G}(x)=argmax_{k\in{g}}\hat{f}_k(x)$ (4.4)

조건부 기대값을 추정하는 회귀방법으로 문제를 바라본다면, 랜덤변수 $Y_k$에 대해 $E(Y_k|X=x) = Pr(G=k|X=x)$이며, 이 또한 적절한 목표가 될 수 있다. 이 방식의 논점은 rigid linear regression model보다 더 우수한지 질문에 답하는 것이다. 

이는 각 예측치에서 얻는 1값이 무조건 하나인 $\sum_{k\in{g}}\hat{f}_k(x)=1$로 인해 쉽게 검증할 수 있다. 그러나 결과값이 음수이거나 1개 이상의 1이 있다면 다른 우수한 방법이 있을 것이다.

만약 인풋의 기저의 확장인 h(X)위에 선형 회귀를 할 수 있다면, 확률의 추정치들을 지속적으로 이끌어 낼 수 있다. 훈련 데이터의 개수 N이 늘어날수록, 이러한 기저함수위에 선형회귀가 조건부 평균에 접근할 수 있도록 더 많은 기저 항목들을 포함해야 한다. 

간단한 방법으로, 각 클래스에 대해 K X K target 단위행렬을 생성하고 이에서 k번째 컬럼값인 tk를 활용하는 것이다. prediction의 목표는 관측치에서 적절한 target의 추정치를 만들어 내는 것이다. 이는 $g_i=k$이면 $y_i=t_k$가 된다. 

* fit the linear model by least square

    <div align="center">
    <img src="imgs/eq_4_5.png" />
    </div>

fit된 vector로 새 관측값을 분류하게 되면,

<div align="center">
<img src="imgs/eq_4_6.png" />
</div>

regression approach의 심각한 단점은 예측하고자하는 클래스 값이 3개 이상인 경우이다. 회귀 모델의 엄격성으로 인해 클래스는 상호간에 겹쳐질 수 있다. 

<div align="center">
<img src="imgs/fig_4_3.png" />
</div>

각각의 3클래스의 데이터를 x축에 투영하면 각각 3개의 중심점을 가진다. 각각의 클래스들의 respond를 Y1,Y2,Y3로 나타낸다. 좌측 이미지의 2번째 클래스의 선은 수평이고 fitted value는 never dominant이다. 그래서 class 2값은 1 또는 3의 값으로 매핑될 것이다. 우측 이미지는 2차식의 회귀선이다. 이는 선형 회귀선과 다르게 문제를 해결할 수 있다. 만약 우측 이미지에서 4개의 클래스로 변화하는 경우 optimal한 경우를 찾기 위해서는 3차식의 변형 또한 고려해야 할 것이다. 

그래서 class의 수 K가 3이상이라면 K-1 수준의 polynomial feature를 생성해야 이를 해결할 수 있을 것이다. 

이를 p차원의 input 값, K개의 class로 일반화하게 되면, K-1, $O(p^{K-1})$의 수준과 복잡도를 해당 문제를 해결하는데 소요된다. 

### 4.3. Linear Discriminant Analysis

#### References

<a href="https://ratsgo.github.io/machine%20learning/2017/03/21/LDA/">https://ratsgo.github.io/machine%20learning/2017/03/21/LDA/</a>

<a href="https://www.youtube.com/watch?v=geIlsP8aPvg">https://www.youtube.com/watch?v=geIlsP8aPvg</a>


판별분석(Discriminant Analysis): 두 개 이상의 모집단에서 추출된 표본들이 지니고 있는 정보(표본들이 Gaussian Distribution을 따른 다던지...)를 이용하여 이 표본들이 어느 모집단에서 추출된 것인지를 결정해 줄 수 있는 기준을 찾는 분석법

특정 축으로 샘플들을 투영했을 때, Class들을 잘 분류할 수 있는가?

* 판별 변수(X): 독립 변수 중 판별력이 높은 변수. 이를 선택하는데 있어, '판별기여도 + 다른 독립변수와의 상관관계'를 고려해야 한다. if 독립변수간의 상관관계가 높은경우, 하나를 선택하고 이와 상관관계가 적은 독립변수를 선택하여 효과적인 판별 함수를 설계.

* 판별 함수: 판별 함수를 통해 각 개체들이 소속집단에 얼마나 잘 판별되는가에 대한 **판별력**(판별점수)을 측정하고 새로운 대상을 어느집단으로 분류할 것이냐를 예측하는데 주요 목적이 있다. 

**판별 점수의 집단간 변동과 집단내 변동의 배율을 최대화하는 판별함수를 도출해야 한다.**

* 표본의 크기
    * 전체 표본의 크기는 독립변수의 개수보다 3배(최소 2배)이상 되어야 한다.
    * 종속 변수의 집단 각각의 표본의 크기 중 최소 크기가 독립변수의 개수보다 커야한다. (판별력을 좌우하는 것은 가장 적은 집단의 표본수이기 때문이다.)

LDA(Linear Discriminant Analysis)

* 가정
    * 각 클래스 집단은 정규분포 형태의 확률분포를 가짐
    * 각 클래스 집단은 비슷한 형태의 공분산 구조를 가진다.

** 공분산 행렬이 2개의 분포에 대해 이와 같이 나타난다면, 
$\sum=\begin{bmatrix}\sigma^2\ 0 \\0\ \sigma^2\end{bmatrix}$ 각 대각행렬의 값은 타원이 어떤 형태를 띄는지 결정하고 나머지 0의 element는 어떻게 회전시킬지를 결정한다. 

<div align="center">
<p>LDA를 적용할 수 있는 사례</p>
<img src="imgs/fig_4_5.png" />
</div>

* LDA&QDA의 기능(즉, DA의 기능)

    <div align="center">
    <p>LDA와 차원 축소</p>
    <img src="imgs/fig_4_9.png" />
    </div>

    \- 판별과 차원 축소의 기능을 가진다. 2-dimensional(X=x1,x2) problem에 분류를 수행하면 차원을 축소하는 기능과 유사하다. 왜냐하면 LDA를 수행함에 있어 하나의 축에 사영함으로써 각 클래스를 구분하기 때문이다. 따라서, 데이터가 각 축에 사영하면서 x1과 x2데이터는 하나의 차원으로 표현이 된다. 즉 판별력도 높이고 차원 축소의 기능도 나타낸다. 


* 결정경계의 특징

    * projection 축에 직교하는 축
    * 정 사영은 두 분포의 특징이 아래의 목표를 달성해야 한다.
        * 각 클래스 집단의 평균의 차이가 큰 지점을 결정 경계로 지정 (각 샘플의 분포의 평균의 차이)
        * 각 클래스 잡단의 분산이 작은 지점을 결정 경계로 지정 (결정 경계를 기준으로 타원의 퍼진 정도)
        * 즉, 분산 대비 평균의 차이를 극대화하는 결정 경계를 찾고자 하는것. ($\frac{diff\ of\ mean}{variance}$)

* 문제점

    \- 공분산 구조가 많이 다른 경우는 반영할 수 없다. 이를 위해 QDA(이차판별분석법)를 사용하여 해결

* QDA

    <div align="center">
    <p>Dicision Boundary: LDA with Data augmentation / QDA</p>
    <img src="imgs/fig_4_6.png" />
    </div>

    \- 공통 공분산 구조 제약을 충족할 수 없는 경우 사용한다. if 선형 결정 경계의 경우에는 이를 올바르게 구분할 수 없다. 그러나 LDA는 변수의 제곱을 추가적인 변수로 사용하여 보완할 수 있다. 

    그러나 QDA는 서로 다른 공분산 데이터를 분류하기 위해 샘플을 많이 필요로 한다. 그런데, 설명변수의 개수가 많을 경우 추정해야하는 모수(coefficients named beta)가 많아져서 연산량이 크다. 


최적의 분류를 위해서는 클래스별 사후확률인 $Pr(G|X)$를 알 필요가 있음을 이전에 알아봤었다. 

$Pr(G|X)$을 계산하기 위해서는 아래의 두 element들이 필요하고

* $f_k(x)$: G가 k인 경우에 X의 class-conditional density
* $\pi_k$: 클래스 k의 사전확률 

이를 통해 수식 4.7을 도출할 수 있다. 

<div align="center">
<img src="imgs/eq_4_7.png" />
</div>

이를 구하기 위해서는 $f_k(x)$를 알아내야만 하고, 이를 찾는것이 $Pr(G=k|X=x)$를 찾는것과 동일하다고 할 수 있다. 

그렇다면, 각각의 class들의 분포가 다변량정규분포(multivariate Gaussian[Normal Distribution])를 따른다고 가정하자.

그렇다면 LDA는 각각의 클래스들의 샘플이 공통적인 공분산 matrix $\sum_k=\sum\forall{k}$를 가정하는 상황에서 활용된다.

<div align="center">
<img src="imgs/eq_4_9.png" />
</div>

임의의 2개의 클래스를 비교하기 위한 수식은 x에 대해 선형적인 방정식으로 나타난다. 이는 k와 l의 결정경계가 p-dimensional한 초평면위에 x에 대해 선형적인 형태로 나타남을 의미한다. 만약 input dataset을 class 1, class 2 등등에 대해 분류하더라도 임의의 초평면에 의해 분리될 것이다. 

<div align="center">
<img src="imgs/fig_4_5.png" />
</div>

위의 예시는 3개의 클래스가 각각이 정규분포를 따르고 동일한 공분산형태를 가지는 경우 결정경계를 선형으로 하여 문제를 해결하는 것을 나타낸 것이다. 

<div align="center">
<img src="imgs/eq_4_10.png" />
</div>

실제로는 정규분포의 파라미터를 모르기 때문에 훈련데이터로부터 아래의 세가지를 추정해야 한다.

<div align="center">
<img src="imgs/eq_4_10-1.png" />
</div>

두개의 class를 위해 linear regression으로 결정경계를 결정하는 방법론은 eq 4.5.이고, LDA 방식으로 결정하는 방식은 아래와 같다.

<div align="center">
<img src="imgs/eq_4_11.png" />
</div>

least sqaure방식을 통한 LDA direction을 도출하는 것은 Gaussian 가정을 하지 않기 때문에 Gaussian 하지 않는 일반적인 데이터에도 활용될 수 있다.

만약 LDA에서 2개의 클래스 이상을 분류해야 한다면 더이상 class indicator matrix의 linear regression한 방식과 동일하지 않으며, LDA는 위에서 나타난 masking problem에 대처할 수 있게 된다.

만약 분류하는데 있어서 class들의 공분산 matrix가 동일하지 않는다면 식 4.9의 편리하게 도출된 식은 사용할 수 없다. 우리는 이를 위해 QDA(quadratic discriminant function)을 사용해야 한다. 

<div align="center">
<img src="imgs/eq_4_12.png" />
</div>

class k와 l간의 결정경계는 2차 방정식인 $\{x:\delta_k(x)=\delta_l(x)\}$로 나타낼 수 있다. 

<div align="center">
<img src="imgs/fig_4_6.png" />
</div>

위의 그림은 결정경계가 2차방정식 즉 곡선의 형태를 나타낸다고 가정했을 때, 접근하는 두 가지 방법론을 나타낸다. 좌측은 feature를 2차식의 형태로 polynomial feature를 만들어 반영하였고, 우측은 QDA를 적용했을 때의 결정경계를 나타낸다. 

QDA의 추정치는 LDA와 각각의 클래스들에 대한 공분산행렬이 다르다는 것만들 제외하면 유사하다. 그러나 차원(컬럼)의 수 p가 늘어나게 되면 추정해야하는 파라미터 또한 엄청나게 증가하게 된다. 