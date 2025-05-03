---
layout: post
title:  "Deep Learning Study"
crawlertitle:  "Deep Learning Study"
summary: "Study post"
date:  25-05-03 23:09:47 +0700
categories: posts
tags: "DeepLearning"
author: "wonwoo"
---
##### Monte Carlo Algorithm
---
<script type="text/javascript"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>


지금까지는 환경모델이 알려진 문제를 다루었다. 에이전트의 행동에 따른 다음 상태(위치)와 보상이 명확했었고, 상태 전이확률 , 보상함수를 이용할 수 있었다.
환경모델이 알려진 문제에서는 에이전트 측에서 '상태,행동,보상'의 전이를 시뮬레이션할 수 있었다.

그러나 현실에는 환경 모델을 알 수 없는 문제가 많다. '상품 재고 관리'를 예시로 든다면, 
{% highlight js %}
이문제에서 '상품이 얼마나 팔릴 것인가'가 환경의 상태 전이확률에 해당한다.그런데 
상품의 판매량은 여러요인이 복잡하게 얽혀 결정되기 떄문에 완벽하게 알아내기가 
현실적으로 불가능하다. 또한 상태전이 확률을 이론적으로 알 수 있더라도 
계산량이 너무 많은 경우가 허다하다.
{% endhighlight %}

#### 몬테카를로법

- 에피소드(시작 → 종료까지 한 게임/한 시뮬레이션)를 여러 번 수행하고,
- 각 상태나 행동에서 얻은 **실제 리턴(총 보상)**을 평균내어,
- 그 상태나 행동의 가치를 추정한다

즉, 시뮬레이션을 반복해서 평균을 구하면 복잡한 수학 없이도 정확한 값을 얻을 수 있다는 게 몬테카를로법이다.

{% highlight js %}
체스에서 “이 수를 두면 앞으로 어떻게 될까?”를 예측하는 대신,

그냥 100번 두어보고 평균적으로 승률이 얼마나 나오는지를 계산하는 방식
{% endhighlight %}

몬테카를로법 샘플 구현

{% highlight js %}
trial = 1000 #샘플링 횟수

sample = []
for _ in range(trial) # 샘플링
  s = sample()
  samples.append(s)
v = sum(samples) / len(samples) #평균 계산
print(V) 
{% endhighlight %}

{% highlight js %}
출력결과 : 6.98
{% endhighlight %}

정답이 7이지만, 결과는 6.98로 정답과 유사하게 나오는 것을 알 수 있다. 
몬테카를로법은 샘플수를 늘릴 수록 분산(variance)이 작아지는데, 뒤에서 더욱 중요하게 다룰 내용이니 참고하도록 하자.


#### 몬테카를로법 정책 평가
에이전트가 실제로 행동하여 얻은 경험(샘플 데이터)로 가치 함수 추정

정책 $$ \pi $$가 주어졌을때, 정책의 가치함수를 몬테카를로법으로 계산

##### $$V(s) $$ : 상태 가치 함수


$$
V^\pi(s) = \mathbb{E}_\pi [ G_t \mid S_t = s ]
$$


- $$ \pi $$: 정책 (state → action 선택 규칙)
- $$ G_t $$: 시점$$ t$$에서 시작해 미래에 받을 누적 보상 (return)
- $$ \mathbb{E}_\pi $$: 정책 $$ \pi $$를 따랐을 때의 기대값

$$
V^\pi(s) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}
$$

정책에 따라 실제로 행동을 취하도록 하고, 이렇게 해서 얻는 실제 수익의 샘플 데이터를 많이 모아서 평균을 구하는 방식이 몬테카를로법이다.
몬테카를로법을 구현하기 위해선 에피소드를 $$n$$번 수행하여 얻은 샘플 데이터의 평균을 구하면 된다.

##### 몬테카를로법을 사용하는 경우(똑같이 상태 $$s$$에서 시작해도 보상 액수는 다를때 )

<div class="mermaid">
graph TD; subgraph 두번째_시도; S2[S]-->B1[행동:0, 보상:0]; B1-->B2[행동:1, 보상:1]; B2-->B3[행동:1, 보상:1]; B3-->G2[G²=2]; end;  subgraph 첫번째_시도; S1[S]-->A1[행동:0, 보상:1]; A1-->A2[행동:0, 보상:0]; A2-->A3[행동:1, 보상:2]; A3-->G1[G¹=3]; end;
</div>
첫번째 시도(수익)    
$$G_1 = 1 + 0 + 2 = 3$$   

두번째 시도(수익)    
$$G_2 = 0 + 1 + 1 = 2$$   

에이전트 정책이 확률적일수도 있고, 환경의 상태 전이가 확률적일 수도 있기 때문에,

둘중 하나라도 확률적이라면 시도할때마다 보상이 확률적으로 달라지는 것을 알 수 있다.

그렇기 때문에 값(보상)이 확률적으로 달라질때 몬테카를로법을 사용한다. 

$$
V^\pi(s^A) = \frac{G^{(1)} + G^{(2)}}{2} = \frac{3 + 2}{2} = 2.5
$$

즉 이 시점의 가치 함수 $$V^\pi(s^A)$$는 2.5가 되며, 시도 횟수를 늘려 수익의 평균을 구한다면 근차치의 정확도가 높아진다.

##### 모든 상태의 가치 함수 구하기

<div class="mermaid">
graph TD; subgraph 부분경로; B2[B]-->C2[C, 보상:R₁]-->End2[종료, 보상:R₂]; end;  subgraph 전체경로; A1[A]-->B1[B, 보상:R₀]-->C1[C, 보상:R₁]-->End1[종료, 보상:R₂]; end;
</div>



$$G_A = R_0 + R_1 + \gamma^2 R_2$$    
($$G_A = R_0 + \gamma G_B$$)

$$G_B = R_1 + \gamma R_2$$    
 ($$G_B = R_1 + \gamma G_C$$)

$$G_C = R_2$$    
($$G_C = R_2$$)


'**한번의 시도**' 만으로 '**세가지 상태에 대한 수익(샘플 데이터)**'를 얻을 수 있다.

> ###### 에이전트의 시작 위치가 고정되어 있더라도 에피소드를 반복하는 동안 모든 상태를 경우할 수 있다면 모든 상태에 대한 수익 샘플 데이터를 수집할 수 있다. 예를 들어 에이전트가 무작위로 행동한다면 에피소드를 반복하면서 다양한 상태로 전이할 것이고, 결국 모든 상태를 경우할 수 있다. 
##### 그렇기에 에이전트의 시작 상태를 임의 위치에 설정할 필요가 없음.

---

#### 몬테카를로법 구현(Gridworld)


<div class="mermaid">
graph TD; subgraph step method Diagram; State[현재 상태 Sₜ]-->Agent[에이전트]; Agent-->Action[행동 Aₜ]; Action-->Env[환경]; Env-->Reward[보상 Rₜ]; Env-->NextState[다음 상태 Sₜ₊₁]; NextState-->Agent; Reward-->Agent; end;
</div>
{% highlight js %}

from common.gridworld import GridWorld

env = GridWorld()
action = [0]  # 행동

next_state, reward, done = env.step(action)  # 행동 수행

print('next state: ', next_state)
print('reward: ', reward)
print('done: ', done)

{% endhighlight %}

출력결과
{% highlight js %}
next state: (1, 0)
reward: 0
done: False
{% endhighlight %}

##### 에이전트 클래스 구현(분포 모델에 따른 구현) 
###### ▶ 샘플 모델에 따른 구현은 TD법에서 설명

{% highlight js %}
from collections import defaultdict
import numpy as np

class RandomAgent:

    # '행동을 샘플링할 수 있다'는 조건의 메서드

    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions) #행동의 확률 분포를 담는 변수(사용 안해도 됨)
        self.v = defaultdict(lambda: 0) #가치함수
        self.cnt = defaultdict(lambda: 0)  #증분방식으로 수익의 평균 구할때 사용
        self.memory = [] # 에이전트의 실제 경험(상태,행동,보상)

    def get_action(self, state):
        actions = self.pi[state].keys()
        probs = list(self.pi[state].values())
        return np.random.choice(list(actions), p=probs) #한개씩 샘플링

    # '행동과 보상'을 기록해주는 메서드

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):  # 메모리를 역방향 순회
            state, action, reward = data
            G = self.gamma * G + reward  # G값(감가된 누적 보상) 계산
            self.cnt[state] += 1
            self.v[state] += (G - self.v[state]) / self.cnt[state] # 상태별 평균 가치 업데이트 (점진적 평균)
{% endhighlight %}
튜플로 묶는 이유 

{% highlight js %}
'코드의 결과 값'
S₀, A₀, R₀, S₁, A₁, R₁, …, S₈, A₈, R₈, S₉

'실제로 저장되는 값'
[(S0, A0, R0), (S1, A1, R1), ..., (S8, A8, R8)]
{% endhighlight %}

- (state, action, reward) 단위로 저장
- 마지막 $$S_9$$은 selft.memory에 저장되지 않기 때문
- **마지막 상태(목표지점)의 가치함수는 0 이므로 갱신할 필요가 없기때문에 저장되지 않는다.**
{% highlight js %}
env = GridWorld()
agent = RandomAgent()
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True: 
        action = agent.get_action(state)  # 행동 선택
        next_state, reward, done = env.step(action)  # 행동 수행, 다음 상태, 보상
        agent.add(state, action, reward)  # 상태, 행동, 보상 저장

        if done:
            agent.eval()  # 에피소드 종료 후 가치 함수 계산
            break

        state = next_state  # 다음 상태로 전환
        
# 모든 에피소드 종료

# 가치 함수 시각화
env.render_v(agent.V)
{% endhighlight %}


[![Figure_1]({{ site.images | relative_url }}/Figure_1.png)]({{ site.images | relative_url }}/Figure_1.png)

- 시작위치는 맨아래의 한 곳으로 고정되어 있지만
무작위 함수 정책이기 때문에 어떠한 위치는 경우할 수 있다.
- 그래서 모든 위치(상태)에서의 가치 함수를 평가할 수 있다.

---

### 몬테카를로법으로 정책 제어하기
핵심 = 최적 정책을 위해서 **평가와 개선을 번갈아 반복하는 것**
- 평가 단계 = 정책을 평가하여 **가치 함수**를 얻음
- 개선 단계 = 가치함수를 **탐욕화**하여 정책을 개선

앞에서 평가를 통해 가치 함수$$ V_\pi(s) $$를 얻었음. 

이번에는 개선단계를 통해 탐욕화를 수행

- 개선단계에서는 가치함수의 값을 최대로 만드는 행동(탐욕화)를 선택



>$$\mu(s) = \arg\max_a Q(s, a)$$    
$$= \arg\max_a \sum_{s'} p(s' \mid s, a) \left\{ r(s, a, s') + \gamma V(s') \right\}$$

- 하지만 이식에는 제약이 있는데, 환경모델을 사용하지 않으면 계산할 수 없다.
- 일반적인 강화학습 문제에서는 $$p(s' \mid s, a), r(s, a, s')$$을 알 수 없다.

>$$\mu(s) = \arg\max_a Q(s, a)$$    

따라서 위의 식을 이용해야 한다. 단순히 $$Q(s, a)$$가가 최대가 되는 행동 $$a$$를 찾아내기만 하면 되므로 환경모델이 필요가 없음

q함수를 평가해야 하는데, 평가단계 이기 때문에 평가 대상을 $$Q$$ 함수로 바꿔줘야 됨.

>**$$V_\pi(s) $$ 에서$$Q(s, a)$$로 전환하는 방식이 필요**

>[상태가치 함수 평가]
- 일반적인 방식:
$$V_n(s) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$
- 증분 방식:
$$V_n(s) = V_{n-1}(s) + \frac{1}{n} \left\{ G^{(n)} - V_{n-1}(s) \right\}$$

>[$$Q$$함수 평가]
- 일반적인 방식:
$$Q_n(s, a) = \frac{G^{(1)} + G^{(2)} + \cdots + G^{(n)}}{n}$$
- 증분 방식:
$$Q_n(s, a) = Q_{n-1}(s, a) + \frac{1}{n} \left\{ G^{(n)} - Q_{n-1}(s, a) \right\}$$

$$Q_n$$ = $$n$$번째 에피소드를 얻을 수 있는 수익

$$V_n$$ = $$n$$번째 에피소드가 끝난 시점의 상태 가치 함수 추정치

$$Q_n(s, a)$$ = $$n$$번째 에피소드가 끝난 시점의 $$Q$$함수 추정치


#####  $$ Q(s, a) $$ : 행동 가치 함수

$$
Q^\pi(s, a) = \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ]
$$

즉:
- 상태$$ s $$에서 행동 $$ a $$를 선택하고
- 이후 정책 $$ \pi $$를 따라가며 받을 누적 보상의 기대값

풀어쓰면:

$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right]$$

---

### 몬테카를로법으로 정책 제어 구현

$$V_\pi(s) $$ 에서$$Q(s, a)$$로 전환하며 구현할 때 , 개선할 점이 2가지 존재
- 완전한 탐욕이 아닌 $$ \epsilon - $$ 탐욕 정책(앱실론 탐욕 정책)으로 변경
- $$Q$$ 갱신을 '고정값 $$a$$ 방식'으로 수행

#### $$ \epsilon - $$ 탐욕 정책(앱실론 탐욕 정책)

탐욕 행동만을 수행하면 에이전트의 경로가 한가지로 고정
모든 상태와 행동 조합에 대한 수익 샘플 데이터 수집이 불가능 해짐 ▷ **에이전트가 '탐색'도 시도하도록 해야함**

그렇기에 나온 대표적인 방법이 $$ \epsilon - $$ 탐욕 정책

- $$Q$$함수의 값이 가장 큰 행동을 선택 하되, 무작위성을 '살짝' 첨가하여 낮은 확률로 아무 행동이나 선택하도록 하는 정책
- 각 상태에서 정해진 행동만 선택되는 문제를 방지할 수 있음
+ 대다수 경우에 탐욕 행동을 취하기 때문에 최적 정책에 가까운 결과 얻을 수 있다

```python
import numpy as np
from collections import defaultdict

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.action_size = 4
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # π: 정책
        self.Q = defaultdict(lambda: 0)               # Q 값
        self.cnts = defaultdict(lambda: 0)            # 방문 횟수
        self.memory = []                             # 경험 메모리

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            self.cnts[key] += 1
            self.Q[key] += (G - self.Q[key]) / self.cnts[key]

        # greedy_probs 적용 (탐욕 정책 확률 계산)
        self.pi[state] = self.greedy_probs(self.Q, state, self.action_size)

    def greedy_probs(Q, state, epsilon=0, action_size=4):
        qs = [Q[(state, action)] for action in range(action_size)]
        max_action = np.argmax(qs)
        
        base_prob = epsilon / action_size
        action_probs = {action: base_prob for action in range(action_size)}

        # 이 시점에서 action_probs = {0:ε/4, 1:ε/4,3:ε/4, 3:ε/4} 
        action_probs[max_action] += (1-epsilon)
        return action_probs
```

탐험은 무작위 선택이 핵심

 - 최적 행동을 완전히 제외하면, 탐험이 편향되고(불완전 랜덤) 잘못된 정책을 학습할 위험이 있다.
 - 확률 ε = 랜덤으로 (탐험) 다른 행동
- 확률 1-ε = 현재 가치 함수 기준 최선의 행동(탐욕 행동)
- ε/4 = 각 행동이 랜덤에서 고르게 선택될 기회

#### 고정값 $$ a $$ 방식으로 수행
{% highlight js %}
**수정 전**
self.Q[key] += (G - self.Q[key]) / self.cnts[state] 

**수정 후**
alpha = 0.1
self.Q[key] += (G - self.Q[key]) * alpha
{% endhighlight %}
수정전 방식은 모듈 샘플 데이터를 균일하게 주고 평균을 낸다.(표본평균)

= 가중치 : 1/n

고정 값 a로 갱신하는 방식
각 데이터마다 가중치가 기하 급수적으로 커지는데, 이를 지수 이동 평균 이라고 한다.
지수 이동 평균은 가중치가 최신 데이터일수록 가중치를 훨씬 크게 준다.

몬테카를로법을 이용한 정책 제어에는 지수 이동 평균이 적합하다.
'수익'이라는 샘플 데이터가 생성되는 확률 분포가 시간에 따라 달라지기 때문에
에피소드가 진행될수록 정책이 갱신되기 때문에 수익이 생성되는 확률 분포가 달라짐.

샘플 데이터(수익)를 생성하는 확률 분포가 일정하지 않은 경우 = 지수 이동 평균이 적합


#### 몬테카를로법으로 정책 반복법 구현

```python
import numpy as np
from collections import defaultdict
 
 # self.epsilon 과 self.alpha 인스턴스 변수로 추가

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1  # (첫 번째 개선) e-탐욕 정책의 ε 무작위 행동 확률
        self.alpha = 0.1    # (두 번째 개선) Q 함수 갱신 시의 고정값 α
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        # self.cnts = defaultdict(lambda:0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)

    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)

            # self.Q[key] += 1
            # self.Q[key] += (G - self.Q[key]) * self.cnts[key] # (개선된 방식)
            self.Q[key] += (G - self.Q[key]) * self.alpha() # Q함수 갱신
            self.pi[state] = greedy_probs(self.Q, state, self.action_size, self.epsilon) #확률 분포로 만듦

def greedy_probs(Q, state, action_size, epsilon):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += 1.0 - epsilon

    return action_probs

# 실행 부분
env = GridWorld()
agent = McAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        agent.add(state, action, reward)

        if done:
            agent.update()
            break

        state = next_state

env.render_q(agent.Q)

```
Figure_2
[![Figure_2]({{ site.images | relative_url }}/Figure_2.png)]({{ site.images | relative_url }}/Figure_2.png)

Q함수로부터 얻은 탐욕 정책 = 최적 정책과 비슷한 결과 확인

에이전트가 ε - 탐욕정책에 따라 어떤 상태에서든 무작위로 행동 가능성 존재하니 주의
But, 대체로 탐욕 행동을 선택하기 때문에 대체로 좋은 결과가 나타남

## 중요도 샘플링(to be continue...)

대상정책 = 평가와 개선의 대상
행동 정책 = 실제 행동을 선택

대상 정책과 행동 정책을 동일시 한것 = 온-정책
대상 정책과 행동 정책을 따로 생각하는 것 = 오프-정책

오프 정책 = 행동 정책에서 탐색 만

온 정책 =  대상 정책에서 활용 만
