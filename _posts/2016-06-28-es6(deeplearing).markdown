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
{% highlight js %}


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

