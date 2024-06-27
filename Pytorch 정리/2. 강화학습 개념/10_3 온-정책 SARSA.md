### 1. 온-정책 형태의 정책 제어 방식
---
온-정책에서 에이전트는 정책을 하나만 가지고 있다. 실제로 행동을 선택하는 정책(행동 정책)과 평가 및 개선할 정책(대상 정책)이 일치하는 것이다. 

역할 측면에서 보면 에이전트의 정책은 두 가지이다. 
1. 대상 정책 : 평가와 개선의 대상으로서의 정책이다. 즉, 정책에 대해 평가한 다음 개선한다. 이러한 정책을 "대상 정책"이라고 한다. 

2. 행동 정책 : 다른 하나는 에이전트가 실제로 행동을 취할 때 활용하는 정책이다. 이 정책에 따라 '상태, 행동 보상'의 샘플 데이터가 생성된다. 이러한 정책이 "행동 정책"이다

>간단하게 평가, 개선은 대상 정책이, 실제로 행동하는 행동 정책이 따로 있단 것이다. 

온 정책의 경우 행동 정책과 대상 정책이 같으므로 개선 단계에서는 완벽하게 탐욕화할 수 없다. 완벽하게 탐욕화하면 '탐색'을 포기해야 하기 때문이다. 그래서 타협한 끝에 ɛ-탐욕 정책을 이용한다. 그렇게 하면 이따금 탐색을 하면서도 대부분의 경우에는 탐욕스럽게 행동할 수 있다. 

뭔 소리일까? 천천히 이해해보자

먼저 에이전트가 정책 π에 따라 행동한다고 하자. 구체적으로 시간 t와 t+1에서 다음 그림처럼 행동했다고 하자

![[Pasted image 20240621005028.png|500]]

Q 함수는 상태와 행동을 묶은 데이터를 하나의 단위로 삼는다. 그림처럼 S와 A를 묶어놨다. 위와 같은 데이터(S_t, A_t, R_t, S_(t+1), A_(t+1))를 얻었다면 우리가 구한 식 

![[Pasted image 20240621005213.png]]

에 대입해서 Q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub>)를 즉시 갱신할 수 있다. 그리고 이 갱신이 끝나면 바로 '개선'단계로 넘어갈 수 있다. 

지금의 예시에서는 Q<sub>π</sub>(S<sub>t</sub>, A<sub>t</sub>)가 갱신되기 때문에 상태 S<sub>t</sub>에서의 정책이 바뀔 수 있다. 구체적으로 나타낸다면

![[Pasted image 20240621005934.png]]

인 것이다.

이러한 ɛ-탐욕 정책에 따라 상태 S<sub>t</sub>에서 행동을 선택하는 방법을 갱신한다. 

즉, 

![[Pasted image 20240621010238.png]]
으로 평가하고

![[Pasted image 20240621005934.png]]
으로 갱신을 한다는 것이다. 

>뭐야 분리되어있잖아? 라고 할 수 있지만, 잘 보면 행동 가치 함수가 중복되어서 사용된다. 즉 대상 정책과 행동 정책이 같은 온-정책이다.  

위 식과 같이 ɛ의 확률로 무작위 행동을 선택하고, 그 외에는 탐욕 행동을 선택한다. 탐욕 행동으로 정책을 개선하고 무작위 행동으로 탐색을 수행하는 것이다. 이러한 ɛ-탐욕 정책에 따라 상태 S<sub>t</sub>에서 행동을 선택하는 방법을 갱신한다.

이러한 알고리즘이 SARSA이다.

참고로 이 이름은 TD법에서 사용하는 데이터 SARSA를 따온 것이다.


### 2. 온-정책 SARSA 구현
---
``` python
class SarsaAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)  # deque 사용 (1)

    def get_action(self, state):
        action_probs = self.pi[state]  # pi에서 선택 (2)
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]
        next_q = 0 if done else self.Q[next_state, next_action] #다음 Q함수 (3)

        # TD법으로 self.Q 갱신 (4)
        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha
        
        # 정책 개선 (5)
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)
```

1) 큐를 사용했다. 가장 최근의 경험 데이터만 보관하기 위해서이다. 수식을 보면 알겠지만 2개만 필요하다
2) 온-정책이기에 정책을 하나만 사용한다. 유일한 정책인 `self.[pi]`에서 행동을 선택한다.
3) done 플래그가 True면 목표에 도달했음을 의미한다. 목표에서의 Q함수는 항상 0이다. Q함수는 미래에 얻을 수 있는 보상의 총합인데, 이미 목표에 도달했으므로 앞으로 더 받을 게 없기 때문이다.
4)  SARSA알고리즘에 따라 self.Q를 갱신한다.
5) 정책을 개선하기 위해 앞 장에서 구현한 greedy_probs() 함수를 사용한다. 이제 정책 self.pi의 상태 state에서의 행동은 ɛ-탐욕 정책에 따라 결정된다.


``` python
env = GridWorld()
agent = SarsaAgent()

episodes = 10000
for episode in range(episodes):
    state = env.reset()
    agent.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)

        agent.update(state, action, reward, done)  # 매번 호출

        if done:
            # 목표에 도달했을 때도 호출
            agent.update(next_state, None, None, None)
            break
        state = next_state

# [그림 6-7] SARSA로 얻은 결과
env.render_q(agent.Q)
```

