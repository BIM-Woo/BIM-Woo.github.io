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