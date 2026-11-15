# free-model-aggregator-2

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/exrazor01-creator/free-model-aggregator-2/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/exrazor01-creator/free-model-aggregator-2/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A one-process Node server that answers OpenAI Chat Completions and Anthropic Messages requests with real completions from live free OpenRouter models — ranked by the house quality table, with rate-limit-aware failover and both API wires in one zero-config process. Standard library only, no dependencies, fully deterministic.

## Run it

```sh
node main.ts --list
```

With a trial brief and env set, `node main.ts` registers its catalog, serves the workload through the gateway, and exits when the work is done:

```sh
MAKEFUN_PROMPT_FILE=prompt.txt MAKEFUN_ENV_URL=http://127.0.0.1:8080 OPENAI_BASE_URL=http://127.0.0.1:18080/v1 node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=9taf1vak&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
