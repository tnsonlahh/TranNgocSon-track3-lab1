# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_qa_first_100.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.83 | 0.89 | 0.06 |
| Avg attempts | 1 | 1.33 | 0.33 |
| Avg token estimate | 1735.75 | 2393.4 | 657.65 |
| Avg latency (ms) | 2446.65 | 3927.19 | 1480.54 |

## Failure modes
```json
{
  "react": {
    "none": 83,
    "wrong_final_answer": 17
  },
  "reflexion": {
    "none": 89,
    "wrong_final_answer": 11
  },
  "overall": {
    "none": 172,
    "wrong_final_answer": 28
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json
- mock_mode_for_autograding
- adaptive_max_attempts
- memory_compression

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
