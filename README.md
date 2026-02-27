# DSE Economics - Opportunity Cost MVP Backend

> FastAPI 后端实现 v1.2.2  
> 严格遵循 v1.2 规则：单一 primary weakness、精确 dimension_counts、STABLE 不回落
> 新增 /metrics 端点用于课堂试验指标统计

---

## 快速启动

### 1. 安装依赖

```bash
cd MVP_Reports/backend
pip install -r requirements.txt
```

### 2. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 3. 访问 API

- API 文档：http://localhost:8000/docs
- 根路径：http://localhost:8000/

---

## API 列表

### POST /micro/start

开始一次 Micro Session，返回 5 道 strict 题。

**Request:**
```json
{
  "user_id": "user001",
  "topic_id": "1.2"
}
```

**Response:**
```json
{
  "session_id": "uuid-xxx",
  "topic_id": "1.2",
  "questions": [
    {
      "question_id": "2016_P1_Q03",
      "stem": "Mr. Ng, a Hong Kong citizen...",
      "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
    },
    ...
  ],
  "total_count": 5
}
```

**注意**：不返回 `ability_dimension` 给前端。

---

### POST /micro/submit

提交 Micro Session，计算 micro_summary，更新 buffer。

**Request (v1.2.1):**
```json
{
  "user_id": "user001",
  "topic_id": "1.2",
  "attempts": [
    {
      "question_id": "2016_P1_Q03",
      "selected_option": "D"
    },
    {
      "question_id": "2018_P1_Q02",
      "selected_option": "C"
    }
  ],
  "is_incomplete": false
}
```

**注意**：Attempt 只接收 `question_id` + `selected_option`，后端根据题库计算 `correct` 和 `ability_dimension`。

**Response:**
```json
{
  "micro_summary": {
    "session_id": "uuid-xxx",
    "total_attempted": 5,
    "total_correct": 4,
    "overall_acc": 0.8,
    "dimension_counts": {"COST_STRUCTURE": 2, "ALT_LOGIC": 2, "SUNK_FILTER": 1},
    "dimension_correct": {"COST_STRUCTURE": 2, "ALT_LOGIC": 1, "SUNK_FILTER": 1},
    "dimension_acc": {"COST_STRUCTURE": 1.0, "ALT_LOGIC": 0.5, "SUNK_FILTER": 1.0}
  },
  "eval_ready": false,
  "buffer_count": 1
}
```

---

### GET /evaluation

获取 Evaluation 结果（buffer 满 3 次后触发）。

**Query:** `?user_id=user001`

**Response:**
```json
{
  "eval_result": {
    "total_attempted": 15,
    "total_correct": 10,
    "overall_acc": 0.667,
    "dimension_counts": {"COST_STRUCTURE": 5, "ALT_LOGIC": 6, "SUNK_FILTER": 4},
    "dimension_correct": {"COST_STRUCTURE": 4, "ALT_LOGIC": 3, "SUNK_FILTER": 3},
    "dimension_acc": {"COST_STRUCTURE": 0.8, "ALT_LOGIC": 0.5, "SUNK_FILTER": 0.75},
    "data_sufficiency": {
      "is_sufficient": true,
      "sufficient_dimensions": ["COST_STRUCTURE", "ALT_LOGIC", "SUNK_FILTER"],
      "insufficient_dimensions": [],
      "min_required": 2
    },
    "weakness_detection": {
      "weak_dimension": "ALT_LOGIC",
      "primary_acc": 0.5,
      "needs_attention": ["SUNK_FILTER"],
      "status": "WEAK_DETECTED",
      "confidence": "HIGH"
    },
    "all_dimensions": {
      "COST_STRUCTURE": {"status": "OK", "acc": 0.8, "count": 5},
      "ALT_LOGIC": {"status": "WEAK_DETECTED", "acc": 0.5, "count": 6},
      "SUNK_FILTER": {"status": "NEEDS_ATTENTION", "acc": 0.75, "count": 4}
    }
  }
}
```

**Data Sufficiency 说明**：
- `MIN_QUESTIONS_PER_DIMENSION = 2`
- 若某维度 attempted < 2，该维度不参与弱点判定
- 若参与比较的维度不足 2 个，返回 `status: "INSUFFICIENT_DATA"`

---

### POST /challenge/start

开始 Challenge 专项训练。

**Request:**
```json
{
  "user_id": "user001",
  "topic_id": "1.2",
  "weak_dimension": "ALT_LOGIC",
  "pool_policy": "strict"
}
```

**Response:**
```json
{
  "session_id": "uuid-xxx",
  "topic_id": "1.2",
  "weak_dimension": "ALT_LOGIC",
  "current_state": {
    "status": "WEAK_DETECTED",
    "consecutive_pass": 1,
    "target": 2
  },
  "questions": [...],
  "total_count": 4
}
```

---

### POST /challenge/submit

提交 Challenge，计算 acc，更新弱点状态。

**Request (v1.2.1):**
```json
{
  "user_id": "user001",
  "topic_id": "1.2",
  "weak_dimension": "ALT_LOGIC",
  "attempts": [
    {"question_id": "2023_P1_Q01", "selected_option": "C"},
    {"question_id": "2015_P1_Q02", "selected_option": "C"},
    {"question_id": "2018_P1_Q03", "selected_option": "B"},
    {"question_id": "2022_P1_Q03", "selected_option": "C"}
  ]
}
```

**注意**：Attempt 只接收 `question_id` + `selected_option`，后端计算 `correct`。

**Response:**
```json
{
  "challenge_acc": 1.0,
  "new_state": {
    "status": "STABLE",
    "consecutive_pass": 0,
    "message": "恭喜！弱点已解除"
  }
}
```

**consecutive_pass 逻辑 (v1.2.1)**：
- `consecutive_pass == 0` 且 `acc < 80%` → 保持 0
- `consecutive_pass == 1` 且 `acc < 80%` → 保持 1（心理缓冲）
- `acc >= 80%` → `consecutive_pass += 1`；到 2 则 STABLE 并归 0
- STABLE 状态不回落

---

### GET /dashboard

获取 Dashboard 状态。

**Query:** `?user_id=user001`

**Response:**
```json
{
  "topic_id": "1.2",
  "topic_name": "Opportunity Cost",
  "eval_buffer": {
    "current": 2,
    "max": 3
  },
  "primary_weakness": {
    "dimension": "ALT_LOGIC",
    "dimension_cn": "比较逻辑",
    "status": "WEAK_DETECTED",
    "consecutive_pass": 1,
    "remaining": 1,
    "last_update": "2026-02-27T10:30:00Z"
  },
  "all_dimensions_status": {
    "COST_STRUCTURE": {"status": "OK", "acc": 0.8},
    "ALT_LOGIC": {"status": "WEAK_DETECTED", "acc": 0.5},
    "SUNK_FILTER": {"status": "NEEDS_ATTENTION", "acc": 0.75}
  }
}
```

---

### GET /metrics

获取课堂试验指标统计（基于内存数据）。

**Query:** 无需参数

**Response:**
```json
{
  "topic_id": "1.2",
  "updated_at": "2026-02-27T12:00:00.000000",
  "metrics": {
    "total_users": 10,
    "users_with_eval_count_ge_1": 8,
    "users_with_eval_count_ge_2": 5,
    "users_entered_challenge": 6,
    "users_completed_challenge_ge_1": 5,
    "users_became_stable": 2,
    "eval_trigger_rate": 0.8,
    "challenge_entry_rate": 0.75,
    "challenge_completion_rate": 0.833
  },
  "per_user": [
    {
      "user_id": "user001",
      "micro_count": 6,
      "eval_count": 2,
      "entered_challenge": true,
      "challenge_count": 2,
      "stable_dimensions": ["ALT_LOGIC"],
      "last_primary_dimension": "ALT_LOGIC"
    }
  ]
}
```

**指标说明**：
| 指标 | 说明 |
|------|------|
| total_users | 总用户数 |
| users_with_eval_count_ge_1 | 至少完成 1 次 evaluation 的用户数 |
| users_with_eval_count_ge_2 | 至少完成 2 次 evaluation 的用户数 |
| users_entered_challenge | 进入过 challenge/start 的用户数 |
| users_completed_challenge_ge_1 | 至少完成 1 次 challenge/submit 的用户数 |
| users_became_stable | 任一维度达到 STABLE 的用户数 |
| eval_trigger_rate | users_with_eval_count_ge_1 / total_users |
| challenge_entry_rate | users_entered_challenge / users_with_eval_count_ge_1 |
| challenge_completion_rate | users_completed_challenge_ge_1 / users_entered_challenge |

---

## v1.2.2 规则实现

| 规则 | 实现 |
|------|------|
| Attempt 只接收 question_id + selected_option | 后端从题库计算 correct 和 ability_dimension |
| 单一 primary weakness | `detect_weakness` 只返回最低维度 |
| needs_attention 标记 | 其他 <70% 维度标记为 `NEEDS_ATTENTION` |
| 精确 dimension_counts | 精确统计，无近似分配 |
| Data Sufficiency | MIN_QUESTIONS_PER_DIMENSION = 2 |
| STABLE 不回落 | `update_weakness_state` 中 STABLE 状态保持 |
| cross-topic strict | 抽题池仅使用 `cross_topic_ids=[]` 的题 |
| 题库来源 | 从 `oc_mvp_mcq_v2.json` 加载 |
| 避免重复 | 用户当日已出题记录在 `used_questions` 中 |
| 指标统计 | /metrics 端点返回实时聚合数据 |

---

## 测试流程（curl 示例）

```bash
# 1. 启动服务
uvicorn main:app --reload --port 8000

# 2. 开始 micro session 1
curl -X POST "http://localhost:8000/micro/start" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test001"}'

# 3. 提交 micro session 1（只传 question_id + selected_option）
curl -X POST "http://localhost:8000/micro/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test001",
    "attempts": [
      {"question_id": "2016_P1_Q03", "selected_option": "D"},
      {"question_id": "2018_P1_Q02", "selected_option": "C"},
      {"question_id": "2023_P1_Q01", "selected_option": "C"},
      {"question_id": "2019_P1_Q03", "selected_option": "B"},
      {"question_id": "2025_P1_Q01", "selected_option": "C"}
    ]
  }'

# 4. 查看 dashboard
curl "http://localhost:8000/dashboard?user_id=test001"

# 5. 重复步骤 2-3 三次，直到 buffer == 3

# 6. 获取 evaluation
curl "http://localhost:8000/evaluation?user_id=test001"

# 7. 开始 challenge（根据 evaluation 结果的 weak_dimension）
curl -X POST "http://localhost:8000/challenge/start" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test001", "weak_dimension": "ALT_LOGIC"}'

# 8. 提交 challenge
curl -X POST "http://localhost:8000/challenge/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test001",
    "weak_dimension": "ALT_LOGIC",
    "attempts": [
      {"question_id": "2023_P1_Q01", "selected_option": "C"},
      {"question_id": "2015_P1_Q02", "selected_option": "C"},
      {"question_id": "2018_P1_Q03", "selected_option": "B"},
      {"question_id": "2022_P1_Q03", "selected_option": "C"}
    ]
  }'

# 9. 再次查看 dashboard
curl "http://localhost:8000/dashboard?user_id=test001"

# 10. 查看指标统计
curl "http://localhost:8000/metrics"
```

---

## 前端（Minimal HTML）

### 快速启动

```bash
# 1. 在前端目录启动 HTTP 服务器
cd MVP_Reports/frontend
python -m http.server 5500

# 2. 打开浏览器访问
# http://localhost:5500
```

### 配置后端地址

前端默认连接 `http://localhost:8000`，如需修改：

```javascript
// 打开 index.html，找到第 13 行
const BASE_URL = 'http://localhost:8000';
// 修改为你的后端地址
const BASE_URL = 'http://your-server-ip:8000';
```

### 前端功能

| 页面 | 功能 |
|------|------|
| 登录页 | 输入 user_id 开始 |
| Dashboard | 显示 buffer 进度 + primary weakness |
| Micro | 5 题单选，提交后 eval_ready 则进入 Evaluation |
| Evaluation | 展示 overall_acc + 三维度 + 弱点消息 |
| Challenge | 针对弱点维度做题，展示结果 |

### 自测流程

1. 打开 http://localhost:5500
2. 输入名字，点击"开始练习"
3. 点击"开始练习"进入 Micro Session
4. 完成 5 题后点击"提交"
5. 重复步骤 3-4 三次（buffer 0/3 → 3/3）
6. 自动进入 Evaluation（若检测到弱点）
7. 点击"开始专项挑战"进入 Challenge
8. 完成 Challenge 后点击"返回 Dashboard"

---

## 依赖

- fastapi >= 0.104.0
- uvicorn >= 0.24.0
- pydantic >= 2.5.0
- python-multipart >= 0.0.6

---

*版本：v1.2.2 | 更新日期：2026-02-27*
