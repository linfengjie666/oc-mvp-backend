"""
Opportunity Cost MVP - FastAPI Backend (v1.3)

严格遵循 v1.2 规则，并修复：
1. Attempt 只接收 question_id + selected_option，后端计算 correct
2. 加入 data_sufficiency 检查
3. 修正 consecutive_pass 逻辑
4. 从 oc_mvp_mcq_v2.json 加载题库
5. 前端不返回 ability_dimension
6. 新增 /metrics 端点用于课堂试验指标统计
7. 新增 /question/feedback 端点用于即时反馈（v1.3）
"""

import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
from datetime import datetime, date

app = FastAPI(title="DSE Economics - Opportunity Cost MVP")

# ============================================================================
# CORS 配置（允许所有前端域名访问）
# ============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 配置
# ============================================================================

# 题库文件路径（相对于 main.py 所在目录）
QUESTION_BANK_PATH = os.path.join(
    os.path.dirname(__file__),
    "oc_mvp_mcq_v2.json"
)

# 解释题库文件路径（包含 ai_explanation）
# 放在 backend/ 目录下，与 main.py 同目录
EXPLANATION_BANK_PATH = os.path.join(
    os.path.dirname(__file__),
    "1.2_Opportunity_Cost_MCQ.json"
)

DIMENSIONS = ["COST_STRUCTURE", "ALT_LOGIC", "SUNK_FILTER"]

# Data Sufficiency 阈值
MIN_QUESTIONS_PER_DIMENSION = 2

# Challenge 阈值
CHALLENGE_PASS_THRESHOLD = 0.80
CONSECUTIVE_PASS_REQUIRED = 2
WEAKNESS_THRESHOLD = 0.70

# ============================================================================
# 加载题库
# ============================================================================

def load_explanation_bank() -> Dict[str, dict]:
    """从解释题库 JSON 加载解释数据，用 uid 作为 key"""
    explanation_map = {}
    
    # 打印调试信息
    resolved_path = EXPLANATION_BANK_PATH
    print(f"=== Loading Explanation Bank ===")
    print(f"Resolved path: {resolved_path}")
    print(f"File exists: {os.path.exists(resolved_path)}")
    
    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            explanations = json.load(f)
        
        for exp in explanations:
            uid = exp.get("uid")
            if uid:
                explanation_map[uid] = exp
        
        print(f"SUCCESS: Loaded {len(explanation_map)} explanations from {resolved_path}")
        print(f"Sample uids: {list(explanation_map.keys())[:5]}")
        return explanation_map
    except FileNotFoundError:
        print(f"ERROR: Explanation bank file NOT FOUND at: {resolved_path}")
        print(f"Please ensure 1.2_Opportunity_Cost_MCQ.json is in the backend/ folder")
        return explanation_map
    except Exception as e:
        print(f"ERROR loading explanation bank: {e}")
        return explanation_map

def load_question_bank() -> List[dict]:
    """从 JSON 文件加载题库，并合并解释数据"""
    # 加载解释题库
    explanation_bank = load_explanation_bank()
    
    try:
        with open(QUESTION_BANK_PATH, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        # 合并解释数据
        merged_count = 0
        for q in questions:
            question_id = q.get("question_id")
            
            # 尝试用 question_id 匹配解释
            exp = explanation_bank.get(question_id)
            
            # 如果没找到，尝试其他可能的映射
            if not exp:
                # 尝试从 question_id 提取 uid 格式 (e.g., 2016_P1_Q03)
                alt_ids = [
                    question_id,
                    question_id.replace("_Q", "_Q"),
                    question_id.replace("P1_", "P1_Q") if "P1_" in question_id else question_id,
                ]
                for alt_id in alt_ids:
                    if alt_id in explanation_bank:
                        exp = explanation_bank[alt_id]
                        break
            
            if exp:
                # 打印调试信息
                print(f"Merging question_id={question_id}")
                print(f"  - ai_explanation type: {type(exp.get('ai_explanation'))}")
                print(f"  - ai_explanation_en type: {type(exp.get('ai_explanation_en'))}")
                print(f"  - distractor_analysis: {exp.get('distractor_analysis') is not None}")
                
                # 合并解释字段 - 修复：处理多种字段格式
                q["year"] = exp.get("year")
                q["paper"] = exp.get("paper")
                q["answer"] = exp.get("answer")
                
                # 处理 ai_explanation 字段 - 支持多种格式
                ai_exp = exp.get("ai_explanation")
                ai_exp_en = exp.get("ai_explanation_en")
                
                # ai_explanation 是对象时的处理
                if ai_exp and isinstance(ai_exp, dict):
                    q["ai_logic_flow"] = ai_exp.get("logic_flow")
                    q["ai_common_trap"] = ai_exp.get("common_trap")
                # ai_explanation_en 是字符串时的处理
                elif ai_exp_en and isinstance(ai_exp_en, str):
                    q["ai_logic_flow"] = ai_exp_en  # 直接使用字符串作为 logic_flow
                    q["ai_common_trap"] = ai_exp.get("common_trap") if ai_exp else None
                else:
                    q["ai_logic_flow"] = None
                    q["ai_common_trap"] = None
                
                # distractor_analysis 直接复制
                q["distractor_analysis"] = exp.get("distractor_analysis")
                merged_count += 1
            else:
                print(f"Warning: No explanation found for question_id={question_id}")
        
        print(f"Merged explanations for {merged_count}/{len(questions)} questions")
        print(f"Loaded {len(questions)} questions from {QUESTION_BANK_PATH}")
        return questions
    except FileNotFoundError:
        print(f"Warning: Question bank not found at {QUESTION_BANK_PATH}")
        return []

# 启动时加载题库
QUESTION_BANK = load_question_bank()

# 创建题库索引供快速查找（v1.3 feedback endpoint）
QUESTION_INDEX = {q.get("question_id"): q for q in QUESTION_BANK if q.get("question_id")}

# ============================================================================
# 内存存储（模拟数据库）
# ============================================================================

# 用户数据结构
# {
#     "user_id": {
#         "date": "2026-02-27",           # 当天日期
#         "used_questions": {"qid1", "qid2"},  # 当天已出题
#         "eval_buffer": [micro_summary1, micro_summary2, micro_summary3],
#         "weakness_state": {
#             "primary_dimension": {...},
#             "all_dimensions": {...}
#         },
#         # Metrics tracking
#         "micro_count": 0,               # 完成的 micro session 次数
#         "eval_count": 0,                # 触发的 evaluation 次数
#         "challenge_count": 0,           # 完成的 challenge 次数
#         "stable_dimensions": [],        # 已稳定的维度列表
#     }
# }
users_db: Dict = {}

# ============================================================================
# Pydantic Models
# ============================================================================

class MicroStartRequest(BaseModel):
    user_id: str
    topic_id: str = "1.2"

# v1.2.1: Attempt 只接收 question_id + selected_option
class Attempt(BaseModel):
    question_id: str
    selected_option: str

class MicroSubmitRequest(BaseModel):
    user_id: str
    topic_id: str = "1.2"
    attempts: List[Attempt]
    is_incomplete: bool = False

class ChallengeStartRequest(BaseModel):
    user_id: str
    topic_id: str = "1.2"
    weak_dimension: str
    pool_policy: str = "strict"

class ChallengeSubmitRequest(BaseModel):
    user_id: str
    topic_id: str = "1.2"
    weak_dimension: str
    attempts: List[Attempt]

# v1.3 feedback endpoint
class FeedbackRequest(BaseModel):
    user_id: str
    question_id: str
    selected_option: str

class FeedbackResponse(BaseModel):
    question_id: str
    selected_option: str
    correct_option: str
    is_correct: bool
    explain: dict
    source: dict

# ============================================================================
# 辅助函数
# ============================================================================

def get_today() -> str:
    """获取当天日期字符串"""
    return date.today().isoformat()

def get_strict_question_pool():
    """获取 strict 池（不含 cross-topic 题）"""
    return [q for q in QUESTION_BANK if q.get("cross_topic_ids") == []]

def get_subtopic_questions(sub_topic_id: str):
    """获取指定 sub_topic 的 strict 题"""
    return [
        q for q in QUESTION_BANK 
        if q.get("sub_topic_id") == sub_topic_id and q.get("cross_topic_ids") == []
    ]

def get_dim_questions(dim: str):
    """获取指定维度的 strict 题"""
    return [
        q for q in QUESTION_BANK 
        if q.get("ability_dimension") == dim and q.get("cross_topic_ids") == []
    ]

def get_question_by_id(qid: str) -> Optional[dict]:
    """根据 question_id 获取题目"""
    for q in QUESTION_BANK:
        if q.get("question_id") == qid:
            return q
    return None

def init_user(user_id: str):
    """初始化用户数据"""
    today = get_today()
    now = datetime.now()
    now_iso = now.isoformat()
    
    if user_id not in users_db:
        users_db[user_id] = {
            "date": today,
            "used_questions": set(),
            "eval_buffer": [],
            "weakness_state": {
                "primary_dimension": None,
                "all_dimensions": {
                    dim: {"status": "OK", "acc": None} for dim in DIMENSIONS
                }
            },
            # Metrics tracking
            "micro_count": 0,
            "eval_count": 0,
            "challenge_count": 0,
            "stable_dimensions": [],
            "entered_challenge": False,
            # Time tracking
            "first_seen_at": now_iso,
            "last_activity_at": now_iso,
            "last_micro_at": None,
            "last_challenge_at": None
        }
    else:
        # 检查是否新一天，重置 used_questions
        if users_db[user_id].get("date") != today:
            users_db[user_id]["date"] = today
            users_db[user_id]["used_questions"] = set()
        # 更新 last_activity_at
        users_db[user_id]["last_activity_at"] = now_iso

def sample_questions(pool: List[dict], count: int, used: set) -> List[dict]:
    """从池中抽取题目，避免重复"""
    # 优先从未用过的题目中抽取
    unused = [q for q in pool if q["question_id"] not in used]
    if len(unused) >= count:
        import random
        return random.sample(unused, count)
    
    # 如果未用完的不够，再从用过的中补
    import random
    remaining = count - len(unused)
    available = [q for q in pool if q["question_id"] in used]
    result = unused + random.sample(available, min(remaining, len(available)))
    random.shuffle(result)
    return result

# ============================================================================
# API 实现
# ============================================================================

@app.post("/micro/start")
def micro_start(req: MicroStartRequest):
    """
    开始一次 Micro Session
    返回 5 道 strict 题
    - 不返回 ability_dimension 给前端
    - 避免重复题目
    """
    init_user(req.user_id)
    user_data = users_db[req.user_id]
    
    pool = get_subtopic_questions("1.2")
    questions = sample_questions(pool, 5, user_data["used_questions"])
    
    # 记录已用题目
    for q in questions:
        user_data["used_questions"].add(q["question_id"])
    
    return {
        "session_id": str(uuid.uuid4()),
        "topic_id": req.topic_id,
        "questions": [
            {
                "question_id": q["question_id"],
                "stem": q["stem"],
                "options": q["options"]
                # 注意：不返回 ability_dimension
            }
            for q in questions
        ],
        "total_count": len(questions)
    }


@app.post("/micro/submit")
def micro_submit(req: MicroSubmitRequest):
    """
    提交 Micro Session
    - 后端计算 correct 和 ability_dimension
    - 更新 evaluation_buffer
    """
    init_user(req.user_id)
    
    # ========================================================================
    # Step 1: 后端计算 correct 和 ability_dimension
    # ========================================================================
    processed_attempts = []
    for attempt in req.attempts:
        q = get_question_by_id(attempt.question_id)
        if not q:
            raise HTTPException(status_code=400, detail=f"Question {attempt.question_id} not found")
        
        is_correct = attempt.selected_option.upper() == q.get("correct_option", "").upper()
        ability_dim = q.get("ability_dimension", "UNKNOWN")
        
        processed_attempts.append({
            "question_id": attempt.question_id,
            "selected_option": attempt.selected_option,
            "correct": is_correct,
            "ability_dimension": ability_dim
        })
    
    # ========================================================================
    # Step 2: compute_micro_session
    # ========================================================================
    attempts = processed_attempts
    total = len(attempts)
    correct = sum(1 for a in attempts if a["correct"])
    
    # 精确统计 dimension_counts 和 dimension_correct
    dimension_counts = {}
    dimension_correct = {}

    for a in attempts:
        dim = a["ability_dimension"]

        if dim not in dimension_counts:
            dimension_counts[dim] = 0
            dimension_correct[dim] = 0

        dimension_counts[dim] += 1

        if a["correct"]:
            dimension_correct[dim] += 1

    # 由 counts 推导 dimension_acc（避免 KeyError & 除以 0）
    dimension_acc = {}
    for dim, cnt in dimension_counts.items():
        dimension_acc[dim] = (
            dimension_correct[dim] / cnt if cnt > 0 else None
        )
    
    micro_summary = {
        "session_id": str(uuid.uuid4()),
        "topic_id": req.topic_id,
        "total_attempted": total,
        "total_correct": correct,
        "overall_acc": correct / total if total > 0 else 0,
        "dimension_counts": dimension_counts,
        "dimension_correct": dimension_correct,
        "dimension_acc": dimension_acc,
        "is_incomplete": req.is_incomplete,
        "timestamp": datetime.now().isoformat()
    }
    
    # ========================================================================
    # Step 3: update_evaluation_buffer
    # ========================================================================
    user_data = users_db[req.user_id]
    user_data["eval_buffer"].append(micro_summary)
    user_data["micro_count"] = user_data.get("micro_count", 0) + 1
    
    # 更新时间戳
    now_iso = datetime.now().isoformat()
    user_data["last_activity_at"] = now_iso
    user_data["last_micro_at"] = now_iso
    
    if len(user_data["eval_buffer"]) > 3:
        user_data["eval_buffer"] = user_data["eval_buffer"][-3:]
    
    eval_ready = len(user_data["eval_buffer"]) >= 3
    
    return {
        "micro_summary": {
            "session_id": micro_summary["session_id"],
            "total_attempted": total,
            "total_correct": correct,
            "overall_acc": micro_summary["overall_acc"],
            "dimension_counts": dimension_counts,
            "dimension_correct": dimension_correct,
            "dimension_acc": dimension_acc,
            "is_incomplete": req.is_incomplete
        },
        "eval_ready": eval_ready,
        "buffer_count": len(user_data["eval_buffer"])
    }


@app.get("/evaluation")
def get_evaluation(user_id: str):
    """
    获取 Evaluation 结果
    - 加入 data_sufficiency 检查
    - 若参与比较的维度不足2个，返回 INSUFFICIENT_DATA
    """
    init_user(user_id)
    user_data = users_db[user_id]
    
    if len(user_data["eval_buffer"]) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Evaluation not ready. Buffer: {len(user_data['eval_buffer'])}/3"
        )
    
    # ========================================================================
    # compute_evaluation_session（精确汇总）
    # ========================================================================
    buffer = user_data["eval_buffer"]
    
    total_attempted = 0
    total_correct = 0
    dimension_counts = {}
    dimension_correct = {}
    
    for micro in buffer:
        total_attempted += micro["total_attempted"]
        total_correct += micro["total_correct"]
        
        for dim, cnt in micro["dimension_counts"].items():
            dimension_counts[dim] = dimension_counts.get(dim, 0) + cnt
        
        for dim, cnt in micro["dimension_correct"].items():
            dimension_correct[dim] = dimension_correct.get(dim, 0) + cnt
    
    # 由精确 counts 推导 dimension_acc
    dimension_acc = {}
    for dim in dimension_counts:
        if dimension_counts[dim] > 0:
            dimension_acc[dim] = dimension_correct[dim] / dimension_counts[dim]
    
    # ========================================================================
    # data_sufficiency 检查
    # ========================================================================
    sufficient_dims = [d for d in DIMENSIONS if dimension_counts.get(d, 0) >= MIN_QUESTIONS_PER_DIMENSION]
    insufficient_dims = [d for d in DIMENSIONS if dimension_counts.get(d, 0) < MIN_QUESTIONS_PER_DIMENSION]
    
    data_sufficiency = {
        "is_sufficient": len(sufficient_dims) >= 2,
        "sufficient_dimensions": sufficient_dims,
        "insufficient_dimensions": insufficient_dims,
        "min_required": MIN_QUESTIONS_PER_DIMENSION
    }
    
    # ========================================================================
    # detect_weakness（考虑 data_sufficiency）
    # ========================================================================
    if not data_sufficiency["is_sufficient"]:
        # 数据不足，无法判定弱点
        weakness_detection = {
            "weak_dimension": None,
            "needs_attention": [],
            "status": "INSUFFICIENT_DATA",
            "message": f"数据不足，继续完成更多练习后再查看评定（需至少 {MIN_QUESTIONS_PER_DIMENSION} 题/维度）"
        }
    else:
        # 只用 sufficient_dims 进行弱点判定
        dim_status = {}
        for dim in sufficient_dims:
            acc = dimension_acc.get(dim, 0)
            if acc < WEAKNESS_THRESHOLD:
                dim_status[dim] = ("BELOW_THRESHOLD", acc)
            else:
                dim_status[dim] = ("OK", acc)
        
        below_threshold = [(d, s[1]) for d, s in dim_status.items() if s[0] == "BELOW_THRESHOLD"]
        
        if not below_threshold:
            weakness_detection = {
                "weak_dimension": None,
                "needs_attention": [],
                "status": "ALL_OK",
                "message": "整体表现良好"
            }
        else:
            sorted_below = sorted(below_threshold, key=lambda x: x[1])
            primary_weak = sorted_below[0][0]
            primary_acc = sorted_below[0][1]
            needs_attention = [d for d, _ in sorted_below[1:]]
            
            total_att = sum(dimension_counts.values())
            confidence = "HIGH" if total_att >= 6 else "MEDIUM"
            
            weakness_detection = {
                "weak_dimension": primary_weak,
                "primary_acc": primary_acc,
                "needs_attention": needs_attention,
                "status": "WEAK_DETECTED",
                "confidence": confidence,
                "message": f"在 {primary_weak} 维度正确率仅 {primary_acc*100:.0f}%，需加强练习"
            }
    
    # 构建 all_dimensions 状态（包括 insufficient 的维度）
    all_dimensions = {}
    for dim in DIMENSIONS:
        acc = dimension_acc.get(dim, 0)
        cnt = dimension_counts.get(dim, 0)
        
        if dim in insufficient_dims:
            all_dimensions[dim] = {"status": "INSUFFICIENT", "acc": acc, "count": cnt}
        elif dim == weakness_detection.get("weak_dimension"):
            all_dimensions[dim] = {"status": "WEAK_DETECTED", "acc": acc, "count": cnt}
        elif dim in weakness_detection.get("needs_attention", []):
            all_dimensions[dim] = {"status": "NEEDS_ATTENTION", "acc": acc, "count": cnt}
        else:
            all_dimensions[dim] = {"status": "OK", "acc": acc, "count": cnt}
    
    eval_result = {
        "user_id": user_id,
        "topic_id": "1.2",
        "total_attempted": total_attempted,
        "total_correct": total_correct,
        "overall_acc": total_correct / total_attempted if total_attempted > 0 else 0,
        "dimension_counts": dimension_counts,
        "dimension_correct": dimension_correct,
        "dimension_acc": dimension_acc,
        "data_sufficiency": data_sufficiency,
        "weakness_detection": weakness_detection,
        "all_dimensions": all_dimensions,
        "timestamp": datetime.now().isoformat()
    }
    
    # 清空 buffer
    user_data["eval_buffer"] = []
    
    # 更新 eval_count
    user_data["eval_count"] = user_data.get("eval_count", 0) + 1
    
    # 更新 weakness_state
    user_data["weakness_state"] = {
        "primary_dimension": {
            "dimension": weakness_detection["weak_dimension"],
            "status": weakness_detection["status"],
            "consecutive_pass": 0,
            "last_update": datetime.now().isoformat()
        } if weakness_detection["weak_dimension"] else None,
        "all_dimensions": all_dimensions
    }
    
    return {"eval_result": eval_result}


@app.post("/challenge/start")
def challenge_start(req: ChallengeStartRequest):
    """
    开始 Challenge
    - 针对 weak_dimension 抽题
    """
    init_user(req.user_id)
    
    if req.pool_policy == "strict":
        pool = get_dim_questions(req.weak_dimension)
    else:
        pool = get_dim_questions(req.weak_dimension)
    
    import random
    questions = random.sample(pool, min(4, len(pool)))
    
    # 记录已用题目
    user_data = users_db[req.user_id]
    user_data["entered_challenge"] = True
    for q in questions:
        user_data["used_questions"].add(q["question_id"])
    
    primary_dim = user_data["weakness_state"]["primary_dimension"]
    consecutive_pass = primary_dim.get("consecutive_pass", 0) if primary_dim else 0
    
    return {
        "session_id": str(uuid.uuid4()),
        "topic_id": req.topic_id,
        "weak_dimension": req.weak_dimension,
        "current_state": {
            "status": primary_dim.get("status", "WEAK_DETECTED") if primary_dim else "WEAK_DETECTED",
            "consecutive_pass": consecutive_pass,
            "target": 2
        },
        "questions": [
            {
                "question_id": q["question_id"],
                "stem": q["stem"],
                "options": q["options"]
            }
            for q in questions
        ],
        "total_count": len(questions)
    }


@app.post("/challenge/submit")
def challenge_submit(req: ChallengeSubmitRequest):
    """
    提交 Challenge
    - 后端计算 correct
    - v1.2.1 修正 consecutive_pass 逻辑
    """
    init_user(req.user_id)
    
    # ========================================================================
    # 后端计算 correct
    # ========================================================================
    processed_attempts = []
    for attempt in req.attempts:
        q = get_question_by_id(attempt.question_id)
        if not q:
            raise HTTPException(status_code=400, detail=f"Question {attempt.question_id} not found")
        
        is_correct = attempt.selected_option.upper() == q.get("correct_option", "").upper()
        processed_attempts.append({
            "question_id": attempt.question_id,
            "selected_option": attempt.selected_option,
            "correct": is_correct
        })
    
    # ========================================================================
    # 计算正确率
    # ========================================================================
    attempts = processed_attempts
    attempted = len(attempts)
    correct = sum(1 for a in attempts if a["correct"])
    acc = correct / attempted if attempted > 0 else 0
    
    # ========================================================================
    # update_weakness_state（v1.2.1 修正版）
    # ========================================================================
    user_data = users_db[req.user_id]
    primary_dim = user_data["weakness_state"]["primary_dimension"]
    
    if primary_dim is None:
        return {
            "challenge_acc": acc,
            "new_state": {
                "status": "NO_WEAKNESS",
                "consecutive_pass": 0,
                "message": "无弱点"
            }
        }
    
    current_status = primary_dim.get("status", "WEAK_DETECTED")
    consecutive_pass = primary_dim.get("consecutive_pass", 0)
    
    if current_status == "STABLE":
        # STABLE 不回落
        if acc >= CHALLENGE_PASS_THRESHOLD:
            message = "继续保持稳定"
        else:
            message = "检测到波动，建议巩固练习"
        
        new_status = "STABLE"
        new_consecutive_pass = 0
    else:
        # WEAK_DETECTED 状态
        if acc >= CHALLENGE_PASS_THRESHOLD:
            # 达标
            consecutive_pass += 1
            if consecutive_pass >= CONSECUTIVE_PASS_REQUIRED:
                new_status = "STABLE"
                new_consecutive_pass = 0
                message = "恭喜！弱点已解除"
            else:
                new_status = "WEAK_DETECTED"
                new_consecutive_pass = consecutive_pass
                message = f"还需 {CONSECUTIVE_PASS_REQUIRED - consecutive_pass} 次达标"
        else:
            # 未达标
            # v1.2.1 修正：
            # - consecutive_pass == 0 时保持 0
            # - consecutive_pass == 1 时保持 1（心理缓冲）
            if consecutive_pass == 0:
                new_consecutive_pass = 0
            else:  # consecutive_pass == 1
                new_consecutive_pass = 1
            
            new_status = "WEAK_DETECTED"
            message = "继续加油"
    
    # 保存状态
    user_data["weakness_state"]["primary_dimension"] = {
        "dimension": req.weak_dimension,
        "status": new_status,
        "consecutive_pass": new_consecutive_pass,
        "last_update": datetime.now().isoformat()
    }
    
    # 更新 metrics
    user_data["challenge_count"] = user_data.get("challenge_count", 0) + 1
    
    # 更新时间戳
    now_iso = datetime.now().isoformat()
    user_data["last_activity_at"] = now_iso
    user_data["last_challenge_at"] = now_iso
    
    if new_status == "STABLE":
        stable_dims = user_data.get("stable_dimensions", [])
        if req.weak_dimension not in stable_dims:
            stable_dims.append(req.weak_dimension)
            user_data["stable_dimensions"] = stable_dims
    
    return {
        "challenge_acc": acc,
        "new_state": {
            "status": new_status,
            "consecutive_pass": new_consecutive_pass,
            "message": message
        }
    }


@app.get("/dashboard")
def get_dashboard(user_id: str):
    """获取 Dashboard 数据"""
    init_user(user_id)
    user_data = users_db[user_id]
    
    buffer_count = len(user_data["eval_buffer"])
    primary_dim = user_data["weakness_state"]["primary_dimension"]
    
    result = {
        "topic_id": "1.2",
        "topic_name": "Opportunity Cost",
        "eval_buffer": {
            "current": buffer_count,
            "max": 3
        },
        "primary_weakness": None,
        "all_dimensions_status": user_data["weakness_state"]["all_dimensions"]
    }
    
    if primary_dim and primary_dim.get("status") == "WEAK_DETECTED":
        dim = primary_dim["dimension"]
        dim_display = {
            "COST_STRUCTURE": "成本结构",
            "ALT_LOGIC": "比较逻辑",
            "SUNK_FILTER": "沉没成本"
        }
        result["primary_weakness"] = {
            "dimension": dim,
            "dimension_cn": dim_display.get(dim, dim),
            "status": "WEAK_DETECTED",
            "consecutive_pass": primary_dim.get("consecutive_pass", 0),
            "remaining": 2 - primary_dim.get("consecutive_pass", 0),
            "last_update": primary_dim.get("last_update")
        }
    elif primary_dim and primary_dim.get("status") == "STABLE":
        dim = primary_dim["dimension"]
        dim_display = {
            "COST_STRUCTURE": "成本结构",
            "ALT_LOGIC": "比较逻辑",
            "SUNK_FILTER": "沉没成本"
        }
        result["primary_weakness"] = {
            "dimension": dim,
            "dimension_cn": dim_display.get(dim, dim),
            "status": "STABLE",
            "consecutive_pass": 0,
            "message": "已稳定"
        }
    
    return result


@app.get("/metrics")
def get_metrics():
    """
    获取课堂试验指标统计（基于内存数据）
    
    聚合 topic_id=1.2 的所有用户数据
    """
    if not users_db:
        return {
            "topic_id": "1.2",
            "message": "No data yet",
            "metrics": {
                "total_users": 0,
                "users_with_eval_count_ge_1": 0,
                "users_with_eval_count_ge_2": 0,
                "users_entered_challenge": 0,
                "users_completed_challenge_ge_1": 0,
                "users_became_stable": 0,
                "eval_trigger_rate": 0.0,
                "challenge_entry_rate": 0.0,
                "challenge_completion_rate": 0.0
            },
            "per_user": []
        }
    
    # 计算聚合指标
    total_users = len(users_db)
    users_with_eval_ge_1 = 0
    users_with_eval_ge_2 = 0
    users_entered_challenge = 0
    users_completed_challenge_ge_1 = 0
    users_became_stable = 0
    
    per_user = []
    
    for user_id, data in users_db.items():
        micro_count = data.get("micro_count", 0)
        eval_count = data.get("eval_count", 0)
        entered_challenge = data.get("entered_challenge", False)
        challenge_count = data.get("challenge_count", 0)
        stable_dims = data.get("stable_dimensions", [])
        
        # 获取当前 primary_dimension 状态
        primary_dim = data.get("weakness_state", {}).get("primary_dimension")
        last_primary_dim = primary_dim.get("dimension") if primary_dim else None
        primary_status = primary_dim.get("status") if primary_dim else None
        
        # 统计
        if eval_count >= 1:
            users_with_eval_ge_1 += 1
        if eval_count >= 2:
            users_with_eval_ge_2 += 1
        if entered_challenge:
            users_entered_challenge += 1
        if challenge_count >= 1:
            users_completed_challenge_ge_1 += 1
        # STABLE: stable_dimensions 非空 或 primary_status == "STABLE"
        if stable_dims or primary_status == "STABLE":
            users_became_stable += 1
        
        per_user.append({
            "user_id": user_id,
            "micro_count": micro_count,
            "eval_count": eval_count,
            "entered_challenge": entered_challenge,
            "challenge_count": challenge_count,
            "stable_dimensions": stable_dims,
            "last_primary_dimension": last_primary_dim
        })
    
    # 计算比率
    eval_trigger_rate = users_with_eval_ge_1 / total_users if total_users > 0 else 0.0
    challenge_entry_rate = users_entered_challenge / users_with_eval_ge_1 if users_with_eval_ge_1 > 0 else 0.0
    challenge_completion_rate = users_completed_challenge_ge_1 / users_entered_challenge if users_entered_challenge > 0 else 0.0
    
    return {
        "topic_id": "1.2",
        "updated_at": datetime.now().isoformat(),
        "metrics": {
            "total_users": total_users,
            "users_with_eval_count_ge_1": users_with_eval_ge_1,
            "users_with_eval_count_ge_2": users_with_eval_ge_2,
            "users_entered_challenge": users_entered_challenge,
            "users_completed_challenge_ge_1": users_completed_challenge_ge_1,
            "users_became_stable": users_became_stable,
            "eval_trigger_rate": round(eval_trigger_rate, 3),
            "challenge_entry_rate": round(challenge_entry_rate, 3),
            "challenge_completion_rate": round(challenge_completion_rate, 3)
        },
        "per_user": per_user
    }


@app.get("/health")
def health_check():
    """健康检查端点"""
    return {"status": "ok"}


# v1.3 feedback endpoint
@app.post("/question/feedback")
def get_question_feedback(req: FeedbackRequest) -> FeedbackResponse:
    """
    获取题目即时反馈（包含解释）
    - 根据 question_id 查找题目
    - 返回 correct_option, is_correct, explain 等信息
    """
    # 用索引快速查找题目
    question = QUESTION_INDEX.get(req.question_id)
    
    if not question:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Question {req.question_id} not found"}
        )
    
    # 打印调试信息
    print(f"Feedback request for question_id={req.question_id}")
    print(f"  - ai_logic_flow: {question.get('ai_logic_flow')}")
    print(f"  - ai_common_trap: {question.get('ai_common_trap')}")
    print(f"  - distractor_analysis: {question.get('distractor_analysis')}")
    
    # 获取正确答案
    correct_option = question.get("answer") or question.get("correct_option", "")
    selected = req.selected_option.upper()
    correct = correct_option.upper()
    is_correct = selected == correct
    
    # 获取解释字段
    ai_logic_flow = question.get("ai_logic_flow") or ""
    ai_common_trap = question.get("ai_common_trap") or ""
    distractor_analysis = question.get("distractor_analysis") or {}
    
    # Fallback: 如果 logic_flow 为空，尝试使用 explanation 字段
    if not ai_logic_flow and question.get("explanation"):
        ai_logic_flow = question.get("explanation")
    
    # 构建 why_selected_wrong（如果选错）
    why_selected_wrong = ""
    if not is_correct and selected in distractor_analysis:
        why_selected_wrong = distractor_analysis[selected]
    
    # 构建 why_correct_right
    why_correct_right = ai_logic_flow if ai_logic_flow else ""
    
    # 构建返回
    return FeedbackResponse(
        question_id=req.question_id,
        selected_option=req.selected_option,
        correct_option=correct_option,
        is_correct=is_correct,
        explain={
            "logic_flow": ai_logic_flow,
            "common_trap": ai_common_trap,
            "why_selected_wrong": why_selected_wrong,
            "why_correct_right": why_correct_right
        },
        source={
            "year": question.get("year"),
            "paper": question.get("paper")
        }
    )


# Debug endpoint
@app.get("/debug/question/{question_id}")
def debug_question(question_id: str):
    """调试端点：返回题目的完整合并后对象"""
    question = QUESTION_INDEX.get(question_id)
    
    if not question:
        raise HTTPException(
            status_code=404,
            detail={"message": f"Question {question_id} not found"}
        )
    
    return {
        "question_id": question_id,
        "full_question": question,
        "fields_available": list(question.keys())
    }


@app.get("/")
def root():
    return {
        "message": "DSE Economics - Opportunity Cost MVP API (v1.3)",
        "endpoints": [
            "GET /health",
            "POST /micro/start",
            "POST /micro/submit",
            "GET /evaluation",
            "POST /challenge/start",
            "POST /challenge/submit",
            "GET /dashboard",
            "GET /metrics",
            "POST /question/feedback",
            "GET /debug/question/{question_id}"
        ]
    }
