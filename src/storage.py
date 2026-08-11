"""حفظ ملفات الأسئلة JSON تحت outputs/<مصدر>/ وبنية metadata للرسالة."""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .question_counts import count_llm_questions, iter_llm_questions

LEGACY_AGGREGATE_PATH = "outputs/questions.json"


def clean_source_stem(source_file: str) -> str:
    """اسم مجلد/لاحقة الملف من اسم المرفوع (نفس منطق الحفظ والتحليل)."""
    return Path(source_file).stem.replace(" ", "_").replace(".", "_")


def model_method_short(model_name: str, method: str) -> Tuple[str, str]:
    model_short = "llama" if "llama" in model_name.lower() else "qwen"
    method_short = "vanilla" if method.lower() == "vanilla" else "rag"
    return model_short, method_short


def questions_json_basename(
    model_name: str,
    method: str,
    source_file: str,
    *,
    revised: bool = True,
) -> str:
    """
    اسم ملف JSON للأسئلة.

    revised=True  → questions_{model}_{method}_{source}_new.json  (توليد لاحق / بعد التحسين)
    revised=False → questions_{model}_{method}_{source}.json      (نسخ quizz-2 / قبل)
    """
    model_short, method_short = model_method_short(model_name, method)
    clean_source = clean_source_stem(source_file)
    base = f"questions_{model_short}_{method_short}_{clean_source}"
    return f"{base}_new.json" if revised else f"{base}.json"


def questions_json_path(
    model_name: str,
    method: str,
    source_file: str,
    *,
    revised: bool = True,
) -> Path:
    """مسار كامل لملف الأسئلة داخل outputs/<clean_source>/."""
    clean_source = clean_source_stem(source_file)
    name = questions_json_basename(model_name, method, source_file, revised=revised)
    return Path("outputs") / clean_source / name


def load_questions_file(
    model_name: str,
    method: str,
    source_file: str,
    *,
    prefer_revised: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    تحميل JSON محفوظ. يجرب _new.json أولاً (أو العكس) ثم النسخة الأخرى إن وُجدت.
    """
    order = (True, False) if prefer_revised else (False, True)
    for revised in order:
        path = questions_json_path(model_name, method, source_file, revised=revised)
        if not path.is_file():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return None


def save_group(group: str, payload: dict) -> None:
    """احتياطي: إلحاق دفعة في outputs/questions.json (مجموعة A=Vanilla، B=RAG)."""
    os.makedirs("outputs", exist_ok=True)
    data: dict = {}
    if os.path.exists(LEGACY_AGGREGATE_PATH):
        try:
            with open(LEGACY_AGGREGATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = {}
    if group not in data:
        data[group] = []
    data[group].append(payload)
    with open(LEGACY_AGGREGATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def metrics_candidate_text(qtype: str, item: dict) -> str:
    """نص candidate للمقاييس: MCQ = السؤال + الخيارات؛ صح/خطأ = السؤال فقط."""
    qtext = (item.get("q") or item.get("question") or "").strip()
    if qtype != "mcq":
        return qtext
    options = [
        str(option).strip()
        for option in item.get("options") or []
        if str(option).strip()
    ]
    if not options:
        return qtext
    return f"{qtext} {' '.join(options)}".strip()


def save_questions_separate_file(
    questions_data: Dict[str, Any],
    model_name: str,
    method: str,
    source_file: str,
    lang: str = "ar",
) -> Tuple[str, int]:
    """حفظ الأسئلة في outputs/<مصدر>/ مع حساب المقاييس الآلية."""
    from .evals import calculate_all_metrics

    clean_source = clean_source_stem(source_file)
    source_folder_path = Path("outputs") / clean_source
    source_folder_path.mkdir(parents=True, exist_ok=True)

    filename = questions_json_basename(model_name, method, source_file, revised=True)
    file_path = source_folder_path / filename

    data: Dict[str, Any] = {
        "metadata": {
            "version": "1.0",
            "source_file": source_file,
            "total_questions": 0,
            "model": model_name,
            "method": method,
            "created_from": "user_upload",
        },
        "questions": [],
    }

    question_id = 1
    source_text = questions_data.get("source_text", "")

    def passage_snippet() -> str:
        return source_text[:200] + "..." if len(source_text) > 200 else source_text

    def default_metrics() -> dict:
        return {
            "perplexity": 0.0,
            "bleu": 0.0,
            "bert_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
        }

    def metrics_for(question_text: str) -> dict:
        try:
            return calculate_all_metrics(question_text, source_text)
        except Exception as e:
            print(f"خطأ في حساب المقاييس: {e}")
            return default_metrics()

    gen_block = {
        "model": model_name,
        "method": method,
    }
    source_block = {
        "file": source_file,
        "page": 1,
        "passage": passage_snippet(),
    }

    for qtype, item in iter_llm_questions(questions_data):
        qtext = item.get("q") or item.get("question") or ""
        record: Dict[str, Any] = {
            "id": f"q_{question_id:03d}",
            "type": qtype,
            "question": qtext,
            "source": source_block,
            "generation": gen_block,
            "metrics": metrics_for(metrics_candidate_text(qtype, item)),
        }
        if qtype == "mcq":
            record["options"] = item.get("options", [])
            record["correct_answer"] = item.get("answer", "")
        else:
            record["correct_answer"] = item.get("answer", False)
        data["questions"].append(record)
        question_id += 1

    total = count_llm_questions(questions_data)
    if total == 0:
        raise ValueError("لا توجد أسئلة للحفظ")

    data["metadata"]["total_questions"] = total

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    if not file_path.is_file():
        raise IOError(f"فشل في حفظ الملف: {file_path}")

    return filename, total
