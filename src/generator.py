"""بناء prompts، استدعاء Ollama، التحقق من JSON، وتوليد الأسئلة (Vanilla/RAG)."""
import json
import re
import time
from difflib import SequenceMatcher
from typing import Literal, Optional

import ollama
from langdetect import detect

# رسالة النظام المشتركة بين Vanilla و RAG (قواعد JSON، MCQ، TF)
SYS_AR = (
    "أنت معلّم خبير في إنشاء الأسئلة. مهمتك: إنشاء أكبر عدد ممكن من الأسئلة المختلفة "
    "والدقيقة من النص المعطى (بحد أقصى 20 سؤال). يجب أن تكون الأسئلة مزيجاً من أسئلة "
    "اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ. كل سؤال يجب أن يكون فريداً "
    "ومستنداً للنص. **مهم جداً جداً:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون "
    "إجابة حقيقية ومناسبة ومحددة من النص. **ممنوع تماماً ومحظور** كتابة 'خيار 1' أو "
    "'خيار 2' أو 'خيار 3' أو 'خيار 4' أو 'جواب 1' أو 'إجابة أ' أو أي خيارات عامة. "
    "**ممنوع تماماً** استخدام أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى، ما) "
    "في أسئلة الصح/الخطأ. أعد JSON فقط بدون أي نص إضافي."
)

# رفض خيارات placeholder مثل «خيار 1» أو «option 2»
_GENERIC_OPTION = re.compile(
    r"^(خيار\s*[1-4]|جواب\s*[1-4]|إجابة\s*[أ-د]|option\s*[1-4]|choice\s*[1-4])$",
    re.I,
)
# ممنوع في بداية سؤال صح/خطأ (يجب «هل» أو جملة خبرية)
_TF_INTERROGATIVE = (
    "من",
    "أين",
    "ماذا",
    "متى",
    "كيف",
    "لماذا",
    "ما",
    "what",
    "where",
    "why",
    "how",
    "when",
    "who",
    "which",
)

# ── اكتشاف اللغة وبناء البرومبت ──────────────────────────────────────────────

def detect_lang(text: str) -> Literal["ar", "en"]:
    """اكتشاف لغة النص (للعرض والتقييم — البرومبت عربي دائماً)."""
    try:
        return "ar" if detect(text) == "ar" else "en"
    except Exception:
        return "ar"


def build_prompt_vanilla(text: str) -> str:
    """برومبت Vanilla: MCQ + TF من الملف المرفوع فقط (بدون RAG)."""
    return f"""{SYS_AR}

=== النص المرفوع (المصدر الوحيد) ===
{text}

**مهم جداً جداً:** هذا هو المصدر الوحيد المتاح. يجب أن تكون جميع الأسئلة والخيارات مستندة **فقط** إلى هذا النص المرفوع. **ممنوع تماماً** استخدام أي معلومات من خارج هذا النص.

التعليمات الدقيقة:
1. **استخدم النص المرفوع أعلاه فقط** - هذا هو المصدر الوحيد المتاح. أنشئ أكبر عدد ممكن من الأسئلة المختلفة من هذا النص فقط (بحد أقصى 20 سؤال)
2. يجب أن تكون الأسئلة مزيجاً من أسئلة اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ
3. لا يوجد عدد محدد لكل نوع - أنشئ أكبر عدد ممكن من الأسئلة المختلفة
4. كل سؤال صح/خطأ يجب أن يكون صياغة خبرية أو يبدأ بـ "هل"
5. كل سؤال يجب أن يكون فريداً ومستنداً **فقط** إلى النص المرفوع أعلاه
6. **مهم جداً جداً - خيارات الاختيار من متعدد:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون إجابة حقيقية ومناسبة ومحددة **فقط من النص المرفوع أعلاه**. **ممنوع تماماً** كتابة "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو "جواب 1" أو "إجابة أ" أو "اختيار ب" أو أي خيارات عامة. **ممنوع تماماً** استخدام معلومات من خارج النص المرفوع. اكتب خيارات حقيقية ومناسبة ومحددة من النص المرفوع فقط
7. **تذكير قوي جداً:** إذا كتبت "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" في أي سؤال، أو إذا استخدمت معلومات من خارج النص المرفوع، فإن الإجابة ستكون خاطئة تماماً. يجب أن تكون جميع الخيارات إجابات حقيقية من النص المرفوع فقط
8. **مهم جداً جداً - تجنب التكرار:** لا تكرر الأسئلة أو الخيارات. كل سؤال يجب أن يكون مختلفاً تماماً عن الأسئلة الأخرى
9. أعد JSON فقط بدون أي نص إضافي
10. تأكد من إغلاق جميع الأقواس والفواصل بشكل صحيح
11. استخدم كلمات عربية فقط في الأسئلة
12. **مهم جداً لأسئلة الصح/الخطأ:** لا تستخدم أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى) في أسئلة الصح/الخطأ. يمكنك استخدام "هل" في بداية السؤال أو استخدام جملة خبرية تحتمل الإيجاب أو النفي

تنسيق JSON المطلوب:
{{
  "mcq": [
    {{"q": "سؤال اختيار من متعدد من النص", "options": ["إجابة حقيقية من النص", "إجابة حقيقية أخرى من النص", "إجابة حقيقية ثالثة من النص", "إجابة حقيقية رابعة من النص"], "answer": "الإجابة الصحيحة"}},
    // ... المزيد من أسئلة الاختيار من متعدد (حسب ما يتوفر في النص)
  ],
  "tf": [
    {{"q": "جملة خبرية من النص", "answer": true}},
    {{"q": "جملة خبرية من النص", "answer": false}},
    // ... المزيد من أسئلة الصح/الخطأ (حسب ما يتوفر في النص)
  ]
}}

**مهم:** أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي). لا يوجد حد أدنى أو عدد محدد لكل نوع.

ابدأ الآن:"""


# ── دمج نصوص RAG (للبرومبت والمقاييس) ───────────────────────────────────────

def combined_retrieved_text(retrieved: list) -> str:
    """دمج نصوص المقاطع المسترجعة."""
    parts = []
    for item in retrieved or []:
        text = (item.get("text") if isinstance(item, dict) else "") or ""
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def combined_rag_source_text(upload_text: str, retrieved: list) -> str:
    """دمج الملف المرفوع مع المقاطع المسترجعة — مصدر موحّد للتوليد والمقاييس."""
    parts = []
    upload = (upload_text or "").strip()
    if upload:
        parts.append(upload)
    retrieved_block = combined_retrieved_text(retrieved)
    if retrieved_block:
        parts.append(retrieved_block)
    return "\n\n".join(parts)


def build_prompt_rag(text: str, retrieved: Optional[list] = None) -> str:
    """برومبت RAG: الملف المرفوع + مقاطع FAISS (أو نص فقط إن لم يُسترجَع شيء)."""
    retrieved = retrieved or []
    has_rag = bool(retrieved)

    if has_rag:
        # تنسيق كل مقطع مسترج: اسم ملف + رقم مقطع + أول 500 حرف
        retrieved_texts = []
        for i, r in enumerate(retrieved, 1):
            passage_text = r["text"][:500]
            filename = r.get("filename", "مصدر غير معروف")
            chunk = (r.get("metadata") or {}).get("chunk_index")
            chunk_part = f" · مقطع {chunk}" if chunk is not None else ""
            retrieved_texts.append(f"[مصدر {i}: {filename}{chunk_part}]\n{passage_text}")
        combined_context = f"""=== النص الأساسي (المصدر الرئيسي) ===
{text}

=== معلومات إضافية من مصادر مشابهة (تم استرجاعها باستخدام RAG) ===
{chr(10).join(retrieved_texts)}

**مهم جداً:** المحتوى أعلاه يتكون من:
1. **النص الأساسي**: المصدر الرئيسي الذي يجب أن يكون الأساس لجميع الأسئلة
2. **المصادر الإضافية**: معلومات مكملة من مصادر مشابهة تم استرجاعها تلقائياً

**يجب استخدام المعلومات من كلا المصدرين معاً** لإنشاء أسئلة دقيقة ومتنوعة.
"""
        instruction_1 = "1. **استخدم المعلومات من النص الأساسي والمصادر الإضافية معاً** لإنشاء أسئلة دقيقة وعميقة. ابدأ بالنص الأساسي كأساس، ثم استخدم المصادر الإضافية لإثراء الخيارات والمعلومات"
        instruction_2 = "2. **لكل سؤال اختيار من متعدد:** استخرج الخيارات من **جميع المصادر المتاحة** (النص الأساسي + المصادر الإضافية). يمكن أن تأتي الخيارات من أي من المصادر المتاحة، لكن يجب أن تكون جميعها حقيقية ومحددة من المحتوى"
    else:
        # بدون مقاطع مسترجعة: نفس منطق Vanilla لكن بصياغة «المحتوى المتاح»
        combined_context = f"""=== المحتوى المتاح ===
{text}
"""
        instruction_1 = "1. أنشئ أسئلة دقيقة وعميقة من النص أعلاه"
        instruction_2 = "2. كل سؤال يجب أن يستند إلى محتوى حقيقي من النص"

    return f"""{SYS_AR}

{combined_context}

التعليمات الدقيقة:
{instruction_1}
{instruction_2}
3. **لإنشاء خيارات الاختيار من متعدد:**
   - اقرأ جميع المصادر المتاحة بعناية (النص الأساسي + المصادر الإضافية إن وجدت)
   - استخرج معلومات محددة من كل مصدر
   - لكل سؤال، أنشئ 4 خيارات حقيقية من المعلومات الموجودة في **جميع المصادر المتاحة**
   - كل خيار يجب أن يكون إجابة حقيقية ومحددة من المحتوى المتاح
   - استخدم معلومات من مصادر مختلفة لإنشاء خيارات متنوعة
   - **مثال:** إذا كان السؤال عن "أين تقع المدينة؟"، فابحث في جميع المصادر عن أسماء أماكن حقيقية واكتبها كخيارات (مثل: "في الشمال"، "في الجنوب"، "على الساحل"، "في الداخل")
4. أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي)
5. يجب أن تكون الأسئلة مزيجاً من أسئلة اختيار من متعدد (4 خيارات لكل سؤال) وأسئلة صح/خطأ
6. لا يوجد عدد محدد لكل نوع - أنشئ أكبر عدد ممكن من الأسئلة المختلفة والفريدة
7. كل سؤال اختيار من متعدد يجب أن يحتوي على 4 خيارات بالضبط
8. **مهم جداً جداً - خيارات الاختيار من متعدد:** كل خيار في أسئلة الاختيار من متعدد يجب أن يكون إجابة حقيقية ومناسبة ومحددة من **أي من المصادر المتاحة** (النص الأساسي أو المصادر الإضافية). **ممنوع تماماً ومحظور** كتابة "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو "جواب 1" أو "إجابة أ" أو "اختيار ب" أو أي خيارات عامة أو رموز. اكتب خيارات حقيقية ومناسبة ومحددة من المحتوى المتاح فقط. كل خيار يجب أن يكون جملة أو عبارة كاملة من المحتوى
9. **تذكير قوي جداً:** إذا كتبت "خيار 1" أو "خيار 2" أو "خيار 3" أو "خيار 4" أو أي خيار عام في أي سؤال، فإن الإجابة ستكون خاطئة تماماً وسيتم رفض السؤال. يجب أن تكون جميع الخيارات إجابات حقيقية ومحددة من المحتوى المتاح. لا تستخدم أرقام أو رموز كخيارات
10. كل سؤال يجب أن يكون فريداً ومستنداً للمحتوى
11. **مهم جداً جداً - تجنب التكرار:** لا تكرر الأسئلة أو الخيارات. كل سؤال يجب أن يكون مختلفاً تماماً عن الأسئلة الأخرى. لا تستخدم نفس السؤال مرتين حتى لو بصياغة مختلفة. لا تستخدم نفس الخيارات في أسئلة مختلفة. كل خيار في كل سؤال يجب أن يكون فريداً ومختلفاً عن جميع الخيارات الأخرى
12. أعد JSON فقط بدون أي نص إضافي
13. تأكد من إغلاق جميع الأقواس والفواصل بشكل صحيح
14. استخدم كلمات عربية فقط في الأسئلة
15. تأكد من أن كل سؤال MCQ له 4 خيارات وليس أقل
16. **مهم جداً جداً لأسئلة الصح/الخطأ:** **ممنوع تماماً** استخدام أدوات الاستفهام (من، أين، ماذا، لماذا، كيف، متى، ما) في أسئلة الصح/الخطأ. يمكنك استخدام "هل" في بداية السؤال أو استخدام جملة خبرية تحتمل الإيجاب أو النفي. إذا استخدمت أداة استفهام في سؤال صح/خطأ، فإن السؤال سيكون خاطئاً تماماً

تنسيق JSON المطلوب:
{{
  "mcq": [
    {{"q": "سؤال اختيار من متعدد من المحتوى", "options": ["إجابة حقيقية من المحتوى", "إجابة حقيقية أخرى من المحتوى", "إجابة حقيقية ثالثة من المحتوى", "إجابة حقيقية رابعة من المحتوى"], "answer": "الإجابة الصحيحة"}},
    // ... المزيد من أسئلة الاختيار من متعدد (حسب ما يتوفر في المحتوى)
  ],
  "tf": [
    {{"q": "جملة خبرية من المحتوى", "answer": true}},
    {{"q": "جملة خبرية من المحتوى", "answer": false}},
    // ... المزيد من أسئلة الصح/الخطأ (حسب ما يتوفر في المحتوى)
  ]
}}

**مهم جداً:** أنشئ أكبر عدد ممكن من الأسئلة المختلفة (بحد أقصى 20 سؤال إجمالي). لا يوجد حد أدنى أو عدد محدد لكل نوع. ركّز على التنوع والجودة.

ابدأ الآن:"""


# ── استدعاء Ollama ────────────────────────────────────────────────────────────

def _extract_chat_response(resp) -> str:
    """استخراج نص content من رد Ollama (كائن أو dict)."""
    content = ""
    if hasattr(resp, "message"):
        content = getattr(resp.message, "content", None) or ""
    elif isinstance(resp, dict):
        msg = resp.get("message") or {}
        content = (msg.get("content") if isinstance(msg, dict) else "") or resp.get("content") or ""
    return (content or "").strip()


def check_and_pull_model(model_name: str) -> bool:
    """التأكد من وجود النموذج في Ollama (سحب بسيط عند الحاجة)."""
    try:
        listed = ollama.list()
        names = []
        if listed and hasattr(listed, "models"):
            for m in listed.models:
                if hasattr(m, "model"):
                    names.append(m.model)
        if model_name in names:
            return True
        print(f"Pulling {model_name}...")
        ollama.pull(model_name)
        return True
    except Exception as e:
        print(f"Model check failed: {e}")
        return False


def call_llama(
    prompt: str,
    max_retries: int = 3,
    model_name: str = "llama3.2:3b",
    temperature: float = 0.7,
    json_mode: bool = True,
) -> str:
    """استدعاء Ollama مع إعادة المحاولة؛ json_mode يفرض format=json."""
    check_and_pull_model(model_name)
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": {
                    "num_ctx": 16384,
                    "num_predict": 4096,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            }
            if json_mode:
                kwargs["format"] = "json"
            resp = ollama.chat(**kwargs)
            text = _extract_chat_response(resp)
            if text and (not json_mode or len(text.strip()) >= 30):
                return text
        except Exception as e:
            print(f"Ollama attempt {attempt + 1} failed: {e}")
        if attempt < max_retries - 1:
            time.sleep(3 * (attempt + 1))
    return ""


# ── التحقق من الجودة وإزالة التكرار ─────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    """نسبة تشابه نصّي بين سؤالين (SequenceMatcher)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def validate_question(question_data: dict, question_type: str, _source_text: str = "") -> tuple[bool, list[str]]:
    """التحقق من MCQ (4 خيارات، لا placeholder) أو TF (لا أداة استفهام في البداية)."""
    errors = []
    if question_type == "mcq":
        options = question_data.get("options", [])
        if len(options) != 4:
            errors.append("options count")
        seen = set()
        unique_opts = 0
        for i, opt in enumerate(options):
            s = str(opt).strip()
            if _GENERIC_OPTION.match(s):
                errors.append(f"generic option {i + 1}")
            key = s.lower()
            if key not in seen:
                unique_opts += 1
            seen.add(key)
        if unique_opts < 2:
            errors.append("too few unique options")
    elif question_type == "tf":
        q = str(question_data.get("q", "")).strip().lower()
        if not q:
            errors.append("empty question")
        for w in _TF_INTERROGATIVE:
            if q.startswith(w.lower()):
                errors.append(f"interrogative: {w}")
                break
    return len(errors) == 0, errors


def _dedupe_similar_questions(items: list) -> list:
    """إزالة أسئلة متشابهة جداً (>85%) ضمن قائمة واحدة."""
    unique, seen = [], []
    for item in items:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()
        if any(_similarity(q, s) > 0.85 for s in seen):
            continue
        seen.append(q.lower())
        unique.append(item)
    return unique


def remove_duplicate_questions(result: dict) -> dict:
    """إزالة أسئلة MCQ/TF المتشابهة جداً (>85%)."""
    for key in ("mcq", "tf"):
        items = result.get(key)
        if isinstance(items, list):
            result[key] = _dedupe_similar_questions(items)
    return result


def postprocess_questions(result: dict) -> dict:
    """إزالة التكرار وحذف الأسئلة غير الصالحة."""
    result = remove_duplicate_questions(result)
    for qtype in ("mcq", "tf"):
        items = result.get(qtype)
        if isinstance(items, list):
            result[qtype] = [
                q
                for q in items
                if isinstance(q, dict) and validate_question(q, qtype)[0]
            ]
    return result


# ── تحليل JSON وتوحيد الحقول ─────────────────────────────────────────────────

def _ensure_mcq_options(result: dict) -> None:
    """ضمان 4 خيارات لكل MCQ (حشو placeholder إن نقصت)."""
    for mcq in result.get("mcq") or []:
        if not isinstance(mcq, dict):
            continue
        opts = mcq.get("options")
        if not isinstance(opts, list):
            mcq["options"] = []
            opts = mcq["options"]
        while len(opts) < 4:
            opts.append(f"— {len(opts) + 1}")
        mcq["options"] = opts[:4]


def _repair_json_text(s: str) -> str:
    """إصلاح JSON مقطوع: إزالة markdown، إغلاق أقواس/فواصل ناقصة."""
    s = re.sub(r"```json|```", "", s, flags=re.I).strip()
    if s.count("{") > s.count("}"):
        s += "}" * (s.count("{") - s.count("}"))
    if s.count("[") > s.count("]"):
        s += "]" * (s.count("[") - s.count("]"))
    lines = s.split("\n")
    if lines and not lines[-1].strip().endswith(("}", "]", ",")):
        lines = lines[:-1]
        s = "\n".join(lines)
        if s.count("{") > s.count("}"):
            s += "\n}"
        if s.count("[") > s.count("]"):
            s += "\n]"
    return s.strip()


def _extract_json_blob(raw: str) -> str:
    """استخراج كائن JSON من نص قد يحتوي markdown أو شرحاً قبل/بعد."""
    s = re.sub(r"```json\s*|```", "", raw, flags=re.I).strip()
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s


def _loads_questions(raw: str) -> Optional[dict]:
    """محاولة json.loads على النص الخام، المستخرج، أو المُصلَح."""
    for candidate in (raw, _extract_json_blob(raw), _repair_json_text(_extract_json_blob(raw))):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _normalize_question_item(item: dict, qtype: str) -> Optional[dict]:
    """توحيد حقول سؤال واحد (q/options/answer) مع أسماء بديلة شائعة."""
    if not isinstance(item, dict):
        return None
    q = (item.get("q") or item.get("question") or "").strip()
    if not q:
        return None
    out = dict(item)
    out["q"] = q
    if qtype == "mcq":
        opts = out.get("options")
        if not isinstance(opts, list):
            opts = out.get("choices") or out.get("answers") or []
        out["options"] = [str(o).strip() for o in opts if str(o).strip()]
        ans = out.get("answer") or out.get("correct_answer") or out.get("correct")
        if ans is not None:
            out["answer"] = str(ans).strip()
    else:
        ans = out.get("answer")
        if isinstance(ans, str):
            low = ans.strip().lower()
            if low in ("true", "صح", "صحيح", "yes", "نعم"):
                out["answer"] = True
            elif low in ("false", "خطأ", "خطا", "no", "لا"):
                out["answer"] = False
    return out


def _normalize_question_list(items, qtype: str) -> list:
    """توحيد قائمة أسئلة mcq أو tf."""
    return [
        norm
        for item in (items or [])
        if (norm := _normalize_question_item(item, qtype))
    ]


def _coerce_question_payload(result: dict) -> dict:
    """توحيد أشكال JSON مختلفة (questions/metadata/أسماء حقول بديلة)."""
    nested = result.get("questions")
    if isinstance(nested, dict):
        for key in ("mcq", "tf", "true_false", "true/false"):
            if key in nested and key not in result:
                result[key] = nested[key]
    if isinstance(nested, list):
        mcq, tf = [], []
        for item in nested:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).lower()
            if kind in ("mcq", "multiple_choice", "choice"):
                mcq.append(item)
            elif kind in ("tf", "true_false", "boolean"):
                tf.append(item)
            elif item.get("options"):
                mcq.append(item)
            else:
                tf.append(item)
        result.setdefault("mcq", mcq)
        result.setdefault("tf", tf)

    for alt_tf in ("true_false", "true/false", "trueFalse"):
        if alt_tf in result and "tf" not in result:
            result["tf"] = result.pop(alt_tf)

    for qtype in ("mcq", "tf"):
        result[qtype] = _normalize_question_list(result.get(qtype), qtype)
    return _normalize_question_payload(result)


def _normalize_question_payload(result: dict) -> dict:
    """ضمان وجود قوائم mcq و tf (حتى لو فارغة)."""
    if not isinstance(result.get("mcq"), list):
        result["mcq"] = []
    if not isinstance(result.get("tf"), list):
        result["tf"] = []
    return result


def safe_json(
    s: str,
    source_text: str = "",
    model_name: str = "llama3.2:3b",
    lang: str = "ar",
    retrieved: Optional[list] = None,
):
    """تحليل JSON من استجابة النموذج → توحيد → تحقق → إزالة تكرار."""
    del source_text, model_name, lang, retrieved  # للتوافق مع مواقع الاستدعاء
    if not s or not s.strip():
        return None
    result = _loads_questions(s)
    if not isinstance(result, dict):
        return None
    result = _coerce_question_payload(result)
    _ensure_mcq_options(result)
    result = postprocess_questions(result)
    if not (result.get("mcq") or result.get("tf")):
        return None
    return result


# ── نقطة الدخول: توليد مع إعادة المحاولة ────────────────────────────────────

def generate_questions_with_retry(
    prompt: str,
    max_retries: int = 3,
    source_text: str = "",
    model_name: str = "llama3.2:3b",
    lang: str = "ar",
    retrieved: Optional[list] = None,
):
    """توليد الأسئلة مع إعادة المحاولة حتى JSON صالح."""
    for attempt in range(max_retries):
        response = call_llama(prompt, model_name=model_name, max_retries=2, json_mode=True)
        if not response:
            continue
        questions = safe_json(response, source_text, model_name, lang, retrieved)
        if (
            isinstance(questions, dict)
            and "mcq" in questions
            and "tf" in questions
            and (questions.get("mcq") or questions.get("tf"))
        ):
            return questions
        time.sleep(3)
    return None
