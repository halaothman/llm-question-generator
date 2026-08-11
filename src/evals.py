"""حساب المقاييس الآلية (عربي فقط): BLEU و BERTScore و Perplexity و F1.

يُستدعى من storage.py عند حفظ الأسئلة في outputs/.
candidate = نص السؤال المُولَّد (MCQ: السؤال + الخيارات)، reference = النص المصدر (source_text).
"""
import math
import re

# نموذج BERTScore — متعدد اللغات ويدعم العربية
_AR_BERT_MODEL = "bert-base-multilingual-cased"
# نموذج Perplexity — GPT-2 عربي
_AR_PPL_MODEL = "aubmindlab/araGPT2-base"


def _default_metrics() -> dict:
    """قيم صفرية عند غياب النص أو فشل الحساب."""
    return {
        "bleu": 0.0,
        "bert_score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "perplexity": 0.0,
    }


def perplexity_list(texts):
    """حساب Perplexity لكل نص — كلما انخفضت القيمة كان النص «أكثر طبيعية» للنموذج."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        tok = AutoTokenizer.from_pretrained(_AR_PPL_MODEL)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        mdl = AutoModelForCausalLM.from_pretrained(
            _AR_PPL_MODEL,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        ppl = []
        for t in texts:
            # نص قصير جداً لا يُعطي loss موثوق
            if not t or len(t.strip()) < 3:
                ppl.append(0.0)
                continue

            normalized_text = normalize_text_for_perplexity(t)
            if not normalized_text or len(normalized_text.strip()) < 3:
                ppl.append(0.0)
                continue

            try:
                enc = tok(
                    normalized_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = mdl(**enc, labels=enc["input_ids"])
                    loss = outputs.loss
                    perplexity = math.exp(loss.item())
                    # قيم شاذة جداً → 0.0 (فشل/عدم ثقة)
                    if perplexity >= 1000.0:
                        ppl.append(0.0)
                    else:
                        ppl.append(perplexity)
            except Exception:
                ppl.append(0.0)
        return ppl
    except Exception:
        # فشل تحميل النموذج — صفر لكل مدخل
        return [0.0] * len(texts)


def tokenize_arabic(text):
    """تقسيم النص إلى كلمات عربية (لـ Precision/Recall/F1)."""
    # إبقاء الحروف العربية والمسافات فقط
    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]",
        " ",
        text,
    )
    return text.split()


def normalize_text_for_perplexity(text):
    """تنظيف السؤال قبل إدخاله لـ araGPT2."""
    if not text:
        return ""

    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\b[A-Za-z]+\b", "[اسم]", text)  # مصطلحات لاتينية → placeholder
    text = re.sub(r"\d+", "[رقم]", text)
    return re.sub(r"\s+", " ", text).strip()


def calculate_bleu_score(candidate, reference):
    """BLEU: تشابه n-gram بين السؤال والمصدر (0–1)."""
    try:
        from sacrebleu import BLEU

        if not candidate or not reference:
            return 0.0

        bleu = BLEU(effective_order=True)
        score = bleu.sentence_score(candidate, [reference])
        return score.score / 100.0  # sacrebleu يُرجع 0–100
    except Exception:
        return 0.0


def calculate_bert_score(candidate, reference):
    """BERTScore: تشابه دلالي عبر embeddings — نُخزّن F1 فقط."""
    try:
        from bert_score import score as bert_score

        if not candidate or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        p, r, f1 = bert_score(
            [candidate],
            [reference],
            model_type=_AR_BERT_MODEL,
            verbose=False,
        )
        return {
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1_score": float(f1[0]),
        }
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


def calculate_precision_recall_f1(candidate, reference):
    """تداخل كلمات عربية بين السؤال والمصدر (bag-of-words)."""
    try:
        if not candidate or not reference:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        candidate_tokens = tokenize_arabic(candidate)
        reference_tokens = tokenize_arabic(reference)

        if not candidate_tokens or not reference_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        candidate_set = set(candidate_tokens)
        reference_set = set(reference_tokens)
        intersection = candidate_set.intersection(reference_set)

        # precision: كم من كلمات السؤال وُجدت في المصدر
        precision = len(intersection) / len(candidate_set) if candidate_set else 0.0
        # recall: كم من كلمات المصدر ظهرت في السؤال
        recall = len(intersection) / len(reference_set) if reference_set else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall
            else 0.0
        )

        return {"precision": precision, "recall": recall, "f1_score": f1_score}
    except Exception:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}


def calculate_all_metrics(question_text, source_text):
    """نقطة الدخول: حساب الستة مقاييس لسؤال واحد مقابل source_text.

    Args:
        question_text: نص السؤال المُولَّد (لـ MCQ يُدمج مع الخيارات في storage.py).
        source_text: النص المرجعي (ملف Vanilla أو ملف+RAG).

    Returns:
        dict: bleu, bert_score (F1), precision, recall, f1_score, perplexity.
    """
    if not question_text or not source_text:
        return _default_metrics()

    bert_scores = calculate_bert_score(question_text, source_text)
    prf_scores = calculate_precision_recall_f1(question_text, source_text)
    perplexity_scores = perplexity_list([question_text])

    return {
        "bleu": round(calculate_bleu_score(question_text, source_text), 4),
        "bert_score": round(bert_scores["f1_score"], 4),
        "precision": round(prf_scores["precision"], 4),
        "recall": round(prf_scores["recall"], 4),
        "f1_score": round(prf_scores["f1_score"], 4),
        "perplexity": round(float(perplexity_scores[0]), 4) if perplexity_scores else 0.0,
    }
