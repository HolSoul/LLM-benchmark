#!/usr/bin/env python
# Kaggle Benchmark: LoRA Fine-tuned LLMs Comparison
# 
# Как использовать на Kaggle:
# 1. Создай Notebook → File → Import Notebook → выбери этот файл
# 2. Runtime → Change runtime type → GPU T4 x2
# 3. Добавь секреты: HF_TOKEN (твой токен с huggingface.co/settings/tokens)
# 4. Add Data → Upload → выбери test_set.jsonl
# 5. Run all

# ============================================================
# 1. Установка зависимостей
# ============================================================
import subprocess, sys, os, json, time, gc, csv, warnings
warnings.filterwarnings("ignore")

# НЕ обновляем bitsandbytes через pip — на Kaggle это ломает torch
# Используем предустановленную версию

# Логин в Hugging Face Hub из Kaggle Secrets
hf_token = None
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
except Exception:
    pass

if not hf_token:
    hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("✅ HF_TOKEN найден, логин в Hugging Face выполнен")
else:
    print("⚠️ HF_TOKEN не найден. Gated модели (Gemma, Llama) не загрузятся.")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# 2. Конфигурация
# ============================================================

TEST_SET_PATH = "/kaggle/input/test_set.jsonl"
RESULTS_PATH = "/kaggle/working/benchmark_results.csv"
SUMMARY_PATH = "/kaggle/working/benchmark_summary.md"

# Модели для тестирования (все LoRA адаптеры с HF Hub)
# От маленьких к большим — чтобы прогревать GPU постепенно
MODELS = [
    {
        "name": "Qwen3-0.6B",
        "base_model": "Qwen/Qwen3-0.6B",
        "adapter": "HolSoul/Qwen3-0.6B-stomatology-patient_7ep",
        "size": "0.6B",
        "family": "Qwen"
    },
    {
        "name": "Gemma-3-1B",
        "base_model": "google/gemma-3-1b-it",
        "adapter": "HolSoul/gemma-3-1b-it-stomatology-patient_7ep",
        "size": "1B",
        "family": "Gemma"
    },
    {
        "name": "Llama-3.2-1B",
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "adapter": "HolSoul/Llama-3.2-1B-stomatology-patient_7ep",
        "size": "1B",
        "family": "Llama"
    },
    {
        "name": "Qwen3-1.7B",
        "base_model": "Qwen/Qwen3-1.7B",
        "adapter": "HolSoul/Qwen3-1.7B-stomatology-patient_7ep",
        "size": "1.7B",
        "family": "Qwen"
    },
    {
        "name": "YandexGPT-5-8B",
        "base_model": "yandex/YandexGPT-5-Lite-8B-instruct",
        "adapter": "HolSoul/YandexGPT-5-Lite-8B-stomatology-patient_5ep",
        "size": "8B",
        "family": "YandexGPT"
    },
    {
        "name": "DeepSeek-R1-Distill-8B",
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "adapter": "HolSoul/DeepSeek-R1-Distill-Llama-8B-stomatology-patient_5ep",
        "size": "8B",
        "family": "DeepSeek"
    },
]

# Параметры инференса
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3  # низкая температура для детерминированного сравнения

# ============================================================
# 3. Загрузка тестовых данных
# ============================================================
print("=" * 60)
print("ЗАГРУЗКА ТЕСТОВОГО НАБОРА")
print("=" * 60)

test_set = []
with open(TEST_SET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        test_set.append(json.loads(line.strip()))

print(f"Загружено {len(test_set)} вопросов")
categories = {}
for item in test_set:
    cat = item["category"]
    categories[cat] = categories.get(cat, 0) + 1
print(f"Категории: {categories}")

# ============================================================
# 4. Инференс
# ============================================================
print("\n" + "=" * 60)
print("ЗАПУСК БЕНЧМАРКА")
print("=" * 60)

all_results = []

for model_config in MODELS:
    print(f"\n{'─' * 50}")
    print(f"МОДЕЛЬ: {model_config['name']} ({model_config['size']})")
    print(f"  Base: {model_config['base_model']}")
    print(f"  Adapter: {model_config['adapter']}")
    print(f"{'─' * 50}")

    model, base, tokenizer = None, None, None
    model_results = []
    try:
        t_start = time.time()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        import torch

        tokenizer = AutoTokenizer.from_pretrained(
            model_config["adapter"],
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Адаптивная квантизация
        try:
            import bitsandbytes
            bnb_version = tuple(int(x) for x in bitsandbytes.__version__.split("."))
            print(f"  bitsandbytes v{bitsandbytes.__version__}")
        except Exception:
            bnb_version = (0, 0, 0)

        if bnb_version >= (0, 46, 1):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                model_config["base_model"],
                quantization_config=bnb_config, device_map="auto",
                trust_remote_code=True,
            )
        elif bnb_version >= (0, 40, 0):
            print(f"  4-bit недоступен, использую 8-bit")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            base = AutoModelForCausalLM.from_pretrained(
                model_config["base_model"],
                quantization_config=bnb_config, device_map="auto",
                trust_remote_code=True,
            )
        else:
            print(f"  Quantization недоступна, загрузка в full precision...")
            base = AutoModelForCausalLM.from_pretrained(
                model_config["base_model"],
                device_map="auto", torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        model = PeftModel.from_pretrained(base, model_config["adapter"])
        model.eval()
        load_time = time.time() - t_start
        print(f"  Загрузка: {load_time:.1f}с")

        # Инференс на всех вопросах
        for item in test_set:
            qid = item["id"]
            question = item["question"]

            try:
                # Используем chat_template модели вместо хардкода
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

                t_inf = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                inf_time = time.time() - t_inf

                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
                tokens_per_sec = tokens_generated / inf_time if inf_time > 0 else 0

                result = {
                    "model": model_config["name"],
                    "family": model_config["family"],
                    "size": model_config["size"],
                    "question_id": qid,
                    "category": item["category"],
                    "question": question,
                    "response": response,
                    "inference_time_s": round(inf_time, 2),
                    "tokens_generated": tokens_generated,
                    "tokens_per_sec": round(tokens_per_sec, 1),
                    "response_length_chars": len(response),
                }
                model_results.append(result)
                all_results.append(result)

                print(f"  [{qid:02d}] {inf_time:.1f}с | {tokens_per_sec:.0f} tok/с | {len(response)} символов")

            except Exception as e:
                print(f"  [{qid:02d}] ОШИБКА: {e}")
                model_results.append({
                    "model": model_config["name"],
                    "family": model_config["family"],
                    "size": model_config["size"],
                    "question_id": qid,
                    "category": item["category"],
                    "question": question,
                    "response": f"<ERROR: {str(e)}>",
                    "inference_time_s": 0,
                    "tokens_generated": 0,
                    "tokens_per_sec": 0,
                    "response_length_chars": 0,
                })

        # Средние показатели по модели
        if model_results:
            avg_time = sum(r["inference_time_s"] for r in model_results) / len(model_results)
            avg_tps = sum(r["tokens_per_sec"] for r in model_results) / len(model_results)
            avg_len = sum(r["response_length_chars"] for r in model_results) / len(model_results)
            print(f"\n  ИТОГО: среднее {avg_time:.1f}с/ответ | {avg_tps:.0f} tok/с | {avg_len:.0f} символов")

    except Exception as e:
        print(f"  КРИТИЧЕСКАЯ ОШИБКА загрузки модели: {e}")
        import traceback
        traceback.print_exc()

    # Очистка памяти
    finally:
        for v in [model, base, tokenizer]:
            if v is not None:
                del v
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  Память очищена")

# ============================================================
# 5. Сохранение результатов
# ============================================================
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)

with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
    if all_results:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

print(f"Результаты сохранены: {RESULTS_PATH}")
print(f"Всего записей: {len(all_results)}")

# ============================================================
# 6. Сводка
# ============================================================
print("\n" + "=" * 60)
print("СВОДКА")
print("=" * 60)

# Группировка по моделям
from collections import defaultdict
model_summary = defaultdict(lambda: {
    "count": 0, "total_time": 0, "total_tokens": 0,
    "total_chars": 0, "errors": 0
})

for r in all_results:
    key = (r["model"], r["family"], r["size"])
    model_summary[key]["count"] += 1
    model_summary[key]["total_time"] += r["inference_time_s"]
    model_summary[key]["total_tokens"] += r["tokens_generated"]
    model_summary[key]["total_chars"] += r["response_length_chars"]
    if r["tokens_generated"] == 0:
        model_summary[key]["errors"] += 1

summary_lines = [
    "# LLM Benchmark Summary\n",
    "| Модель | Семейство | Размер | Среднее время | Tok/с | Средняя длина | Ошибки |\n",
    "|--------|-----------|--------|---------------|-------|---------------|--------|\n",
]

for key in sorted(model_summary.keys(), key=lambda k: (int(k[2].replace("B","").split(".")[0]), k[0])):
    _, family, size = key
    s = model_summary[key]
    avg_time = s["total_time"] / s["count"]
    avg_tps = s["total_tokens"] / s["total_time"] if s["total_time"] > 0 else 0
    avg_len = s["total_chars"] / s["count"]
    errors = s["errors"]
    line = f"| {key[0]} | {family} | {size} | {avg_time:.1f}с | {avg_tps:.0f} | {avg_len:.0f} | {errors} |\n"
    summary_lines.append(line)
    print(f"  {key[0]:20s} | {avg_time:.1f}с | {avg_tps:.0f} tok/с | {avg_len:.0f} chars | errors={errors}")

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.writelines(summary_lines)

print(f"\nСводка сохранена: {SUMMARY_PATH}")
print("\n✅ БЕНЧМАРК ЗАВЕРШЁН")
