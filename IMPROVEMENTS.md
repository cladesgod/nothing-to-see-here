# Gelistirme Raporu

Proje boyunca karsilasilan sorunlar, uygulanan cozumler ve elde edilen sonuclar.

---

## Structured Output (Tarihsel — artik kullanilmiyor)

> **Not:** 2026-02-09 tarihinde tum structured output (JSON parsing) kaldirildi. Agentlar artik dogal dil (plain text) ile iletisim kuruyor — paper'in AutoGen chat yaklasimina uygun. Asagidaki kayitlar tarihsel referans icin saklanmistir.

### 1. Token Runaway

- **Sorun:** Llama 3.3 70B, `function_calling` modunda JSON uretmek yerine 16K+ token serbest metin yaziyordu. Basarisizlik orani ~%70-80.
- **Cozum:** `max_tokens=4096` siniri + `method="json_schema"` ile `strict=True`'ya gecis.
- **Sonuc:** ~%70-80 → ~%20-30.

### 2. Native Structured Output Destegi

- **Sorun:** Llama 3.3 70B, `json_schema` modunu native desteklemiyor. Basarisizlik orani ~%20-30.
- **Cozum:** Tum reviewer'lar `llama-4-maverick`'e gecildi (native `structured_outputs` destegi).
- **Sonuc:** ~%20-30 → ~%0 (model tarafli hata). Ancak platform hatasi ortaya cikti.

### 3. OpenRouter Eksik Brace Hatasi

- **Sorun:** OpenRouter'in `json_schema` modu bazen JSON'in acilis `{` parantezini atliyor.
- **Cozum:** `invoke_with_repair()` helper'i ile otomatik onarim.
- **Sonuc:** ~%30-50 → ~%3.

### 4-5. Diger Onarimlar

- Eksik `{"` repair, custom kod minimizasyonu.
- **Nihai karar:** Tum bu sorunlar dogal dil iletisimine gecilerek tamamen ortadan kaldirildi.

---

## Tasarim Kararlari

Paper'dan farkli olarak bilincli tercih yaptigimiz noktalar.

### 1. Paralel Reviewer'lar

- **Paper:** Sirasal (AutoGen nested chat: content → linguistic → bias → meta_editor).
- **Biz:** Paralel fan-out/fan-in (LangGraph StateGraph). Uc reviewer ayni anda calisir, sonuclar meta_editor'de birlesir.
- **Avantaj:** ~3x daha hizli. Reviewer'lar birbirinden bagimsiz oldugu icin sonuc ayni.

### 2. Deterministik Critic

- **Paper:** LLM-tabanli Critic Agent (GPT-4 ile routing karari).
- **Biz:** Python fonksiyonu (`critic_router`). `current_phase` state degiskenine gore deterministik routing.
- **Avantaj:** Daha hizli (0ms vs ~2s), ucuz ($0), %100 guvenilir, kolay test edilebilir.

### 3. Dogal Dil Iletisimi

- **Paper:** AutoGen GroupChat — agentlar dogal dil mesajlari ile iletisim kurar.
- **Biz:** Ayni yaklasim. Tum agentlar plain text uretir. JSON structured output yok.
- **Avantaj:** JSON parse hatalari sifir. Model ciktisi dogrudan okunabilir ve debug edilebilir.

### 4. Batch Review (4 LLM cagrisi)

- **Paper:** Item bazli review (per-item chat).
- **Biz:** Tum item'lar toplu olarak review edilir: 3 reviewer + 1 meta editor = 4 LLM cagrisi.
- **Avantaj:** 8 item × 4 agent = 32 cagri yerine sadece 4 cagri. Maliyet ve sure ~8x azalir.

---

## Paper Uyumluluk Duzeltmeleri

### 5. Yanlis Construct Tanimi Duzeltildi

- **Sorun:** AAAW = "Adjustment to Academic Work" olarak tanimlanmisti. Gercekte AAAW = "Attitudes Toward the Use of AI in the Workplace" (Park et al., 2024).
- **Cozum:** `constructs.py` tamamen yeniden yazildi. 6 dogru boyut.
- **Dosya:** `src/schemas/constructs.py`

### 6. Content Validity Formulu (Colquitt et al., 2019) Duzeltildi

- **Sorun:** c-value ve d-value hesaplamalari yanlis. Colquitt yonteminde 3 rating (target + 2 orbiting) gerekli.
- **Sonraki:** c-value = target_rating/6 (>= 0.83), d-value = mean(target-orbiting)/6 (>= 0.35)
- **Dosya:** `src/prompts/templates.py`, `src/agents/content_reviewer.py`

### 7. Prompt Alignment (Paper Table 2)

- **Item Writer:** 12 guideline, reverse-coded item'lar yasaklandi.
- **Linguistic Reviewer:** 4 kriter / 5-point scale (grammar, understanding, negative language, clarity).
- **Bias Reviewer:** 5-point scale (gender, religion, race, age, culture).
- **Meta Editor:** 4 sorumluluk (synthesize, edit/discard, integrate feedback, identify issues).
- **Dosya:** `src/prompts/templates.py`

---

## Eklenen Ozellikler

### 8. LewMod — Otomatik Uzman Feedback

- **Sorun:** Human-in-the-loop her seferinde manuel onay gerektiriyor.
- **Cozum:** `--lewmod` flagi ile otomatik uzman geri bildirimi. Dr. LewMod (senior psikometrist LLM persona) item'lari degerlendirir ve APPROVE/REVISE karar verir.
- **Dosya:** `src/agents/lewmod.py`

### 9. Merkezi Agent Konfigurasyonu (agents.toml)

- **Sorun:** Model, temperature ve diger parametreler agent dosyalarinda hardcoded.
- **Cozum:** `agents.toml` — tek bir dosyadan tum agent parametreleri yonetilir. `tomllib` (stdlib) ile parse edilir.
- **Dosya:** `agents.toml`, `src/config.py`

### 10. Rich Console Ciktisi

- **Sorun:** Agent iletisimi duz metin olarak gorunuyordu.
- **Cozum:** `rich` kutuphanesi ile renkli, formatli agent mesajlari. Paper stilinde `AgentName (to Target):` formati.
- **Dosya:** `src/utils/console.py`

### 11. Retry Mekanizmasi + Fallback Provider Zinciri

- **Sorun:** Tum `llm.ainvoke()` cagrilari korumasiz — OpenRouter rate limit, network hatasi veya gecici ariza durumunda workflow coker.
- **Cozum:** Iki katmanli guvenilirlik:
  - **Katman 1 — LangGraph RetryPolicy:** Node seviyesinde otomatik retry. Exponential backoff ile 3 deneme. `agents.toml` `[retry]` bolumunden konfigure edilir.
  - **Katman 2 — Fallback Provider Zinciri:** `create_llm()` icinde `with_fallbacks()` ile OpenRouter → Groq → Ollama zinciri. Her provider'in modeli agent bazinda `agents.toml`'dan konfigure edilir. Per-agent override: `groq_model` / `ollama_model`.
  - Iki katman birlikte calisir: her retry denemesinde tam fallback zinciri tekrar calisir.
- **Sonuc:** 61 test (onceki 40). Sifir downtime riski — 3 bagimsiz provider zinciri ile yuksek erisilebilirlik.
- **Dosyalar:** `pyproject.toml`, `agents.toml`, `src/config.py`, `src/models.py`, `src/graphs/main_workflow.py`, `src/graphs/review_chain.py`, `src/utils/console.py`, `tests/test_config.py`, `tests/test_models.py` (yeni), `tests/test_workflow.py`

### 12. SQLite Persistence + Anti-Homogeneity Guard

- **Sorun:** Pipeline her calistirmada sifirdan basliyor — onceki run'lardaki maddeler, review'lar, feedback kaybolur. Ayrica Lee et al. (2025) AI-uretilen maddelerin birbirine cok benzer oldugunu buldu (3/24 item dusuk content validity).
- **Cozum:**
  - SQLite persistence layer (`src/persistence/`). 6 tablo: `runs`, `research`, `generation_rounds`, `reviews`, `feedback`, `eval_results`. Zero dependency (`sqlite3` stdlib).
  - `get_previous_items()` ile Item Writer, onceki run'larin final maddelerini DB'den okuyarak prompt'a enjekte eder.
  - Meta Editor'e Rule 6 (inter-item similarity check) eklendi.
  - `agents.toml`'dan kontrol edilebilir: `memory_enabled = true/false`, `memory_limit = 5` (kac onceki run dahil edilsin).
- **Sonuc:** 133 test (onceki 103). Cross-run ve cross-round item diversity sistematik olarak zorlanir. Tum pipeline state'i kalici olarak saklanir.
- **Dosyalar:** `src/persistence/db.py`, `src/persistence/repository.py`, `src/agents/item_writer.py`, `src/agents/web_surfer.py`, `src/agents/lewmod.py`, `src/graphs/main_workflow.py`, `src/prompts/templates.py`, `src/schemas/state.py`, `src/config.py`, `agents.toml`, `run.py`, `tests/test_persistence.py` (yeni), `tests/test_config.py`

