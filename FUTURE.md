# Future Improvements

Walkthrough sunumunda "daha fazla zamanim olsa ne yapardim" bolumu icin.

---

## 1. Selective Revision (Token Optimizasyonu)

**Problem:** Revision dongusunde KEEP ve REVISE itemlar birlikte tekrar isleniyor. MetaEditor 8 itemdan 6'sini KEEP dese bile, tum 8 item yeniden item_writer + 3 reviewer + meta_editor'den geciyor (5 LLM call, hepsi gereksiz).

**Cozum:** KEEP itemlari state'de ayir, sadece REVISE itemlari donguye sok. Dongü bitince birlestir.

**Etki:** ~%50-75 token tasarrufu, daha hizli revision turlari, KEEP itemlarin bozulma riski sifir.

---

## 2. Multi-Construct Destegi

**Problem:** Sistem su an AAAW construct'una ozgu. `constructs.py`'de 6 boyut ve orbiting dimension'lar hardcoded.

**Cozum:** Construct tanimini JSON/TOML dosyasindan oku. Kullanici kendi construct'unu tanimlayabilsin (boyutlar, orbiting'ler, item sayisi). CLI: `python run.py --construct my_construct.toml`

**Etki:** Herhangi bir psikolojik olcek icin kullanilabilir hale gelir.

---

## 3. ~~PostgreSQL Persistence~~ → IMPLEMENTED as SQLite Persistence

**Durum:** IMPLEMENTED (2026-02-10)

SQLite persistence layer eklendi (`src/persistence/`). 6 tablo: `runs`, `research`, `generation_rounds`, `reviews`, `feedback`, `eval_results`. Her pipeline adimi DB'ye kaydediliyor. `get_previous_items()` ile cross-run item diversity saglaniyor.

Production icin PostgreSQL'e gecis hala bir secenek ama mevcut SQLite cozumu tek kullanici senaryolari icin yeterli.

---

## 4. LLM-based Critic Agent

**Problem:** Critic su an deterministik — sabit faz sirasini takip ediyor (research → generate → review → feedback → revise). Nuansli kararlar veremez.

**Cozum:** Critic'i LLM-tabanli yap. Ornegin: "Bu item'lar zaten yeterince iyi, review'a gerek yok" veya "Web research'e geri don, daha fazla bilgi lazim" gibi kararlar alabilsin.

**Trade-off:** Daha yavas ($, latency), daha az tahmin edilebilir. Deterministic critic daha guvenilir ve test edilebilir.

---

## 5. Security Hardening

**Problem:** Prompt injection korumalari, input validation, rate limiting henuz yok.

**Yapilacaklar:**
- Input sanitization (kullanici feedback'inde prompt injection kontrolu)
- LLM output validation (unexpected format kontrolu)
- Rate limiting (API cagri limitleri)
- Secrets management (env var'lar disinda vault entegrasyonu)
- Audit logging (kim ne zaman ne yapti)

---

## 6. Online Eval & Monitoring

**Problem:** Eval pipeline offline calisiyor. Production'da surekli izleme yok.

**Cozum:** LangSmith online evaluator'lar ile her run otomatik degerlendirilsin. Kalite dustu mu? Alert gonder. Dashboard'da trend takibi.

**Etki:** Production kalite guvencesi, regression tespiti.

---

## 7. Factor Analysis Integration

**Problem:** Pipeline item uretip review ediyor ama empirik dogrulama yok. Gercek pilot test sonrasi faktor analizi gerekiyor.

**Cozum:** R veya Python (factor_analyzer) entegrasyonu. Pilot test verisini al → CFA (Confirmatory Factor Analysis) calistir → sonuclari pipeline'a geri besle.

**Etki:** End-to-end dogrulama: LLM-generated items → pilot test → faktor analizi → item refinement.

---

## 8. ~~Anti-Homogeneity / Item Diversity Guard~~ → IMPLEMENTED

**Durum:** IMPLEMENTED (2026-02-10)

**Problem:** Lee et al. (2025) AI-uretilen maddelerin birbirine cok benzer oldugunu buldu. 24 maddeden 3'u construct blurring nedeniyle dusuk content validity aldi. Makalenin tek cozumu human feedback'te "not too similar" demekti — sistematik mekanizma yoktu.

**Cozum:**
- SQLite persistence layer ile onceki run'larin final maddeleri saklanir
- `get_previous_items()` ile Item Writer, onceki maddeleri DB'den okur
- Prompt'lara anti-homogeneity talimatlari eklendi (GENERATE + REVISE)
- Meta Editor'e inter-item similarity check (Rule 6) eklendi
- `previously_approved_items` state field'i ile ayni run icindeki turlar arasi da izleniyor

**Etki:** Cross-run ve cross-round item diversity sistematik olarak zorlanir.
