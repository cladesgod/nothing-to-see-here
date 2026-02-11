# Future Improvements

Walkthrough sunumunda "daha fazla zamanim olsa ne yapardim" bolumu icin.

---

## 1. LLM-based Critic Agent

**Problem:** Critic su an deterministik — sabit faz sirasini takip ediyor (research → generate → review → feedback → revise). Nuansli kararlar veremez.

**Cozum:** Critic'i LLM-tabanli yap. Ornegin: "Bu item'lar zaten yeterince iyi, review'a gerek yok" veya "Web research'e geri don, daha fazla bilgi lazim" gibi kararlar alabilsin.

**Trade-off:** Daha yavas ($, latency), daha az tahmin edilebilir. Deterministic critic daha guvenilir ve test edilebilir.

---

## 2. API Human Feedback (Phase 2)

**Durum:** PLANNED — Endpoint mevcut ama `501 Not Implemented` donuyor.

**Problem:** API uzerinden human feedback mode kullanildiginda (`lewmod: false`), graph `interrupt()` ile duruyor ama API bunu resume edemiyoruz. Su an sadece LewMod (otomatik) mode API'den calisiyor.

**Yapilacaklar:**
1. `WorkerPool._execute()` — `GraphInterrupt` yakalama + `asyncio.Event` ile feedback bekleme
2. `WorkerPool.resume(run_id, feedback_data)` — Event'i tetikleyip graph'i `Command(resume=...)` ile devam ettirme
3. `app.py` feedback endpoint — `POST /api/v1/runs/{id}/feedback` aktif etme
4. Run status'te `waiting_feedback` state'i duzgun yonetme

**Mimari:**
- `RunInfo`'ya `_graph`, `_graph_config`, `_feedback_event`, `_feedback_data` field'lari eklenmeli
- `_execute()` icinde interrupt sonrasi `Event.wait()` ile blok → feedback gelince `Command(resume=data)` ile devam
- PostgreSQL checkpointer kullanilirsa graph state disk'te kalir, server restart sonrasi bile resume mumkun

**Etki:** API uzerinden tam human-in-the-loop destegi — CLI'daki interaktif feedback deneyiminin API karsiligi.

---

## 3. Evaluation Pipeline (Yeniden Yapilacak)

**Problem:** Eval pipeline kaldirildi (2026-02-10), bastan yapilacak.

**Yapilacaklar:** Yeni eval pipeline tasarimi ve implementasyonu. LLM-as-a-Judge yaklasimi ile pipeline ciktisinin bagimsiz kalite olcumu.

---

## 4. Factor Analysis Integration

**Problem:** Pipeline item uretip review ediyor ama empirik dogrulama yok. Gercek pilot test sonrasi faktor analizi gerekiyor.

**Cozum:** R veya Python (factor_analyzer) entegrasyonu. Pilot test verisini al → CFA (Confirmatory Factor Analysis) calistir → sonuclari pipeline'a geri besle.

**Etki:** End-to-end dogrulama: LLM-generated items → pilot test → faktor analizi → item refinement.
