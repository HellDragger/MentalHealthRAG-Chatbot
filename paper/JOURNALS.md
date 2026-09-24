# Candidate Q1/Q2 journals

> **How the quartiles were checked (2026-09-24).** scimagojr.com blocks automated access with a bot check, which we did
> not try to bypass. The SJR values and quartiles below come from web-search results that point at each journal's
> Scimago page (linked) and at metric aggregators. **Before submitting, open each Scimago link yourself** and confirm the
> latest year's quartile *in the subject category that matters to you*. A journal can be Q1 in one category and Q2 in
> another. JCR (Clarivate) quartiles and impact factors need an institutional Web of Science login, and we could not
> check them. APC amounts change often, so none are listed here. Use the publisher page linked for each journal.

| Journal | Scimago page | Best quartile (as reported) | SJR (as reported) | Access model |
|---|---|---|---|---|
| JMIR Mental Health | [link](https://www.scimagojr.com/journalsearch.php?q=21101030407&tip=sid) | Q1 (Psychiatry & Mental Health) | 2.091 | Fully open access, APC |
| JMIR Formative Research | [link](https://www.scimagojr.com/journalsearch.php?q=21101028582&tip=sid) | Q2 | 0.723 | Fully open access, APC |
| Journal of Biomedical Informatics (Elsevier) | [link](https://www.scimagojr.com/journalsearch.php?q=23706&tip=sid) | Q1 | 1.32 | Hybrid (subscription or OA with APC) |
| International Journal of Medical Informatics (Elsevier) | [link](https://www.scimagojr.com/journalsearch.php?q=23689&tip=sid) | Q1 | 1.189 | Hybrid |
| Artificial Intelligence in Medicine (Elsevier) | [link](https://www.scimagojr.com/journalsearch.php?q=24140&tip=sid) | Q1 | 1.396 | Hybrid |
| IEEE Journal of Biomedical and Health Informatics | [link](https://www.scimagojr.com/journalsearch.php?q=21100256982&tip=sid) | Q1 | 1.624 | Hybrid |
| Computers in Biology and Medicine (Elsevier) | [link](https://www.scimagojr.com/journalsearch.php?q=17957&tip=sid) | Q1 | not captured | Hybrid |
| Expert Systems with Applications (Elsevier) | [link](https://www.scimagojr.com/journalsearch.php?q=24201&tip=sid) | Q1 | 1.854 | Hybrid |
| Frontiers in Digital Health | [link](https://www.scimagojr.com/journalsearch.php?q=21101090720&tip=sid) | Q1 (sources disagree on the SJR value) | 1.070–1.247 | Fully open access, APC |
| Frontiers in Psychiatry | [link](https://www.scimagojr.com/journalsearch.php?q=21100216569&tip=sid) | Q1 or Q2 (sources disagree) | ~1.19 | Fully open access, APC |
| IEEE Access | [link](https://www.scimagojr.com/journalsearch.php?q=21100374601&tip=sid) | Q1 (as reported for 2025) | not captured | Fully open access, APC |

The manuscript uses the Elsevier `elsarticle` template, so the Elsevier journals need the fewest formatting changes.
JMIR and Frontiers have their own templates. IEEE journals use `IEEEtran`.

## Fit, and what each journal will expect

**Common to all clinical and health-informatics venues.** Every venue in this area will ask for three things the project
doesn't have yet:
1. **Human evaluation by qualified raters.** Clinicians (psychiatrists, clinical psychologists or mental-health nurses)
   rating responses blind, with agreement statistics. The protocol and scripts are in `eval/human_eval/`.
2. **An ethics statement.** Ethics approval or a documented exemption for the rater study, plus a clear statement
   that the system is not a medical device and was not tested with patients.
3. **Reporting against a guideline.** For example, DECIDE-AI or the CONSORT-AI/SPIRIT-AI extensions for
   AI evaluations. JMIR also expects iCHECK-DH for digital-health interventions. At this stage (no users), a
   "development and validation" framing is appropriate, and outcome claims about users are not.

| Journal | Scope fit | Extra work it is likely to expect |
|---|---|---|
| **JMIR Mental Health** | Excellent: digital mental-health tools, chatbots, LLM safety. | Clinician evaluation is close to mandatory for a Q1 slot. Ethics statement. Discussion of clinical implications. A user study (even a small usability study with a SUS questionnaire) would strengthen it a lot. |
| **JMIR Formative Research** | Very good: explicitly for early-stage and formative evaluations. | The most realistic first target: a formative evaluation (automatic metrics plus a small expert rating) fits its remit. Still needs ethics and a human evaluation component. |
| **Journal of Biomedical Informatics** | Good: methodology (retrieval, evaluation framework, safety gating). | Stronger methodological novelty and broader baselines (for example RAGAS/ARES comparisons, more embedders or rerankers, a published medical QA benchmark). Statistical rigour (already provided). A GPU-scale LLM benchmark (run the notebook). |
| **International Journal of Medical Informatics** | Good: applied health-informatics systems. | Evaluation with end users or clinicians. Deployment and usability evidence. Discussion of integration into care pathways. |
| **Artificial Intelligence in Medicine** | Good: AI methods for medicine. | Methodological contribution beyond system integration (for example a better-generalising safety gate and comparisons with published crisis-detection models). Complete multi-LLM results. |
| **IEEE JBHI** | Moderate to good: health informatics with a technical emphasis. | Stronger technical novelty and extensive quantitative comparisons. Human evaluation helps. The IEEE template. |
| **Computers in Biology and Medicine** | Moderate: broad computational biomedicine. | Complete experiments (all GPU runs), clinical relevance and a clear comparison with prior mental-health chatbots. |
| **Expert Systems with Applications** | Moderate: applied AI systems. | Engineering depth and ablations (present), plus a larger benchmark. Mental-health-specific validation would still be requested by reviewers. |
| **Frontiers in Digital Health** | Very good: digital health, conversational agents, ethics. | Human evaluation, an ethics statement and a data-availability statement (already drafted). |
| **Frontiers in Psychiatry** | Moderate: the audience is clinical. | A clinician evaluation is essential. The framing must centre clinical and psychoeducational value rather than engineering. |
| **IEEE Access** | Broad: accepts applied systems papers. | The fastest route, but the lowest prestige of the list. Complete results and a clear novelty statement are still needed. |

## Recommendation
1. Finish the GPU notebook runs (full multi-LLM table) and the **clinician rating study**, which needs about
   3 raters × 60 items × 3–5 systems (see `eval/human_eval/rubric.md`).
2. Target **JMIR Mental Health** first. If the human study stays small, go to **JMIR Formative Research** or
   **Frontiers in Digital Health**.
3. For a methods-focused paper instead (retrieval, safety-gate generalisation, benchmark), target the
   **Journal of Biomedical Informatics** or **Artificial Intelligence in Medicine**, and add more baselines.
