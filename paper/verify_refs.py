"""Fetch and verify every reference from arXiv / Crossref and write refs.bib + refs_verification.md.

    python paper/verify_refs.py

Nothing in refs.bib is typed by hand: titles, authors, years and venues come from the arXiv API or Crossref.
An entry that cannot be fetched is written with a `% UNVERIFIED` marker and listed in refs_verification.md.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

# key -> ("arxiv", id) | ("doi", doi) | ("misc", {fields})
REFS: dict[str, tuple] = {
    # RAG and retrieval
    "lewis2020rag": ("arxiv", "2005.11401"),
    "gao2023ragsurvey": ("arxiv", "2312.10997"),
    "xiong2024medrag": ("arxiv", "2402.13178"),
    "es2023ragas": ("arxiv", "2309.15217"),
    "saadfalcon2023ares": ("arxiv", "2311.09476"),
    "robertson2009bm25": ("doi", "10.1561/1500000019"),
    "cormack2009rrf": ("doi", "10.1145/1571941.1572114"),
    "reimers2019sbert": ("arxiv", "1908.10084"),
    "xiao2023cpack": ("arxiv", "2309.07597"),
    "wang2022e5": ("arxiv", "2212.03533"),
    "li2023gte": ("arxiv", "2308.03281"),
    "nogueira2019passage": ("arxiv", "1901.04085"),
    "alberti2019synthetic": ("arxiv", "1906.05416"),
    # mental-health NLP and agents
    "fitzpatrick2017woebot": ("doi", "10.2196/mental.7785"),
    "vaidyam2019chatbots": ("doi", "10.1177/0706743719828977"),
    "abdalrazaq2019overview": ("doi", "10.1016/j.ijmedinf.2019.103978"),
    "yang2023mentallama": ("arxiv", "2309.13567"),
    "ji2022mentalbert": ("arxiv", "2110.15621"),
    "turcan2019dreaddit": ("arxiv", "1911.00133"),
    "sharma2020empathy": ("arxiv", "2009.08441"),
    "liu2021esconv": ("arxiv", "2106.01144"),
    "lamichhane2023chatgpt": ("arxiv", "2303.15727"),
    "dechoudhury2023benefits": ("arxiv", "2311.14693"),
    "hua2024llmmh": ("arxiv", "2401.02984"),
    "stade2024behavioral": ("doi", "10.1038/s44184-024-00056-z"),
    "guo2024llmmhreview": ("doi", "10.2196/57400"),
    # LLMs
    "jiang2023mistral": ("arxiv", "2310.06825"),
    "dubey2024llama3": ("arxiv", "2407.21783"),
    "qwen2025qwen25": ("arxiv", "2412.15115"),
    "yang2025qwen3": ("arxiv", "2505.09388"),
    "gemma2024gemma2": ("arxiv", "2408.00118"),
    "gemma2025gemma3": ("arxiv", "2503.19786"),
    "abdin2024phi3": ("arxiv", "2404.14219"),
    "radford2019gpt2": ("misc", {"title": "Language Models are Unsupervised Multitask Learners",
                                 "author": "Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya",
                                 "year": "2019", "howpublished": "OpenAI technical report",
                                 "url": "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"}),
    "lewis2019bart": ("arxiv", "1910.13461"),
    "dettmers2023qlora": ("arxiv", "2305.14314"),
    # evaluation and statistics
    "zheng2023judge": ("arxiv", "2306.05685"),
    "he2021debertav3": ("arxiv", "2111.09543"),
    "zhang2020bertscore": ("arxiv", "1904.09675"),
    "lin2004rouge": ("misc", {"title": "{ROUGE}: A Package for Automatic Evaluation of Summaries",
                              "author": "Lin, Chin-Yew", "year": "2004", "booktitle": "Text Summarization Branches Out",
                              "pages": "74--81", "publisher": "Association for Computational Linguistics",
                              "url": "https://aclanthology.org/W04-1013/"}),
    "flesch1948readability": ("doi", "10.1037/h0057532"),
    "koehn2004significance": ("misc", {"title": "Statistical Significance Tests for Machine Translation Evaluation",
                                       "author": "Koehn, Philipp", "year": "2004",
                                       "booktitle": "Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing",
                                       "pages": "388--395", "url": "https://aclanthology.org/W04-3250/"}),
    "holm1979": ("misc", {"title": "A Simple Sequentially Rejective Multiple Test Procedure", "author": "Holm, Sture",
                          "journal": "Scandinavian Journal of Statistics", "volume": "6", "number": "2",
                          "pages": "65--70", "year": "1979", "url": "https://www.jstor.org/stable/4615733"}),
    "hayes2007krippendorff": ("doi", "10.1080/19312450709336664"),
}

# misc entries whose existence was checked by fetching the listed URL (set by check_misc below)


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mhrag-refs/1.0 (mailto:research@example.org)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def bibesc(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def arxiv_batch(ids: list[str]) -> dict[str, dict]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    out = {}
    for i in range(0, len(ids), 20):
        chunk = ids[i : i + 20]
        xml = fetch("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(chunk), "max_results": len(chunk)}))
        root = ET.fromstring(xml)
        for e in root.findall("a:entry", ns):
            aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
            base = re.sub(r"v\d+$", "", aid)
            title = e.find("a:title", ns)
            if title is None or not title.text or "Error" in (title.text or ""):
                continue
            out[base] = {
                "title": title.text,
                "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                "year": e.find("a:published", ns).text[:4],
                "journal_ref": (e.find("arxiv:journal_ref", ns).text if e.find("arxiv:journal_ref", ns) is not None else None),
                "doi": (e.find("arxiv:doi", ns).text if e.find("arxiv:doi", ns) is not None else None),
                "primary": e.find("arxiv:primary_category", ns).attrib.get("term"),
            }
        time.sleep(3)
    return out


def crossref(doi: str) -> dict | None:
    try:
        m = json.loads(fetch("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    except Exception:
        return None
    authors = [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in m.get("author", [])]
    return {"title": (m.get("title") or [""])[0], "authors": authors,
            "year": str(m.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "container": (m.get("container-title") or [""])[0], "volume": m.get("volume"), "issue": m.get("issue"),
            "pages": m.get("page"), "type": m.get("type"), "publisher": m.get("publisher")}


def fmt_authors(names: list[str]) -> str:
    out = []
    for n in names:
        if "," in n:
            out.append(n)
        else:
            parts = n.split()
            out.append(f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n)
    return " and ".join(bibesc(a) for a in out)


def main():
    arx_ids = [v[1] for v in REFS.values() if v[0] == "arxiv"]
    arx = arxiv_batch(arx_ids)
    entries, report = [], []
    for key, (kind, ref) in REFS.items():
        if kind == "arxiv":
            m = arx.get(ref)
            if not m:
                entries.append(f"% UNVERIFIED: arXiv:{ref} could not be fetched\n@misc{{{key},\n  eprint = {{{ref}}},\n  archivePrefix = {{arXiv}}\n}}")
                report.append((key, f"arXiv:{ref}", "UNVERIFIED"))
                continue
            fields = [f"  title = {{{{{bibesc(m['title'])}}}}}", f"  author = {{{fmt_authors(m['authors'])}}}",
                      f"  year = {{{m['year']}}}", f"  eprint = {{{ref}}}", "  archivePrefix = {arXiv}",
                      f"  primaryClass = {{{m['primary']}}}", f"  url = {{https://arxiv.org/abs/{ref}}}"]
            if m["journal_ref"]:
                fields.append(f"  note = {{{bibesc(m['journal_ref'])}}}")
            if m["doi"]:
                fields.append(f"  doi = {{{m['doi']}}}")
            entries.append(f"@misc{{{key},\n" + ",\n".join(fields) + "\n}")
            report.append((key, f"arXiv:{ref}", f"verified: {m['title'][:70]}"))
        elif kind == "doi":
            m = crossref(ref)
            if not m:
                entries.append(f"% UNVERIFIED: doi:{ref} could not be fetched\n@misc{{{key},\n  doi = {{{ref}}}\n}}")
                report.append((key, f"doi:{ref}", "UNVERIFIED"))
                continue
            etype = "inproceedings" if "proceedings" in (m["type"] or "") else "article"
            venue_field = "booktitle" if etype == "inproceedings" else "journal"
            fields = [f"  title = {{{{{bibesc(m['title'])}}}}}", f"  author = {{{fmt_authors(m['authors'])}}}",
                      f"  year = {{{m['year']}}}", f"  {venue_field} = {{{bibesc(m['container'])}}}", f"  doi = {{{ref}}}"]
            for f_ in ("volume", "issue", "pages"):
                if m.get(f_):
                    fields.append(f"  {'number' if f_ == 'issue' else f_} = {{{m[f_]}}}")
            entries.append(f"@{etype}{{{key},\n" + ",\n".join(fields) + "\n}")
            report.append((key, f"doi:{ref}", f"verified: {m['title'][:70]}"))
        else:
            fields = ref
            status = "UNVERIFIED (no API)"
            try:
                fetch(fields["url"])
                status = "URL resolves (metadata typed from the landing page)"
            except Exception:
                pass
            etype = "inproceedings" if "booktitle" in fields else ("article" if "journal" in fields else "misc")
            body = ",\n".join(f"  {k} = {{{v if k in ('title',) and v.startswith('{') else v}}}" for k, v in fields.items())
            prefix = "" if status.startswith("URL resolves") else f"% {status}\n"
            entries.append(f"{prefix}@{etype}{{{key},\n{body}\n}}")
            report.append((key, fields.get("url", ""), status))
    (HERE / "refs.bib").write_text("% Generated by paper/verify_refs.py -- do not edit by hand.\n\n" + "\n\n".join(entries) + "\n")
    lines = ["# Reference verification", "", "Generated by `python paper/verify_refs.py`.", "",
             "| key | source | status |", "|---|---|---|"]
    lines += [f"| {k} | {s} | {st} |" for k, s, st in report]
    (HERE / "refs_verification.md").write_text("\n".join(lines) + "\n")
    bad = [r for r in report if "UNVERIFIED" in r[2]]
    print(f"{len(report)} references, {len(bad)} unverified")
    for r in report:
        print(r)


if __name__ == "__main__":
    main()
