"""Assemble the MovieMind report from the college sample.
Usage: python build.py            (pass 1 + page-number pass 2, writes out/MovieMind_PDL_Report.docx)
"""
import json, os, re, shutil, subprocess, sys, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import docx_lib as L
import content as C
HM = C.HM

SAMPLE = os.path.join(HERE, "template", "sample.docx")
WORK = os.path.join(HERE, "work")
OUT = os.path.join(HERE, "out")
FINAL = os.path.join(OUT, "MovieMind_PDL_Report.docx")
os.makedirs(OUT, exist_ok=True)

NAME, REG = "Vikranath A", "111724203119"
TITLE_FULL = "MovieMind: AI-Powered Movie Recommendation System"
SUP = "Ms. S. Nandhini"


def split_body(d):
    b0 = d.index("<w:body>") + 8
    b1 = d.rindex("</w:body>")
    body = d[b0:b1]
    kids, depth, start = [], 0, 0
    for m in re.finditer(r"<(/?)(w:p|w:tbl|w:sectPr)(?=[\s>/])([^>]*?)(/?)>", body):
        close, tag, attrs, selfc = m.groups()
        if selfc:
            if depth == 0:
                kids.append(body[m.start():m.end()])
            continue
        if not close:
            if depth == 0:
                start = m.start()
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                kids.append(body[start:m.end()])
    return d[:b0], kids, d[b1:]


def ptext(x):
    return "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", x))


def set_single_text(x, new):
    """Replace the text of a paragraph that has one visible run."""
    return re.sub(r"<w:t(?: [^>]*)?>[^<]*</w:t>", f"<w:t>{L.esc(new)}</w:t>", x, count=1)


def ppr(x):
    m = re.search(r"<w:pPr>.*?</w:pPr>", x, flags=re.S)
    return m.group(0) if m else ""


def rebuild_para(x, parts):
    """Keep paragraph properties, replace runs. parts: list of (text, bold, tab_count_before)."""
    out = ""
    for text, bold, tabs in parts:
        rp = "<w:rPr><w:b/><w:bCs/></w:rPr>" if bold else ""
        out += "".join(f"<w:r>{rp}<w:tab/></w:r>" for _ in range(tabs))
        if text:
            out += f'<w:r>{rp}<w:t xml:space="preserve">{L.esc(text)}</w:t></w:r>'
    return "<w:p>" + ppr(x) + out + "</w:p>"


def para_like(x, runs_xml):
    return "<w:p>" + ppr(x) + runs_xml + "</w:p>"


def build(pages):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    with zipfile.ZipFile(SAMPLE) as z:
        names = z.namelist()
        z.extractall(WORK)
    docp = os.path.join(WORK, "word", "document.xml")
    d = open(docp, encoding="utf8").read()
    head, k, tail = split_body(d)
    sect = k[-1]
    assert sect.startswith("<w:sectPr")

    # ---------------- cover
    k[17] = k[18] = k[19] = ''
    k[20] = '<w:p><w:pPr><w:spacing w:before="1400" w:after="200"/><w:jc w:val="center"/></w:pPr>' + L._run('NOVEMBER 2026', bold=True, sz=26) + '</w:p>'
    k[6] = set_single_text(k[6], "MOVIEMIND: AI-POWERED MOVIE")
    k[7] = set_single_text(k[7], "RECOMMENDATION SYSTEM")
    k[9] = set_single_text(k[9], f"{NAME} – {REG}")
    # ---------------- bonafide
    k[23] = para_like(k[23],
        L._run('Certified that this Product Development Lab III report "MovieMind: AI-Powered Movie Recommendation System" is the bonafide work of ')
        + L._run(f"“ {NAME} ({REG}) ”", bold=True)
        + L._run(" who carried out the 24ME511 – Product Development Lab III under my supervision."))
    k[29] = rebuild_para(k[29], [("Dr. Sheerin Banu M, M.Tech, Ph.D", False, 0), (f"{SUP},", False, 3)])
    k[30] = rebuild_para(k[30], [("Professor and Head,", False, 0), ("Supervisor, Assistant Professor,", False, 5)])
    # ---------------- evaluation table
    k[40] = L.plain_table(["Title of Project", "Name of the Students", "Name of the Supervisor with designation"],
                          [[TITLE_FULL, f"{NAME} ({REG})", f"{SUP}, Assistant Professor"]], [3000, 3000, 3000], sz=22)
    # ---------------- acknowledgement (coordinator unknown, supervisor Ms. S. Nandhini)
    k[53] = para_like(k[53],
        L._run("We would like to express our sincere gratitude to our Course Coordinator ")
        + L._run("Dr. [Course Coordinator name]", bold=True, hl=True)
        + L._run(", for her/his valuable suggestions towards the successful completion of this project."))
    k[54] = para_like(k[54],
        L._run("We wish to record our thanks to our project supervisor ")
        + L._run(f"{SUP}", bold=True)
        + L._run(", for her valuable guidance and support during each stage of our project."))

    # ---------------- abstract
    abstract = [
        "Choosing what to watch is a problem of abundance. Streaming catalogues hold tens of thousands of titles, yet most recommender systems describe a viewer with a fixed profile, such as a list of liked genres or a row of past ratings, and treat the viewer's current mood either as a hard genre filter or not at all. Genre labels are too coarse to describe what a film feels like, keyword-based text matching such as TF-IDF cannot connect synonyms, and a mood filter that ignores taste can return films the viewer would never enjoy.",
        "This project presents MovieMind, a web-based movie recommendation system built around one idea: movies, users and moods can live in the same mathematical space. Each movie is converted into a 384-dimensional vector by the all-MiniLM-L6-v2 Sentence Transformer applied to its plot, tagline, genres, keywords, director and cast. A user is represented by the centroid of the vectors of the movies they like. Each of ten moods is represented by a direction vector computed as the mean of a set of exemplar movies minus the mean of all movies. A recommendation is the set of movies nearest to the search position user_taste + λ × mood_direction, where λ is a number between 0 and 1 that the viewer controls with a slider.",
        "Because the user, the movies and the mood share one space, the system also explains each recommendation from the geometry: how well it matches the user's taste, how much the mood shift changed its ranking, and which previously liked movie it is closest to. The system is implemented with Flask, SQLite, NumPy and scikit-learn, runs on a CPU with no paid API, and adds a single dependency, sentence-transformers, to the earlier TF-IDF version of the project.",
        f"The final design is a hybrid. It blends content similarity with a collaborative signal (what people with similar likes enjoyed) and a quality prior into a base score, retrieves the 60 best unseen films, and then lets the chosen mood re-rank them by the strength λ, so the mood changes the order without lowering match quality. The approach was evaluated offline on MovieLens with users held out: item statistics come from 60% of users, blend weights are tuned on 20% and results are reported on the remaining 20%, over five random splits. Hit Rate@10 is {HM['Hybrid, no mood']['hr']['mean']:.2f} for the hybrid, against {HM['B1: TF-IDF + genre (v1)']['hr']['mean']:.2f} for the earlier TF-IDF engine, {HM['Content only (semantic centroid)']['hr']['mean']:.2f} for content embeddings alone and {HM['POP: most-liked films (train crowd)']['hr']['mean']:.2f} for recommending the most popular films, and the hybrid is ahead of popularity in every split. The first design, which moved the search point toward the mood, trailed popularity; the redesign fixed this. The mood re-rank changes about two of ten titles at the default strength without a measurable loss in accuracy, and with an exactly known mood it improves hit rate, but the offline data contain no mood labels, so whether a user-chosen mood improves satisfaction remains untested.",
        "The system was implemented end to end with 99 automated tests, answers a request in a median of 3.6 ms on a CPU, and has a redesigned interface with a mood slider and a taste summary. The report documents the design, both evaluations, the limitations (small dataset, no user study, noisy weight selection) and a roadmap that includes stronger collaborative models, multi-interest taste and a user study.",
    ]
    abs_xml = "".join(para_like(k[58], L.runs(t)) for t in abstract)
    kw = para_like(k[63], L.runs("Keywords: Recommender System, Sentence Embeddings, Semantic Search, Mood-Aware Recommendation, Explainable AI, Content-Based Filtering, Cosine Similarity, Flask, MovieLens."))

    # ---------------- body
    body_xml = "".join(C.body)

    # ---------------- front-matter lists
    front_pages = pages.get("front", {})
    toc = []
    for t in ["Acknowledgement", "Abstract", "List of Figures", "List of Tables", "List of Abbreviations"]:
        toc.append(("front", "", t, front_pages.get(t, "")))
    sec_i = 0
    for n, t in C.chaps:
        toc.append(("chap", str(n), t, pages.get(f"CHAPTER {n}", "")))
        for s in C.secs:
            if s.startswith(f"{n}.") and re.match(r"^\d+\.\d+\s", s):
                toc.append(("sec", "", s, pages.get(s, "")))
        if n == 9:
            toc.append(("sec", "", "Patent / Paper Publication Proof", pages.get("Patent / Paper Publication Proof", "")))
    toc_xml = L.toc_table(toc)
    lof = L.plain_table(["Figure No.", "Description", "Page No."],
                        [[n, dsc, pages.get(f"Figure {n}:", "")] for n, dsc in C.figs], [1500, 6000, 1500], align=["center", "left", "center"])
    lot = L.plain_table(["Table No.", "Description", "Page No."],
                        [[n, dsc, pages.get(f"Table {n}:", "")] for n, dsc in C.tabs], [1500, 6000, 1500], align=["center", "left", "center"])
    loa = L.plain_table(["Abbreviation", "Expansion"], [list(a) for a in C.ABBR], [2600, 6400])

    new = k[:58] + [abs_xml, kw] + k[64:66] + [toc_xml] + k[67:69] + [lof] + k[70:72] + [lot] + k[73:75] + [loa]
    doc = head + "".join(new) + body_xml + sect + tail

    # ---------------- relationships / media
    relp = os.path.join(WORK, "word", "_rels", "document.xml.rels")
    rels = open(relp, encoding="utf8").read()
    used = set(re.findall(r'r:embed="(rId\d+)"', doc)) | set(re.findall(r'r:id="(rId\d+)"', doc))
    for rid, path in C.images.items():
        pass
    # drop unused image rels + files
    img_rels = re.findall(r'<Relationship Id="(rId\d+)" Type="[^"]*/image" Target="([^"]+)"/>', rels)
    for rid, target in img_rels:
        if rid not in used:
            rels = re.sub(rf'<Relationship Id="{rid}" [^>]*/>', "", rels)
            f = os.path.join(WORK, "word", target)
            if os.path.exists(f):
                os.remove(f)
    add = ""
    for i, (rid, path) in enumerate(C.images.items()):
        fn = f"mm_{i:02d}.png"
        shutil.copy(path, os.path.join(WORK, "word", "media", fn))
        add += f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/>'
    rels = rels.replace("</Relationships>", add + "</Relationships>")
    open(relp, "w", encoding="utf8").write(rels)
    open(docp, "w", encoding="utf8").write(doc)

    # core props
    cp = os.path.join(WORK, "docProps", "core.xml")
    if os.path.exists(cp):
        c = open(cp, encoding="utf8").read()
        c = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{TITLE_FULL}</dc:title>", c)
        c = re.sub(r"<dc:creator>.*?</dc:creator>", f"<dc:creator>{NAME}</dc:creator>", c)
        open(cp, "w", encoding="utf8").write(c)

    # ---------------- zip (keep original order, mimetype-first not needed for docx)
    if os.path.exists(FINAL):
        os.remove(FINAL)
    with zipfile.ZipFile(FINAL, "w", zipfile.ZIP_DEFLATED) as z:
        order = [n for n in names if os.path.exists(os.path.join(WORK, n)) and not n.endswith("/")]
        extra = [os.path.relpath(os.path.join(r, f), WORK) for r, _, fs in os.walk(WORK) for f in fs]
        for n in order + [e for e in extra if e not in order]:
            z.write(os.path.join(WORK, n), n)


def render():
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", OUT, FINAL], check=True, capture_output=True)
    pdf = FINAL.replace(".docx", ".pdf")
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    texts = []
    for i in range(1, n + 1):
        t = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", pdf, "-"], capture_output=True, text=True).stdout
        texts.append(re.sub(r"\s+", " ", t))
    return texts


def find_pages(texts):
    pages, front = {}, {}
    norm = lambda s: re.sub(r"\s+", " ", s)
    for key, up in [("Acknowledgement", "ACKNOWLEDGEMENT"), ("Abstract", "ABSTRACT"), ("List of Figures", "LIST OF FIGURES"),
                    ("List of Tables", "LIST OF TABLES"), ("List of Abbreviations", "LIST OF ABBREVIATIONS")]:
        for i, t in enumerate(texts, 1):
            if up in t and "CERTIFICATE" not in t[:60]:
                front[key] = i
                break
    pages["front"] = front
    start = front.get("List of Abbreviations", 12) + 1
    keys = [f"CHAPTER {n}" for n, _ in C.chaps] + [norm(s) for s in C.secs] + [f"Figure {n}:" for n, _ in C.figs] + [f"Table {n}:" for n, _ in C.tabs]
    for key in keys:
        for i in range(start, len(texts) + 1):
            t = texts[i - 1]
            if key.startswith(("Figure", "Table")):
                hit = key in t and re.search(re.escape(key) + r" [A-Z]", t)
            else:
                hit = key in t
            if hit:
                pages[key] = i
                break
    # secs stored with original spacing
    for s in C.secs:
        if norm(s) in pages:
            pages[s] = pages[norm(s)]
    return pages


if __name__ == "__main__":
    pages = {}
    build(pages)
    texts = render()
    pages = find_pages(texts)
    build(pages)
    texts = render()
    pages2 = find_pages(texts)
    if pages2 != pages:  # TOC length may shift pages once; rebuild again
        build(pages2)
        texts = render()
    json.dump(pages2, open(os.path.join(OUT, "pages.json"), "w"), indent=1)
    print("pages:", len(texts), "->", FINAL)
