"""Generates all report diagrams (PNG) with PIL. Run: python diagrams.py

Every figure here is a *design* diagram. Data-driven figures (screenshots,
metric plots) are produced as clearly-marked placeholders until v2 is run.
"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
S = 2  # supersampling factor
FD = "/usr/share/fonts/truetype/liberation/"
NAVY, BLUE, TEAL, ORANGE, GREEN, PURPLE = "#1F4E79", "#2E75B6", "#00897B", "#E67E22", "#2E8B57", "#6A4C93"
LBLUE, LTEAL, LORANGE, LGREEN, LPURPLE, GREY, LGREY = "#DCE9F5", "#D5EFEA", "#FCE8D5", "#DDF1E3", "#E8E0F0", "#555555", "#F2F2F2"


def font(size, bold=False, mono=False, italic=False):
    if mono:
        return ImageFont.truetype(FD + "LiberationMono-Regular.ttf", size * S)
    name = "LiberationSans-" + ("BoldItalic" if bold and italic else "Bold" if bold else "Italic" if italic else "Regular")
    return ImageFont.truetype(FD + name + ".ttf", size * S)


class Canvas:
    def __init__(self, w, h, bg="white"):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * S, h * S), bg)
        self.d = ImageDraw.Draw(self.img)

    # -- primitives -------------------------------------------------
    def _wrap(self, text, f, maxw):
        out = []
        for para in text.split("\n"):
            words, line = para.split(" "), ""
            for wd in words:
                t = (line + " " + wd).strip()
                if f.getlength(t) <= maxw * S or not line:
                    line = t
                else:
                    out.append(line)
                    line = wd
            out.append(line)
        return out

    def text(self, x, y, s, fs=18, bold=False, color="black", anchor="mm", mono=False, italic=False, maxw=None, lead=1.25):
        f = font(fs, bold, mono, italic)
        lines = self._wrap(s, f, maxw) if maxw else s.split("\n")
        lh = fs * lead * S
        y0 = y * S - (len(lines) - 1) * lh / 2 if anchor[1] == "m" else y * S
        for i, ln in enumerate(lines):
            self.d.text((x * S, y0 + i * lh), ln, font=f, fill=color, anchor=anchor)

    def box(self, x, y, w, h, text="", fill=LBLUE, outline=NAVY, fs=17, bold=False, color="black", r=12, lw=2, dash=False, maxw=None):
        if dash:
            self.dashed_rect(x, y, w, h, outline, lw)
            if fill:
                self.d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], r * S, fill=fill)
                self.dashed_rect(x, y, w, h, outline, lw)
        else:
            self.d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], r * S, fill=fill, outline=outline, width=lw * S)
        if text:
            self.text(x + w / 2, y + h / 2, text, fs, bold, color, maxw=maxw or (w - 16))

    def dashed_rect(self, x, y, w, h, color, lw=2, dash=10):
        for (x1, y1, x2, y2) in [(x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)]:
            self.line(x1, y1, x2, y2, color, lw, dash=dash)

    def line(self, x1, y1, x2, y2, color="black", lw=2, dash=0):
        if not dash:
            self.d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=color, width=lw * S)
            return
        L = math.hypot(x2 - x1, y2 - y1)
        n = int(L // (dash * 1.6)) + 1
        for i in range(n):
            a = i * dash * 1.6 / L
            b = min(1, (i * dash * 1.6 + dash) / L)
            if a >= 1:
                break
            self.d.line([(x1 + (x2 - x1) * a) * S, (y1 + (y2 - y1) * a) * S, (x1 + (x2 - x1) * b) * S, (y1 + (y2 - y1) * b) * S], fill=color, width=lw * S)

    def arrow(self, pts, color="black", lw=2, head=11, dash=0):
        for a, b in zip(pts[:-1], pts[1:]):
            self.line(a[0], a[1], b[0], b[1], color, lw, dash)
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        p = [(x2, y2), (x2 - head * math.cos(ang - 0.4), y2 - head * math.sin(ang - 0.4)), (x2 - head * math.cos(ang + 0.4), y2 - head * math.sin(ang + 0.4))]
        self.d.polygon([(px * S, py * S) for px, py in p], fill=color)

    def ellipse(self, cx, cy, rx, ry, text="", fill=LBLUE, outline=NAVY, fs=15, lw=2, color="black"):
        self.d.ellipse([(cx - rx) * S, (cy - ry) * S, (cx + rx) * S, (cy + ry) * S], fill=fill, outline=outline, width=lw * S)
        if text:
            self.text(cx, cy, text, fs, color=color, maxw=rx * 1.7)

    def diamond(self, cx, cy, w, h, text="", fill=LORANGE, outline=ORANGE, fs=15):
        pts = [(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)]
        self.d.polygon([(px * S, py * S) for px, py in pts], fill=fill, outline=outline, width=2 * S)
        if text:
            self.text(cx, cy, text, fs, maxw=w * 0.62)

    def dot(self, x, y, r, fill, outline=None):
        self.d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=fill, outline=outline)

    def stick(self, x, y, label, color=NAVY):
        self.d.ellipse([(x - 12) * S, (y - 12) * S, (x + 12) * S, (y + 12) * S], outline=color, width=3 * S)
        for a in [(x, y + 12, x, y + 48), (x - 22, y + 28, x + 22, y + 28), (x, y + 48, x - 18, y + 76), (x, y + 48, x + 18, y + 76)]:
            self.line(*a, color, 3)
        self.text(x, y + 98, label, 16, True, color)

    def save(self, name):
        img = self.img.resize((self.w, self.h), Image.LANCZOS)
        img.save(os.path.join(OUT, name), dpi=(150, 150))
        print("wrote", name)


def title(c, s):
    c.text(c.w / 2, 26, s, 22, True, NAVY)


# ---------------------------------------------------------------- 2.1
def fig_2_1():
    c = Canvas(1500, 760)
    title(c, "End-to-end block view of MovieMind")
    y = 120
    xs = [30, 330, 630, 930, 1230]
    labels = [("Movie Metadata\n(plot, tagline, genres,\nkeywords, cast, director)", LGREY, GREY),
              ("Sentence Transformer\nall-MiniLM-L6-v2", LBLUE, NAVY),
              ("Movie Embedding Matrix\n(N x 384, .npy)", LPURPLE, PURPLE),
              ("Recommendation Engine\nlambda-shifted cosine search", LTEAL, TEAL),
              ("Flask REST API + Web UI\nTop-k movies + explanations", LGREEN, GREEN)]
    for x, (t, f, o) in zip(xs, labels):
        c.box(x, y, 240, 130, t, f, o, fs=17, bold=True)
    for a, b in zip(xs[:-1], xs[1:]):
        c.arrow([(a + 240, y + 65), (b, y + 65)], NAVY, 3)
    c.text(750, 90, "Offline: encode once", 16, True, GREY, italic=True)
    c.dashed_rect(20, 100, 870, 170, GREY, 2)
    # lower branch
    c.box(630, 400, 240, 110, "Mood Direction Vectors\n(10 x 384, .npy)", LORANGE, ORANGE, fs=17, bold=True)
    c.arrow([(750, 400), (750, 270 + 0)], ORANGE, 3)
    c.line(750, 270, 750, 250, ORANGE, 3)
    c.arrow([(750, 400), (750, 250)], ORANGE, 3)
    c.box(930, 400, 240, 110, "User Taste Centroid\n(384-d, stored per user)", LTEAL, TEAL, fs=17, bold=True)
    c.arrow([(1050, 400), (1050, 250)], TEAL, 3)
    c.box(1230, 400, 240, 110, "User Controls\nLike / Unlike, Mood,\nlambda slider", LBLUE, NAVY, fs=17, bold=True)
    c.arrow([(1350, 400), (1350, 250)], NAVY, 3)
    c.arrow([(1230, 455), (1170, 455)], NAVY, 3)
    c.arrow([(930, 455), (870, 455)], NAVY, 3)
    c.arrow([(930, 205), (870, 205)], NAVY, 0)
    c.text(750, 590, "base = content + collaborative + quality + popularity;   mood re-ranks the top 60 (lambda)", 20, True, NAVY)
    c.text(750, 640, "recommendations = top-k after mood re-ranking and the diversity cap", 18, False, GREY)
    c.save("fig_2_1_block.png")


# ---------------------------------------------------------------- 2.2
def fig_2_2():
    c = Canvas(1500, 1000)
    title(c, "Data flow / system architecture (offline and online pipelines)")
    c.box(20, 60, 1460, 300, "", "#F7F9FC", GREY, dash=True)
    c.text(40, 78, "OFFLINE PIPELINE  (run once, or when the catalogue changes)", 16, True, GREY, anchor="lm")
    ox = [40, 330, 620, 910, 1200]
    items = [("Movie DB\n(SQLite)", LGREY, GREY), ("Rich text builder\nplot + tagline + genres\n+ keywords + cast", LBLUE, NAVY),
             ("Sentence\nTransformer", LBLUE, NAVY), ("movie_embeddings.npy\n(N x 384)", LPURPLE, PURPLE), ("mood_vectors.npy\n(10 x 384)", LORANGE, ORANGE)]
    for x, (t, f, o) in zip(ox, items):
        c.box(x, 120, 230, 110, t, f, o, fs=16, bold=True)
    for a, b in zip(ox[:-1], ox[1:]):
        c.arrow([(a + 230, 175), (b, 175)], NAVY, 3)
    c.box(660, 270, 520, 70, "Mood direction = mean(mood exemplars) - mean(all movies), unit-normalised", LORANGE, ORANGE, fs=15)
    c.arrow([(1035, 230), (1035, 270)], ORANGE, 2)
    c.arrow([(1180, 305), (1315, 305), (1315, 230)], ORANGE, 2)
    c.box(20, 390, 1460, 580, "", "#F7FBF9", GREY, dash=True)
    c.text(40, 408, "ONLINE PIPELINE  (per user request)", 16, True, GREY, anchor="lm")
    c.box(40, 450, 230, 90, "User likes / unlikes\na movie", LBLUE, NAVY, fs=16, bold=True)
    c.box(330, 450, 260, 90, "Taste centroid update\n(mean of liked embeddings)", LTEAL, TEAL, fs=16, bold=True)
    c.box(650, 450, 260, 90, "users.taste_embedding\n(384 floats, JSON)", LGREY, GREY, fs=16, bold=True)
    c.arrow([(270, 495), (330, 495)], NAVY, 3)
    c.arrow([(590, 495), (650, 495)], NAVY, 3)
    c.box(40, 640, 230, 100, "User picks mood\nand lambda", LBLUE, NAVY, fs=16, bold=True)
    c.box(330, 640, 260, 100, "Blend content + collab +\nquality scores (top 60)", LTEAL, TEAL, fs=16, bold=True)
    c.box(650, 640, 260, 100, "Mood re-rank (lambda)\n+ diversity cap", LTEAL, TEAL, fs=16, bold=True)
    c.box(970, 640, 240, 100, "Exclude liked and\nfavourited titles", LTEAL, TEAL, fs=16, bold=True)
    c.box(1260, 640, 200, 100, "Top-k + geometric\nexplanations", LGREEN, GREEN, fs=16, bold=True)
    for a, b in [(270, 330), (590, 650), (910, 970), (1210, 1260)]:
        c.arrow([(a, 690), (b, 690)], NAVY, 3)
    c.arrow([(780, 540), (780, 590), (460, 590), (460, 640)], GREY, 2)
    c.text(620, 575, "taste_vector", 14, False, GREY, italic=True)
    # stores feeding online
    c.box(970, 450, 260, 90, "movie_embeddings.npy", LPURPLE, PURPLE, fs=15, bold=True)
    c.box(1260, 450, 200, 90, "mood_vectors.npy", LORANGE, ORANGE, fs=15, bold=True)
    c.arrow([(1100, 540), (1100, 600), (780, 600), (780, 640)], PURPLE, 2)
    c.arrow([(1360, 540), (1360, 580), (520, 580), (520, 640)], ORANGE, 2)
    c.text(750, 790, "Stores: SQLite (movies, users, likes, favourites, watchlist) and NumPy arrays loaded at start-up; item_stats.npz (quality, popularity, item-item similarity) is built offline from MovieLens ratings.", 13, False, GREY, italic=True, maxw=1300)
    c.box(40, 840, 1420, 100, "Response (JSON): movie id, title, genres, similarity score, taste contribution, mood contribution, nearest liked movie, rank change caused by mood", LGREEN, GREEN, fs=16)
    c.arrow([(1360, 740), (1360, 840)], GREEN, 3)
    c.save("fig_2_2_dfd.png")


# ---------------------------------------------------------------- 2.3
def fig_2_3():
    c = Canvas(1400, 900)
    title(c, "Use case diagram")
    c.box(260, 60, 880, 800, "", "#FBFBFB", NAVY, lw=2)
    c.text(700, 85, "MovieMind System", 18, True, NAVY)
    ucs = ["Sign up / Log in", "Browse and search movies", "Like / unlike a movie", "Select a mood", "Adjust lambda slider",
           "View recommendations", "View recommendation explanation", "Manage movie catalogue", "Rebuild embeddings and mood vectors"]
    pos = {}
    ys = [150, 235, 320, 405, 490, 575, 660, 745, 810]
    for i, (u, y) in enumerate(zip(ucs, ys)):
        x = 520 if i < 7 else 900
        yy = y if i < 7 else (190 if i == 7 else 290)
        fill = LTEAL if i < 7 else LORANGE
        c.ellipse(x, yy, 190, 36, u, fill, TEAL if i < 7 else ORANGE, fs=15)
        pos[i] = (x, yy)
    c.stick(110, 330, "User", NAVY)
    c.stick(1290, 330, "Administrator", ORANGE)
    for i in range(7):
        c.line(135, 360, pos[i][0] - 190 + 6, pos[i][1], NAVY, 2)
    for i in (7, 8):
        c.line(1265, 360, pos[i][0] + 190 - 6, pos[i][1], ORANGE, 2)
    r = 710
    c.arrow([(r, pos[3][1]), (r + 40, pos[3][1]), (r + 40, pos[5][1] - 8), (r, pos[5][1] - 8)], GREY, 2, 10, dash=8)
    c.arrow([(r, pos[4][1]), (r + 20, pos[4][1]), (r + 20, pos[5][1] + 8), (r, pos[5][1] + 8)], GREY, 2, 10, dash=8)
    c.text(r + 140, 500, "<<extend>>", 13, False, GREY, italic=True)
    c.arrow([(r, pos[6][1]), (r + 70, pos[6][1]), (r + 70, pos[5][1]), (r, pos[5][1])], GREY, 2, 10, dash=8)
    c.text(r + 140, 635, "<<include>>", 13, False, GREY, italic=True)
    c.save("fig_2_3_usecase.png")


# ---------------------------------------------------------------- 2.4
def fig_2_4():
    c = Canvas(1400, 1330)
    title(c, "Activity diagram - hybrid recommendation with mood re-ranking")
    cx = 500
    c.dot(cx, 70, 14, "black")
    c.box(cx - 190, 120, 380, 55, "User requests recommendations (mood, lambda, k)", LBLUE, NAVY, fs=16)
    c.arrow([(cx, 84), (cx, 120)], "black", 2)
    c.box(cx - 190, 205, 380, 55, "Load liked titles and taste centroid", LBLUE, NAVY, fs=16)
    c.arrow([(cx, 175), (cx, 205)], "black", 2)
    c.arrow([(cx, 260), (cx, 295)], "black", 2)
    c.diamond(cx, 335, 330, 80, "Has liked titles?")
    c.box(880, 308, 330, 60, "Cold start: preferred-genre\ncentroid; no collaborative part", LGREY, GREY, fs=15)
    c.arrow([(cx + 165, 335), (880, 335)], "black", 2)
    c.text(cx + 200, 320, "No", 15, True)
    c.arrow([(cx, 375), (cx, 430)], "black", 2)
    c.text(cx + 25, 402, "Yes", 15, True)
    c.box(cx - 220, 430, 440, 70, "Score every movie: content similarity,\ncollaborative score (liked together), quality, popularity", LTEAL, TEAL, fs=15)
    c.arrow([(1045, 368), (1045, 465), (cx + 220, 465)], "black", 2)
    steps = [(550, "Blend: base = sum of weight x z-score", LTEAL, TEAL), (635, "Exclude titles already liked or favourited", LTEAL, TEAL),
             (720, "Retrieve the top 60 candidates by base score", LTEAL, TEAL)]
    c.arrow([(cx, 500), (cx, 550)], "black", 2)
    for i, (y, t, f, o) in enumerate(steps):
        c.box(cx - 220, y, 440, 55, t, f, o, fs=16)
        if i < len(steps) - 1:
            c.arrow([(cx, y + 55), (cx, steps[i + 1][0])], "black", 2)
    c.arrow([(cx, 775), (cx, 810)], "black", 2)
    c.diamond(cx, 850, 380, 80, "Mood chosen and lambda > 0?")
    c.box(880, 823, 330, 55, "final = base", LGREY, GREY, fs=16)
    c.arrow([(cx + 190, 850), (880, 850)], "black", 2)
    c.text(cx + 225, 835, "No", 15, True)
    c.arrow([(cx, 890), (cx, 940)], "black", 2)
    c.text(cx + 25, 917, "Yes", 15, True)
    c.box(cx - 250, 940, 500, 64, "final = (1 - lambda) x base + lambda x mood fit\n(re-orders the candidates only)", LORANGE, ORANGE, fs=15, bold=True)
    c.arrow([(1045, 878), (1045, 972), (cx + 250, 972)], "black", 2)
    c.arrow([(cx, 1004), (cx, 1050)], "black", 2)
    c.box(cx - 220, 1050, 440, 55, "Diversity cap per vibe region; take top k", LTEAL, TEAL, fs=16)
    c.arrow([(cx, 1105), (cx, 1145)], "black", 2)
    c.box(cx - 220, 1145, 440, 55, "Explain each pick from the numbers", LGREEN, GREEN, fs=16)
    c.arrow([(cx, 1200), (cx, 1245)], "black", 2)
    c.dot(cx, 1258, 14, "black")
    c.d.ellipse([(cx - 20) * S, (1258 - 20) * S, (cx + 20) * S, (1258 + 20) * S], outline="black", width=2 * S)
    c.text(cx + 120, 1258, "Return JSON", 15, False, GREY, italic=True, anchor="lm")
    c.save("fig_2_4_activity.png")


# ---------------------------------------------------------------- 2.5
def fig_2_5():
    c = Canvas(1500, 860)
    title(c, "UML collaboration diagram - recommendation request")
    nodes = {"B": (150, 330, "Browser\n(dashboard.html)"), "R": (650, 330, "Recommendation\nroute (Flask)"),
             "E": (1150, 330, "Recommendation\nEngine"), "S": (1150, 110, "EmbeddingStore\n(.npy matrices)"),
             "D": (650, 620, "Database\n(SQLAlchemy)"), "X": (1150, 600, "Explainer")}
    for k, (x, y, t) in nodes.items():
        grey = k in "SD"
        c.box(x - 110, y - 45, 220, 90, t, LGREY if grey else LBLUE, GREY if grey else NAVY, fs=16, bold=True)

    def link(p, q, label, lx, ly, col=NAVY):
        c.arrow([p, q], col, 2)
        c.text(lx, ly, label, 14, True, col, maxw=230)

    link((260, 310), (540, 310), "1: GET /api/recommend\n?mood=&lambda=&k=", 400, 272)
    link((540, 355), (260, 355), "8: JSON response", 400, 392)
    link((650, 375), (650, 575), "2: load user and\ntaste_embedding", 540, 475)
    link((760, 310), (1040, 310), "3: get_recommendations\n(user, mood, lambda)", 900, 272)
    link((1040, 355), (760, 355), "7: ranked movies +\nexplanations", 900, 396)
    link((1100, 285), (1100, 155), "4: embeddings,\nmood vector", 1000, 220)
    link((1100, 375), (1100, 555), "5: explain(top-k)", 1000, 465)
    link((1200, 555), (1200, 375), "6: contributions", 1300, 465)
    c.box(60, 720, 1380, 110, "", "white", GREY, r=6, lw=1)
    c.text(80, 745, "Messages: 1 browser requests recommendations; 2 route loads the user and taste vector from the database; 3 route calls the engine; "
                    "4 engine reads movie embeddings and the mood vector; 5-6 engine asks the explainer for taste / mood contributions; 7 engine returns ranked "
                    "movies with explanations; 8 route returns JSON. Liking or unliking a movie separately triggers TasteUpdater, which rewrites users.taste_vector.",
           14, False, GREY, anchor="lt", maxw=1340, lead=1.4)
    c.save("fig_2_5_collab.png")


# ---------------------------------------------------------------- 2.6
def fig_2_6():
    c = Canvas(1500, 1000)
    title(c, "UML class diagram (key classes)")

    def cls(x, y, w, name, attrs, meths, color=NAVY, fill=LBLUE):
        h1 = 40
        h2 = 22 * len(attrs) + 14
        h3 = 22 * len(meths) + 14
        c.box(x, y, w, h1 + h2 + h3, "", "white", color, r=4)
        c.d.rectangle([x * S, y * S, (x + w) * S, (y + h1) * S], fill=fill, outline=color, width=2 * S)
        c.text(x + w / 2, y + h1 / 2, name, 18, True, color)
        for i, a in enumerate(attrs):
            c.text(x + 10, y + h1 + 18 + 22 * i, a, 14, anchor="lm", mono=False)
        c.line(x, y + h1 + h2, x + w, y + h1 + h2, color, 2)
        for i, m in enumerate(meths):
            c.text(x + 10, y + h1 + h2 + 18 + 22 * i, m, 14, anchor="lm")
        return (x, y, w, h1 + h2 + h3)

    u = cls(30, 80, 400, "User", ["user_id: int", "username, email: str", "password_hash: str", "taste_vector: JSON (v1 genre profile)", "taste_embedding: JSON (384 floats)", "preferred_genres: JSON"],
            ["set_password(pw) / check_password(pw)", "get_taste_vector() / set_taste_vector(v)"])
    m = cls(30, 470, 400, "Movie", ["movie_id, tmdb_id: int", "title, director: str", "overview, tagline: text", "genres, cast, keywords: JSON", "vote_average, popularity: float"],
            ["get_genres_list() / get_cast_list()", "to_dict()"])
    f = cls(30, 790, 400, "UserLike / UserFavorite / UserWatchlist", ["user_id, movie_id: FK (unique pair)", "UserLike: preference signal (teaches taste)", "UserFavorite, UserWatchlist: collections only", "UserInteraction logs every action with its type"], [])
    e = cls(560, 80, 400, "EmbeddingStore", ["E: ndarray (N x 384), ids", "id_to_row: dict[int, int]", "mood_vectors: dict[str, ndarray]", "clusters: ndarray, global_mean"],
            ["load(models_dir)", "centroid(movie_ids)", "search_position(taste, mood, lam)", "scores(position), similar_to(id, n)"], PURPLE, LPURPLE)
    t = cls(560, 400, 400, "taste_updater", [], ["refresh_embedding_taste(user_id)", "# recomputes the unit centroid from the", "# favourites table on every like / unlike"], TEAL, LTEAL)
    r = cls(1040, 80, 430, "EmbeddingRecommender", ["store: EmbeddingStore", "DEFAULT_LAMBDA = 0.3"],
            ["get_recommendations(user_id, mood, limit, lam)", "_hybrid(...)  # blend, retrieve, re-rank", "taste_for(user, liked_ids)", "get_similar_movies(movie_id, n)"], TEAL, LTEAL)
    x = cls(1040, 430, 430, "geometric_explanation", [], ["taste_match = cos(taste, movie)", "mood_effect = cos(search, movie) - taste_match", "nearest_liked, rank_without / with mood", "text built from these numbers"], GREEN, LGREEN)
    b = cls(560, 700, 400, "offline scripts", [], ["build_embeddings.py: texts -> vectors, moods", "build_item_stats.py: quality, popularity,", "  item-item similarity (hybrid.py)", "evaluate_hybrid.py: held-out evaluation"], ORANGE, LORANGE)
    c.arrow([(430, 250), (560, 250)], GREY, 2, 10)
    c.arrow([(960, 230), (1040, 230)], GREY, 2, 10)
    c.arrow([(960, 480), (1040, 540)], GREY, 2, 10)
    c.arrow([(230, 360), (230, 470)], GREY, 2, 10)
    c.arrow([(230, 770), (230, 790)], GREY, 2, 10)
    c.arrow([(760, 400), (760, 335)], GREY, 2, 10)
    c.arrow([(760, 700), (760, 335 + 0)], GREY, 2, 10) if False else None
    c.arrow([(960, 770), (1250, 770), (1250, 640)], GREY, 2, 10) if False else None
    c.text(745, 960, "Arrows show 'uses / depends on' relationships.", 14, False, GREY, italic=True)
    c.save("fig_2_6_class.png")


# ---------------------------------------------------------------- 4.1
def fig_4_1():
    c = Canvas(1500, 640)
    title(c, "Movie encoding pipeline")
    c.box(30, 100, 330, 250, "", LGREY, GREY)
    c.text(195, 125, "Rich text per movie", 17, True, GREY)
    c.text(45, 160, "plot summary + tagline\n+ genres (x2)\n+ keywords\n+ director\n+ top cast", 16, anchor="lt", lead=1.5)
    c.arrow([(360, 225), (450, 225)], NAVY, 3)
    c.box(450, 100, 330, 250, "", LBLUE, NAVY)
    c.text(615, 125, "Sentence Transformer", 18, True, NAVY)
    c.text(615, 160, "all-MiniLM-L6-v2", 15, False, NAVY, italic=True)
    for i, t in enumerate(["Tokenizer", "6 transformer layers", "Mean pooling", "L2 normalise"]):
        c.box(480, 190 + i * 38, 270, 30, t, "white", NAVY, fs=14, r=6)
    c.arrow([(780, 225), (870, 225)], NAVY, 3)
    c.box(870, 100, 280, 250, "", LPURPLE, PURPLE)
    c.text(1010, 125, "384-d vector", 18, True, PURPLE)
    c.text(1010, 215, "[0.23, -0.45, 0.78,\n 0.12, -0.33, ... ]", 16, anchor="mm", mono=True, lead=1.4)
    c.text(1010, 300, "illustrative values", 13, False, GREY, italic=True)
    c.arrow([(1150, 225), (1230, 225)], NAVY, 3)
    c.box(1230, 100, 240, 250, "", LGREEN, GREEN)
    c.text(1350, 125, "movie_embeddings.npy", 15, True, GREEN)
    c.text(1350, 215, "N x 384 float32\nrow i = movie i", 15, lead=1.5)
    c.text(750, 420, "Memory for the 4,988 enriched movies:  4,988 x 384 x 4 bytes  =  7.66 MB  (plus ~80 MB model, loaded only offline)", 17, False, NAVY)
    c.text(750, 470, "Related experiences (e.g. 'heist', 'robbery', 'caper') land near each other even with zero shared words.", 16, False, GREY, italic=True)
    c.save("fig_4_1_encoding.png")


# ---------------------------------------------------------------- 4.2 / 4.3 / 4.4 (illustrative geometry)
def _map(c, x0, y0, w, h):
    c.box(x0, y0, w, h, "", "#FAFBFD", GREY, r=6, lw=1)


def fig_4_2():
    rng = np.random.default_rng(7)
    c = Canvas(1500, 900)
    title(c, "Illustrative semantic map of movies, user taste and mood (schematic, not measured data)")
    _map(c, 40, 60, 1420, 780)
    clusters = [("Heist / caper", 420, 330, TEAL), ("Slow-burn horror", 1150, 250, PURPLE), ("Psychological mystery", 780, 600, BLUE),
                ("Action / high-energy", 1200, 640, ORANGE), ("Romance / drama", 250, 650, "#C2185B")]
    for name, cx, cy, col in clusters:
        for _ in range(26):
            a, r = rng.uniform(0, 2 * math.pi), abs(rng.normal(0, 55))
            c.dot(cx + r * math.cos(a), cy + r * math.sin(a), 5, col)
        c.text(cx, cy - 100, name, 16, True, col)
    c.dot(440, 360, 13, NAVY, "white")
    c.text(440, 420, "user taste (centroid of liked movies)", 15, True, NAVY, maxw=240)
    c.arrow([(452, 352), (1030, 280)], ORANGE, 4, 16, dash=0)
    c.text(760, 290, "mood direction x lambda", 16, True, ORANGE)
    c.d.ellipse([(1160 - 16) * S, (255 - 16) * S, (1160 + 16) * S, (255 + 16) * S], outline=ORANGE, width=3 * S) if False else None
    c.dot(720, 318, 11, GREEN, "white")
    c.text(720, 352, "search position", 15, True, GREEN)
    c.d.ellipse([(720 - 120) * S, (318 - 120) * S, (720 + 120) * S, (318 + 120) * S], outline=GREEN, width=2 * S)
    c.text(720, 466, "nearest movies", 14, False, GREEN, italic=True)
    c.text(750, 870, "Same-genre films (all 'Thriller') sit in different regions; mood moves the search point without leaving the user's neighbourhood.", 15, False, GREY, italic=True)
    c.save("fig_4_2_space.png")


def fig_4_3():
    rng = np.random.default_rng(3)
    c = Canvas(1500, 760)
    title(c, "Computing a mood direction vector (schematic)")
    _map(c, 40, 60, 1420, 560)
    for _ in range(140):
        c.dot(rng.uniform(80, 1000), rng.uniform(100, 580), 4, "#B0B7C3")
    for _ in range(22):
        c.dot(rng.normal(900, 40), rng.normal(230, 40), 6, ORANGE)
    c.dot(540, 340, 14, NAVY, "white")
    c.text(540, 380, "mean of ALL movies", 15, True, NAVY)
    c.dot(900, 230, 14, ORANGE, "white")
    c.text(900, 160, "mean of 'excited' exemplars", 15, True, ORANGE)
    c.arrow([(552, 332), (888, 238)], ORANGE, 5, 18)
    c.text(700, 262, "excited_direction", 17, True, ORANGE)
    c.box(1030, 100, 400, 480, "", "white", GREY, r=6)
    c.text(1230, 128, "Procedure", 17, True, NAVY)
    c.text(1050, 160, "1. Pick exemplar movies for the mood\n    (keyword / metadata rules)\n2. mood_center = mean(exemplar embeddings)\n3. global_center = mean(all embeddings)\n4. direction = mood_center - global_center\n5. Normalise to unit length\n6. Store in mood_vectors.npy",
           15, anchor="lt", lead=1.55)
    c.text(750, 670, "mood_direction = mean(mood_exemplars) - mean(all_movies)", 22, True, NAVY)
    c.text(750, 712, "Repeated for the 10 moods: happy, sad, excited, relaxed, romantic, thrilled, nostalgic, adventurous, thoughtful, scared.", 15, False, GREY, italic=True)
    c.save("fig_4_3_mooddir.png")


def fig_4_4():
    c = Canvas(1500, 800)
    title(c, "Effect of lambda on the search position (schematic)")
    _map(c, 40, 60, 1420, 560)
    start, end = np.array([200, 400]), np.array([1260, 250])
    c.dot(*start, 14, NAVY, "white")
    c.text(start[0], start[1] + 40, "user taste", 16, True, NAVY)
    c.dot(*end, 18, ORANGE, "white")
    c.text(end[0], end[1] - 40, "'excited' region", 16, True, ORANGE)
    c.arrow([tuple(start + [14, -3]), tuple(end - [20, -2])], "#BBBBBB", 3, 14)
    for lam, col in [(0.0, NAVY), (0.3, GREEN), (0.7, TEAL), (1.0, ORANGE)]:
        p = start + lam * (end - start) * 0.9
        c.dot(*p, 11 if lam else 14, col, "white")
        c.text(p[0], p[1] + (45 if lam in (0.3, 1.0) else -38), f"lambda = {lam}", 16, True, col)
    c.text(750, 575, "Default lambda = 0.3 (gentle nudge)", 16, True, GREEN)
    # slider
    c.line(200, 700, 1300, 700, GREY, 6)
    for lam, lab in [(0.0, "0.0\nPure taste"), (0.3, "0.3\nGentle nudge (default)"), (0.7, "0.7\nStrong shift"), (1.0, "1.0\nPure mood")]:
        x = 200 + lam * 1100
        c.dot(x, 700, 12, BLUE, "white")
        c.text(x, 745, lab, 15, True, NAVY)
    c.save("fig_4_4_lambda.png")


# ---------------------------------------------------------------- 4.5
def fig_4_5():
    c = Canvas(1400, 760)
    title(c, "Layered software architecture")
    layers = [("Presentation", "Sky Light design system (custom CSS), HTML templates, vanilla JavaScript: mood chips, lambda slider", LBLUE, NAVY),
              ("API / Application", "Flask 3 blueprints: auth, recommendation, user, pages;  Flask-Login sessions;  JSON REST endpoints", LTEAL, TEAL),
              ("Recommendation core", "NumPy / scikit-learn:  taste centroid, mood shift, cosine similarity, diversity filter, geometric explanations", LORANGE, ORANGE),
              ("Representation", "sentence-transformers (all-MiniLM-L6-v2):  movie_embeddings.npy, mood_vectors.npy  (built offline)", LPURPLE, PURPLE),
              ("Persistence", "SQLite via SQLAlchemy 2:  movies, users (taste_vector), user_favorites, user_interactions", LGREY, GREY),
              ("Data sources", "TMDB API (plots, cast, keywords, posters);  MovieLens ml-latest-small (ratings for evaluation)", LGREEN, GREEN)]
    for i, (n, d, f, o) in enumerate(layers):
        y = 70 + i * 112
        c.box(30, y, 250, 92, n, f, o, fs=19, bold=True)
        c.box(300, y, 1070, 92, d, "white", o, fs=16)
    c.save("fig_4_5_layers.png")


def fig_3_2_hybrid():
    c = Canvas(1500, 700)
    title(c, "Hybrid ranking pipeline")
    comps = [("Content", "cosine(taste, film)\nwhat the film is like", LPURPLE, PURPLE), ("Collaborative", "liked by people who\nliked what you liked", LBLUE, NAVY),
             ("Quality", "smoothed mean rating", LGREEN, GREEN), ("Popularity", "log(1 + likes)", LGREY, GREY)]
    for i, (n, d, f, o) in enumerate(comps):
        y = 80 + i * 120
        c.box(30, y, 300, 96, n + "\n" + d, f, o, fs=15, bold=False)
        c.arrow([(330, y + 48), (430, 300)], o, 2)
    c.box(430, 240, 260, 120, "Weighted blend of\nz-scores = base score\n(weights tuned on\nvalidation users)", LTEAL, TEAL, fs=15, bold=True)
    c.arrow([(690, 300), (780, 300)], NAVY, 3)
    c.box(780, 240, 220, 120, "Retrieve top 60\nunseen titles", LBLUE, NAVY, fs=16, bold=True)
    c.arrow([(1000, 300), (1090, 300)], NAVY, 3)
    c.box(1090, 190, 380, 220, "", LORANGE, ORANGE)
    c.text(1280, 222, "Mood re-rank", 18, True, ORANGE)
    c.text(1280, 300, "final = (1 - lambda) x base\n         + lambda x mood fit", 15, False, "black", mono=False)
    c.text(1280, 372, "lambda = 0 leaves the order unchanged", 13, False, GREY, italic=True)
    c.arrow([(1280, 410), (1280, 480)], NAVY, 3)
    c.box(1090, 480, 380, 80, "Diversity cap, top k + explanations", LGREEN, GREEN, fs=16, bold=True)
    c.box(430, 520, 560, 100, "The mood only reorders titles the engine already\njudged good, so it cannot pull in poor matches.", "white", GREY, fs=14, dash=True)
    c.save("fig_3_2_hybrid.png")


# ---------------------------------------------------------------- placeholders
def placeholder(name, label, w=1400, h=800):
    c = Canvas(w, h, "#FFF8E1")
    c.box(8, 8, w - 16, h - 16, "", "#FFF8E1", "#E0A800", r=6, lw=3, dash=True)
    c.text(w / 2, h / 2 - 40, "[ PLACEHOLDER ]", 40, True, "#B7791F")
    c.text(w / 2, h / 2 + 30, label, 24, False, "#7A5C00", maxw=w - 200)
    c.text(w / 2, h / 2 + 110, "Replace after v2 is built and run.", 18, False, "#7A5C00", italic=True)
    c.save(name)


def all_figs():
    for fn in [fig_2_1, fig_2_2, fig_2_3, fig_2_4, fig_2_5, fig_2_6, fig_3_2_hybrid, fig_4_1, fig_4_2, fig_4_3, fig_4_4, fig_4_5]:
        fn()
    placeholder("ph_5_1_login.png", "Screenshot: Login / Sign-up page", 1400, 700)
    placeholder("ph_5_2_dashboard.png", "Screenshot: Dashboard with mood selector and lambda slider")
    placeholder("ph_5_3_recs.png", "Screenshot: Recommendations with geometric explanations")
    placeholder("ph_5_4_terminal.png", "Terminal capture: build_embeddings.py output and API response", 1400, 700)
    placeholder("ph_7_1_lambda.png", "Chart: HR@10 and NDCG@10 versus lambda (0.0 to 1.0)")
    placeholder("ph_7_2_methods.png", "Chart: B1 / B2 / B3 / proposed method comparison", 1400, 700)


if __name__ == "__main__":
    all_figs()
