"""Report content for MovieMind. Uses the builders from docx_lib and registers headings/figures/tables
so the front-matter lists can be generated. Anything that depends on v2 results is a [[placeholder]]."""
import os
from docx_lib import *

FIG = os.path.join(os.path.dirname(__file__), "figures")
import ast, json as _json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = _json.load(open(os.path.join(os.path.dirname(__file__), "results", "results.json")))
LAT = _json.load(open(os.path.join(os.path.dirname(__file__), "results", "latency_api.json")))
META = _json.load(open(os.path.join(ROOT, "data", "models", "embeddings_meta.json")))
SYS = {x["system"]: x for x in R["systems"]}
POP, B1, B2, B3, PRO, CTL = [SYS[k] for k in SYS]
PAIR = R["paired_vs_B2"]
SWEEP = {(x["lambda"], x["mood_source"]): x for x in R["sweep"]}
f3 = lambda v: f"{v:.3f}"


def src(path, names, header=None):
    """Real source of top-level functions / class methods, extracted with ast (never hand-copied)."""
    text = open(os.path.join(ROOT, path)).read()
    tree = ast.parse(text)
    out = [f"# {header or path}"]
    for n in names:
        found = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == n:
                found = node; break
        assert found, f"{n} not found in {path}"
        seg = ast.get_source_segment(text, found)
        if isinstance(found, ast.FunctionDef) and found.col_offset:
            seg = "\n".join(l[found.col_offset:] if l.strip() else l for l in seg.split("\n"))
        out.append(seg)
    return "\n\n".join(out) + "\n"


body = []
figs, tabs, secs, chaps = [], [], [], []
images = {}  # rId -> png path


def add(*x):
    body.extend(x)


def chapter(n, title):
    chaps.append((n, title))
    add(h1(f"CHAPTER {n}", title))


def sec(text):
    secs.append(text)
    add(h2(text))


def figure_(num, desc, png, w=6.2):
    rid = f"rId{300 + len(images)}"
    images[rid] = os.path.join(FIG, png)
    figs.append((num, desc))
    add(figure(rid, images[rid], w, f"Figure {num}: {desc}"))


def table_(num, desc, headers, rows, widths, **kw):
    tabs.append((num, desc))
    add(caption(f"Table {num}: {desc}", before=120), table(headers, rows, widths, **kw))


P = lambda t: add(p_body(t))
B = lambda *items: add(*[p_bullet(i) for i in items])

# =========================================================== CHAPTER 1
chapter(1, "INTRODUCTION")
sec("1.1  Overview")
P("Choosing what to watch has become a problem of abundance rather than scarcity. Streaming catalogues contain tens of thousands of titles, and a viewer who opens an application on a tired weeknight is rarely looking for the same experience they wanted on Saturday afternoon. Recommender systems exist to narrow this space, but most of them model a viewer with a single, fixed profile: a list of genres liked in the past, or a row in a ratings matrix. Neither representation captures the fact that the same person wants different things on different days.")
P("This project presents **MovieMind**, a web-based movie recommendation system whose central idea is that **movies, users and moods can all be placed in the same mathematical space**. Every movie is converted into a 384-dimensional semantic vector (an embedding) by a pre-trained Sentence Transformer; a user is represented by the centroid of the embeddings of the movies they have liked; and each mood (happy, sad, excited, relaxed, and so on) is represented by a direction vector in that same space. A recommendation is produced by taking the user's position, nudging it a controllable distance in the direction of the current mood, and returning the nearest movies.")
P("The idea is summarised by a single equation, which the rest of this report develops:")
add(equation("search_position = user_taste + λ × mood_direction"))
P("Here λ (lambda) is a number between 0 and 1 that the user can change with a slider. With λ = 0 the system recommends purely from long-term taste; with λ = 1 it recommends purely from the selected mood; values in between let the viewer say \"my usual taste, but a little more exciting today\". Because the user, the movies and the mood live in the same space, the system can also explain every recommendation geometrically, for example by naming the liked movie that a suggestion is closest to and reporting how much of its ranking came from the mood shift.")
P("MovieMind is implemented as a Flask application with a SQLite database, NumPy and scikit-learn for vector arithmetic, and the `sentence-transformers` library for encoding. It runs entirely on a local machine, requires no GPU at query time, and uses no paid API. An earlier version of the system (referred to as v1 in this report) used TF-IDF vectors, genre-based taste profiles and a hard-coded mood-to-genre table; v1 is retained as the baseline against which the proposed approach is evaluated.")

sec("1.2  Scope of the Project")
P("MovieMind is designed as a complete, small-scale recommendation product rather than an isolated algorithm. Its scope covers the following areas:")
B("**Semantic content representation:** converting plot summaries, taglines, genres, keywords, director and cast into dense embeddings for a catalogue of roughly five thousand movies.",
  "**Personalisation:** maintaining a per-user taste vector that updates whenever a movie is liked or unliked.",
  "**Mood-aware discovery:** ten selectable moods, each backed by a learned direction vector, with a user-controlled intensity (λ).",
  "**Explainable output:** every recommendation is returned together with geometry-based reasons (taste match, mood effect, nearest liked movie).",
  "**Web application:** user registration and login, movie browsing and search, favourites, a profile page and a dashboard containing the mood selector and λ slider.",
  "**Evaluation:** an offline experiment on the MovieLens dataset comparing the proposed method against three baselines and sweeping λ from 0.0 to 1.0.")
P("The project deliberately does not attempt large-scale deployment, real-time streaming-platform integration or an online user study; these are discussed as future work in Chapter 8.")

sec("1.3  Objectives")
B("To replace sparse TF-IDF movie vectors with dense semantic embeddings so that related concepts (for example \"heist\", \"robbery\" and \"caper\") are recognised as similar even when they share no words.",
  "To represent each user as a point in the same embedding space as the movies, computed as the centroid of the movies the user likes.",
  "To learn a direction vector for each of ten moods from the data itself, instead of using a hand-written genre lookup table.",
  "To generate recommendations by a single operation, a λ-shifted nearest-neighbour search, and to expose λ as a user-facing slider.",
  "To produce explanations computed from actual vector relationships rather than from fixed text templates.",
  "To evaluate the approach against baselines using Hit Rate@10, NDCG@10, Diversity@10 and Serendipity@10, and to measure how quality changes as λ varies.",
  "To keep the whole system lightweight: CPU-only, locally runnable and free of paid services.")

sec("1.4  Existing System")
P("The existing system (v1) is the first version of MovieMind. It is a conventional hybrid content-based recommender with the following behaviour:")
B("**TF-IDF content vectors:** each movie's plot, genres, cast and keywords are converted to a sparse TF-IDF vector; similarity between two movies is the cosine of their vectors.",
  "**Genre-based taste profile:** a user is represented by a dictionary of genre weights, e.g. {Drama: 0.8, Thriller: 0.4}, updated by an exponential moving average with learning rate α = 0.2 whenever a movie is liked.",
  "**Hard-coded mood table:** each mood maps to a fixed set of genres with boost factors (for example \"excited\" boosts Action 1.5, Adventure 1.4, Thriller 1.3). Movies whose genres match are multiplied by the boost.",
  "**Weighted scoring:** the final score is a weighted sum of content similarity, genre match and mood boost, followed by a per-genre diversity cap.",
  "**Template explanations:** the explanation shown to the user is chosen from fixed sentence templates filled with a genre or movie name.")

sec("1.5  Existing System Disadvantages")
B("**Genre labels do not describe experience:** four films tagged \"Thriller\" may be a witty heist comedy, a slow-burn occult horror, a twisty domestic drama and a gritty procedural. A genre-based profile treats them as interchangeable.",
  "**Keyword matching misses meaning:** TF-IDF gives zero similarity to \"bank heist\" versus \"robbery\" or \"mind-bending\" versus \"cerebral\", because it only counts shared words.",
  "**Mood and taste are disconnected:** the mood table overrides what the system knows about the user. A viewer who only watches slow dramas and selects \"Excited\" is shown generic action films they are unlikely to enjoy.",
  "**Mood is binary:** mood is either applied or not; there is no way to ask for a little more excitement.",
  "**Explanations are not grounded:** template sentences are not computed from the model, so they can be generic or misleading.")
P("Table 1.1 summarises how the proposed approach compares with the existing system on six practical criteria.")
table_("1.1", "Comparison of Existing and Proposed Approaches",
       ["Criterion", "TF-IDF + Genre (v1)", "Collaborative Filtering", "Proposed MovieMind v2"],
       [["Understands synonyms / meaning", "No (word overlap)", "Not applicable", "Yes (sentence embeddings)"],
        ["User representation", "Genre-weight dictionary", "Row of ratings / latent factors", "384-d centroid in movie space"],
        ["Mood handling", "Genre lookup table", "Usually none", "Direction vector in same space"],
        ["Mood intensity control", "None (on / off)", "None", "Continuous λ ∈ [0, 1]"],
        ["Cold-start for new movies", "Works (content)", "Fails (no ratings)", "Works (content)"],
        ["Explanation basis", "Fixed templates", "Opaque factors", "Geometric: taste, mood, nearest liked"]],
       [2200, 2100, 2200, 2500])

sec("1.6  Literature Review")
P("A review of prior published work was carried out to position MovieMind relative to existing research on semantic text embeddings, content-based and embedding-based recommendation, affective (emotion- and mood-aware) recommendation, and evaluation methodology. Only sources whose existence and bibliographic details were confirmed against a publisher, repository or archive page are cited; the full list is in Chapter 9.")
lit = [
    ("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", "N. Reimers and I. Gurevych, EMNLP-IJCNLP, 2019",
     "The paper fine-tunes BERT in a siamese setting so that whole sentences are mapped to vectors that can be compared directly with cosine similarity. It reports that finding the most similar pair among 10,000 sentences drops from about 65 hours with raw BERT to about 5 seconds. This is the basis for MovieMind's shared embedding space and its cosine nearest-neighbour search [1]."),
    ("MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers", "W. Wang et al., arXiv:2002.10957, 2020",
     "MiniLM compresses large Transformer models by distilling the self-attention behaviour of the final layer into a much smaller student. The `all-MiniLM-L6-v2` encoder used in this project builds on this compact architecture, which is why encoding a catalogue of several thousand movies is practical on a CPU [2]."),
    ("The MovieLens Datasets: History and Context", "F. M. Harper and J. A. Konstan, ACM TiiS, 2015",
     "This paper describes the origin of the MovieLens datasets, how they are released and the limitations that apply when they are used for research. It justifies the use of MovieLens for the offline evaluation in this project and informs the limitations discussed in Chapter 7 [3]."),
    ("Content-based Recommender Systems: State of the Art and Trends", "P. Lops, M. de Gemmis and G. Semeraro, Recommender Systems Handbook, 2011",
     "The chapter surveys how items and user profiles are represented in content-based systems and how profiles are learned from item descriptions. It supports both the v1 TF-IDF baseline and the idea of building a user profile from the content of liked items [4]."),
    ("Item2Vec: Neural Item Embedding for Collaborative Filtering", "O. Barkan and N. Koenigstein, IEEE MLSP, 2016",
     "Item2Vec adapts the word2vec skip-gram model to learn item embeddings from sets of co-occurring items and reports results competitive with SVD-based collaborative filtering. It supports the premise that items can be represented as points in a vector space where distance reflects similarity [5]."),
    ("Neural Collaborative Filtering", "X. He et al., WWW, 2017",
     "The paper replaces the inner product of matrix factorisation with a learned neural interaction function over user and item embeddings. MovieMind differs by using a fixed, content-derived embedding and a simple centroid for the user, trading some accuracy potential for transparency and cold-start robustness [6]."),
    ("A Survey of Affective Recommender Systems: Modeling Attitudes, Emotions, and Moods for Personalization", "T. Hasan and R. Bunescu, arXiv:2508.20289, 2025",
     "This survey classifies affect-aware recommenders into attitude-, emotion-, mood- and hybrid categories, and discusses how affect is extracted and used. It is the main source used to situate MovieMind's mood-aware design and to frame the gap addressed here: affect is usually handled as a separate signal rather than as a direction in the same space as the content. It is an arXiv preprint [7]."),
    ("Cumulated Gain-Based Evaluation of IR Techniques", "K. Järvelin and J. Kekäläinen, ACM TOIS, 2002",
     "The paper introduces cumulated gain, discounted cumulated gain and their normalised form (nDCG), which reward placing relevant items near the top of a ranking. nDCG@10 is one of the main metrics used in the evaluation plan [8]."),
    ("Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems", "M. Kaminskas and D. Bridge, ACM TiiS, 2016",
     "The authors survey beyond-accuracy objectives and show how optimising one affects the others. This motivates reporting Diversity@10 and Serendipity@10 alongside accuracy, and suggests that increasing λ may trade some accuracy for novelty [9]."),
    ("Linguistic Regularities in Continuous Space Word Representations", "T. Mikolov, W.-t. Yih and G. Zweig, NAACL-HLT, 2013",
     "The paper shows that semantic relations appear as consistent vector offsets in learned word spaces (the well-known king − man + woman ≈ queen example). This is the direct precedent for treating a mood as an offset, mean(mood exemplars) − mean(all movies), and adding it to a taste vector [10]."),
    ("Efficient Estimation of Word Representations in Vector Space", "T. Mikolov et al., arXiv:1301.3781, 2013",
     "The paper introduces the CBOW and skip-gram models for learning word vectors efficiently from large corpora, and underlies the idea that meaning can be encoded geometrically [11]."),
    ("Scikit-learn: Machine Learning in Python", "F. Pedregosa et al., JMLR, 2011",
     "The paper describes the scikit-learn library, from which the TF-IDF vectoriser and cosine-similarity routines used by the baseline system are taken [12]."),
]
for t, v, s in lit:
    add(h3(t), p_italic(v), p_body(s))
P("Table 1.2 consolidates the literature survey and shows where each source is applied in the project.")
table_("1.2", "Summary of Literature Survey",
       ["Ref.", "Contribution", "Applied To"],
       [["[1]", "Sentence embeddings comparable by cosine", "Movie encoding, nearest-neighbour search"],
        ["[2]", "Compact distilled Transformer encoder", "all-MiniLM-L6-v2 model choice"],
        ["[3]", "MovieLens dataset history and limits", "Evaluation dataset, limitations"],
        ["[4]", "Content-based profiles from item text", "v1 baseline, user centroid"],
        ["[5]", "Item embeddings for recommendation", "Items as points in a vector space"],
        ["[6]", "Neural user / item embeddings", "Contrast with centroid user model"],
        ["[7]", "Survey of affective recommenders", "Research gap, mood modelling"],
        ["[8]", "nDCG ranking metric", "NDCG@10"],
        ["[9]", "Diversity and serendipity metrics", "Diversity@10, Serendipity@10"],
        ["[10]", "Vector offsets encode relations", "Mood direction vector"],
        ["[11]", "Word-vector learning", "Embedding-space intuition"],
        ["[12]", "scikit-learn library", "TF-IDF baseline, cosine similarity"]],
       [900, 4300, 3800], align=["center", "left", "left"])

sec("1.7  Proposed System")
B("MovieMind v2 encodes every movie into a 384-dimensional vector using the `all-MiniLM-L6-v2` Sentence Transformer applied to a rich text built from plot, tagline, genres, keywords, director and cast. The vectors are computed once, offline, and stored as a NumPy matrix.",
  "A user's taste is the unit centroid of the embeddings of liked movies, stored as 384 numbers in a new `users.taste_embedding` column (the original genre profile is kept so that v1 can run as a baseline). Unliking a movie recomputes the centroid from the remaining likes.",
  "Ten mood direction vectors are computed once from the data: for each mood, the mean embedding of a set of exemplar movies minus the mean embedding of all movies, normalised to unit length.",
  "To recommend, the system forms search_position = user_taste + λ × mood_direction, computes the cosine similarity to every movie embedding, removes already-liked movies, applies a diversity filter and returns the top k.",
  "Each result carries a geometric explanation: taste contribution, mood contribution and the nearest liked movie.",
  "The user controls λ with a slider on the dashboard; the default is 0.3.")
P("Figure 1.1 illustrates the idea schematically: movies form neighbourhoods by experience rather than by genre label, the user sits in one neighbourhood, and the mood moves the search point a controlled distance toward another region.")
figure_("1.1", "Schematic Semantic Map of Movies, User Taste and Mood (illustrative)", "fig_4_2_space.png")

sec("1.8  Proposed System Advantages")
B("**Semantic understanding:** related concepts are close in the space even when they share no words.",
  "**Mood that respects identity:** the mood shifts the taste vector rather than replacing it, so results stay in the user's neighbourhood.",
  "**Controllable intensity:** λ gives both the user and the evaluator a continuous control from pure taste to pure mood.",
  "**Explainable by construction:** explanations are computed from distances in the same space the recommendation was made in.",
  "**Cold-start friendly:** a new user starts at the global mean embedding; new movies need only a description to be placed in the space.",
  "**Simple and light:** one recommendation path, one similarity function, one extra dependency (`sentence-transformers`), no GPU at query time and no paid API.")

sec("1.9  Hardware and Software Requirements")
P("Tables 1.3 and 1.4 list the minimum hardware and software needed to reproduce the project. No specialised hardware is required.")
table_("1.3", "Hardware Requirements",
       ["Component", "Specification", "Purpose"],
       [["Processor", "Any modern x86-64 / ARM CPU (4 cores recommended)", "Offline encoding, query-time vector maths"],
        ["Memory", "4 GB RAM minimum (8 GB recommended)", "Model load (~80 MB) and embedding matrices (~8 MB)"],
        ["Storage", "1 GB free disk", "Environment, model cache, SQLite database, datasets"],
        ["GPU", "Not required", "Encoding of ~5,000 movies is feasible on CPU"],
        ["Network", "Internet for first setup only", "Download dependencies and model; TMDB metadata fetch"]],
       [1800, 3700, 3500])
table_("1.4", "Software Requirements",
       ["Software / Library", "Version", "Purpose"],
       [["Python", "3.10+", "Implementation language"],
        ["Flask, Flask-SQLAlchemy, Flask-Login, Flask-CORS", "3.0+", "Web framework, ORM, sessions"],
        ["sentence-transformers", "3.0+", "all-MiniLM-L6-v2 encoder (new in v2)"],
        ["NumPy, SciPy, scikit-learn", "1.26+ / 1.12+ / 1.4+", "Vector arithmetic, cosine similarity, TF-IDF baseline"],
        ["pandas", "2.1+", "Data preparation and evaluation"],
        ["SQLite / SQLAlchemy", "SQLAlchemy 2.0+", "Persistence of movies, users, favourites"],
        ["Custom CSS + vanilla JavaScript", "—", "Front-end design system, mood list, λ ruler, vibe map"],
        ["pytest, pytest-cov", "8.0+ / 4.1+", "Automated tests"],
        ["Git", "Any recent", "Version control"]],
       [3800, 1800, 3400])

sec("1.10  Non-Functional Requirements")
B(f"**Latency:** a recommendation request should complete in well under one second for a catalogue of about 5,000 movies, because it is a single matrix–vector product. Measured end-to-end through Flask (100 requests, 5 liked movies) it takes a median of {LAT['api_median_ms']} ms and 95% finish within {LAT['api_p95_ms']} ms (Chapter 7).",
  "**Reproducibility:** embeddings and mood vectors are generated by a script (`build_embeddings.py`) so that the whole pipeline can be rebuilt from the database.",
  "**Local operation:** no cloud service or paid API is needed at any stage.",
  "**Maintainability:** the recommendation logic is isolated in `app/ml/`; routes only call it.",
  "**Security:** passwords are stored as hashes and all personal endpoints require an authenticated session.",
  "**Transparency:** every recommendation must be accompanied by a reason that is derived from the model, not invented.")

sec("1.11  Comparison with Alternative Approaches")
P("Several alternatives to a shared embedding space were considered during design. Table 1.5 compares them on the factors that mattered for this project.")
table_("1.5", "Comparison with Alternative Recommendation Approaches",
       ["Approach", "Needs ratings?", "Mood support", "Explainability", "Suitability"],
       [["TF-IDF + genres (v1)", "No", "Genre lookup", "Template", "Baseline only"],
        ["Matrix factorisation / CF", "Yes (many)", "None natively", "Low", "Weak cold-start; no mood"],
        ["Neural CF [6]", "Yes (many)", "Needs extra model", "Low", "Heavier; needs training data"],
        ["LLM-based recommender", "No", "Via prompt", "Free text", "Needs paid or large model; hard to control"],
        ["Shared embedding space (chosen)", "No (content only)", "Direction vector + λ", "Geometric", "Best fit for a small, local, explainable system"]],
       [2300, 1400, 1700, 1500, 2100])
P("A shared content embedding was chosen because it works with no ratings, treats mood and taste in one geometry, and is cheap to run. Collaborative signals can be added later as a second embedding (Section 8.3).")

# =========================================================== CHAPTER 2
chapter(2, "SYSTEM DESIGN")
sec("2.1  General Description")
P("MovieMind is a web application with an offline machine-learning preparation stage and an online recommendation stage. During preparation, movie metadata is turned into embeddings and mood direction vectors. During operation, the user interacts with a browser interface; the Flask back end updates the user's taste vector when movies are liked and, on request, computes mood-shifted recommendations from the stored matrices.")
P("This chapter describes the design through complementary views: a block diagram of the end-to-end pipeline, a data-flow diagram that separates the offline and online paths, a use-case diagram of the actors, an activity diagram of the recommendation algorithm, and UML collaboration and class diagrams of the software objects.")
sec("2.2  Prototype / Block Diagram")
P("Figure 2.1 shows the pipeline from movie metadata to recommendations. Movie metadata is encoded by the Sentence Transformer into an embedding matrix. The recommendation engine combines that matrix with a user taste centroid and a mood direction vector to produce the search position, and the Flask API returns the nearest movies with explanations.")
figure_("2.1", "System Block Diagram", "fig_2_1_block.png")
sec("2.3  System Architecture (Data Flow Diagram)")
P("Figure 2.2 expands the block diagram into a data-flow view. The offline pipeline reads the movie database, builds a rich text per movie, encodes it and stores two matrices: `movie_embeddings.npy` (N × 384) and `mood_vectors.npy` (10 × 384). The online pipeline updates the taste centroid when a user likes a movie, and on a recommendation request loads the stored matrices, forms the shifted search position, scores all movies by cosine similarity, filters and ranks them, and attaches explanations.")
figure_("2.2", "Data Flow / System Architecture Diagram", "fig_2_2_dfd.png")
P("The offline stage runs once per catalogue change. The online stage performs no model inference: it is vector arithmetic over matrices already held in memory, which is why latency stays low.")
sec("2.4  Use Case Diagram")
P("Figure 2.3 identifies two actors. The **User** signs up and logs in, browses and searches movies, likes or unlikes movies, selects a mood, adjusts the λ slider, views recommendations and views the explanation for each. Selecting a mood and adjusting λ extend the recommendation view, and viewing an explanation is part of viewing recommendations. The **Administrator** manages the movie catalogue and rebuilds the embeddings and mood vectors when the catalogue changes.")
figure_("2.3", "Use Case Diagram", "fig_2_3_usecase.png")
sec("2.5  Activity Diagram (Algorithm Flowchart)")
P("Figure 2.4 shows the algorithm executed for each recommendation request. The taste vector is loaded (or the global mean is used for a new user). If a mood is selected and λ is greater than zero, the search position is the taste vector plus λ times the mood direction; otherwise it is the taste vector alone. Similarities to all movies are computed, already-liked movies are removed, the list is sorted, a diversity filter is applied and explanations are computed for the top k.")
figure_("2.4", "Activity Diagram - Mood-Shifted Recommendation", "fig_2_4_activity.png", 5.3)
sec("2.6  UML Collaboration Diagram")
P("Figure 2.5 shows the objects that collaborate on one request. The browser sends the request to the recommendation route (message 1). The route loads the user and taste vector from the database (2) and calls the recommendation engine (3). The engine reads the movie embeddings and mood vector from the embedding store (4) and asks the explainer for contributions (5, 6). The ranked movies with explanations return to the route (7), which sends JSON to the browser (8). Liking or unliking a movie separately triggers the taste updater.")
figure_("2.5", "UML Collaboration Diagram", "fig_2_5_collab.png")
sec("2.7  UML Class Diagram")
P("Figure 2.6 shows the key classes. `User`, `Movie`, `UserFavorite` and `UserInteraction` are the persistent SQLAlchemy models; `User.taste_embedding` is a new column holding 384 floats as JSON; the v1 genre dictionary in `taste_vector` is kept for the baseline. `EmbeddingStore` loads the two NumPy matrices and exposes embeddings and mood directions. `TasteUpdater` maintains the centroid. `RecommendationEngine` performs the λ-shifted search and the diversity filter, and `RecommendationExplainer` computes the geometric explanation. `build_embeddings.py` is the offline script.")
figure_("2.6", "UML Class Diagram", "fig_2_6_class.png")
sec("2.8  Database Design")
P("The relational schema of v1 is retained. One column is added, `users.taste_embedding`, holding a JSON list of 384 floating-point numbers; a small start-up check adds it to databases created before v2. Table 2.1 lists the tables.")
table_("2.1", "Database Tables",
       ["Table", "Key columns", "Notes"],
       [["movies", "movie_id (PK), tmdb_id, title, release_year, overview, tagline, genres, cast, keywords, director, vote_average, popularity, poster_path", "Metadata from TMDB; genres / cast / keywords stored as JSON text"],
        ["users", "user_id (PK), username, email, password_hash, preferred_genres, taste_vector, taste_embedding", "taste_vector: v1 genre profile; taste_embedding: JSON list of 384 floats (v2)"],
        ["user_favorites", "id (PK), user_id (FK), movie_id (FK), added_at", "One row per liked movie; basis of the centroid"],
        ["user_interactions", "id (PK), user_id (FK), movie_id (FK), interaction_type, rating, timestamp", "view / like / unlike / rate events"]],
       [1700, 4600, 2700])
sec("2.9  API Design")
P("Existing endpoints are retained and the recommendation endpoint gains optional `lambda` and `mood` parameters. Table 2.2 lists the REST endpoints.")
table_("2.2", "REST API Endpoints",
       ["Endpoint", "Method", "Description"],
       [["/api/auth/signup, /login, /logout, /check", "POST / GET", "Registration, authentication, session check"],
        ["/api/recommend?mood=&lambda=&limit=", "GET", "Personalised recommendations with geometric explanations (λ new in v2; 400 if λ is outside 0–1 or the mood is unknown)"],
        ["/api/similar/<id>", "GET", "Movies nearest to a given movie in embedding space"],
        ["/api/movies, /api/movies/<id>", "GET", "Browse, search, details"],
        ["/api/moods, /api/genres", "GET", "Mood options and genre list"],
        ["/api/user/favorites", "GET / POST / DELETE", "List, add (like), remove (unlike) favourites"],
        ["/api/user/taste", "GET", "Current taste vector summary"],
        ["/api/user/taste/reset", "POST", "Reset taste vector"],
        ["/api/user/profile, /history", "GET / PUT", "Profile and interaction history"]],
       [3600, 1500, 3900])

# =========================================================== CHAPTER 3
chapter(3, "SYSTEM IMPLEMENTATION")
sec("3.1  Module Description")
P("The system is organised into six cooperating modules. This decomposition keeps the code maintainable, allows each module to be tested independently and makes it easy to change one part, for example swapping the embedding model, without touching the rest.")
h = lambda t: add(h3(t))
h("3.1.1  Data Preparation Module")
P("This module loads movie records from the SQLite database (populated from TMDB metadata and the MovieLens identifiers) and builds one text string per movie by concatenating the overview, tagline, genres (repeated to give them weight), keywords, director and top cast. Movies without any description are skipped because they would be placed at an arbitrary position in the space. The catalogue used in this project contains 4,988 enriched movies.")
h("3.1.2  Embedding Generation Module")
P("Implemented by `scripts/build_embeddings.py`, this module loads `all-MiniLM-L6-v2`, encodes every movie text in batches and L2-normalises the output so that cosine similarity reduces to a dot product. The result is saved as `movie_embeddings.npy` together with a mapping from movie id to row index. Encoding is a one-time offline step.")
h("3.1.3  Taste Module")
P("This module replaces the genre-based exponential moving average. When a user likes a movie, the taste vector is recomputed as the mean of the embeddings of all liked movies and saved to `users.taste_embedding`. When a movie is unliked, the centroid is recomputed from scratch from the remaining likes rather than by subtracting a vector, which avoids accumulated error. For a user with no likes the global mean embedding is used (cold start). Before the mood shift, the centroid is normalised to unit length so that λ has the same meaning for every user.")
h("3.1.4  Mood Direction Module")
P("For each of the ten moods the module selects exemplar movies using keyword and metadata rules (for example, for \"excited\": fast-paced, action-packed, high-stakes, adrenaline, relentless) and computes the direction as mean(exemplar embeddings) − mean(all embeddings), normalised to unit length. The ten vectors are stored in `mood_vectors.npy`. Because the directions are learned from the catalogue, they can be recomputed whenever the catalogue changes.")
P("Table 3.1 gives the number of exemplar movies found for each mood in the 4,988-movie catalogue; every mood has several hundred, and the script refuses to build a mood with fewer than 30.")
table_("3.1", "Exemplar Movies per Mood", ["Mood", "Exemplars", "Mood", "Exemplars"],
       [[m.capitalize(), META["exemplar_counts"][m], m2.capitalize(), META["exemplar_counts"][m2]]
        for m, m2 in zip(META["moods"][:5], META["moods"][5:])], [2000, 2000, 2000, 2000], align=["left", "center", "left", "center"])
P("Figure 3.1 shows how a mood direction is obtained from the global mean and the mean of the mood's exemplars.")
figure_("3.1", "Computing a Mood Direction Vector (schematic)", "fig_4_3_mooddir.png")
h("3.1.5  Recommendation Engine Module")
P("Given a user, an optional mood, λ and k, the engine forms the search position, computes cosine similarity against all movie embeddings, excludes liked movies, sorts and applies a diversity filter. The filter limits how many results may come from the same region of the space (at most max(2, k/3) per cluster, where clusters are obtained offline by k-means over the movie embeddings). Complexity is O(N × 384) per request.")
P("Figure 3.2 shows how increasing λ moves the search position from the user's taste toward the mood region.")
figure_("3.2", "Effect of λ on the Search Position (schematic)", "fig_4_4_lambda.png")
h("3.1.6  Explanation Module")
P("For each recommended movie m the module computes: **taste contribution** = cos(user_taste, m); **mood contribution** = cos(search_position, m) − cos(user_taste, m); and **nearest liked movie** = the liked movie with the highest cosine similarity to m. It also reports how much the mood shift moved the movie in the ranking. These values are rendered as a sentence in the interface.")
sec("3.2  Module Interaction Summary")
P("Table 3.2 consolidates the modules, their responsibilities and the modules they exchange data with.")
table_("3.2", "Module Interaction Summary",
       ["Module", "Primary Responsibility", "Interacts With"],
       [["Data Preparation", "Build rich text per movie", "Embedding Generation"],
        ["Embedding Generation", "Encode text to 384-d vectors, save matrix", "Data Preparation, Mood Direction"],
        ["Taste", "Maintain user centroid", "Recommendation Engine, Database"],
        ["Mood Direction", "Compute and store mood vectors", "Embedding Generation, Recommendation Engine"],
        ["Recommendation Engine", "λ-shifted search, filtering, ranking", "Taste, Mood Direction, Explanation"],
        ["Explanation", "Geometric reasons for each result", "Recommendation Engine"]],
       [2200, 3800, 3000])
sec("3.3  Testing Strategy")
P("Testing follows a bottom-up strategy. Unit tests check each module in isolation (shape and normalisation of embeddings, correctness of the centroid and its recomputation on unlike, the identity λ = 0 ⇒ pure taste, exclusion of liked movies, the diversity cap). Integration tests then call the REST API end to end. The v1 project already contained 55 automated tests (`test_api.py`, `test_models.py`, `test_ml.py`); v2 adds 18 in `tests/test_embeddings.py`, so the suite now has 73 tests, all of which pass. During this work an existing test-isolation fault was also found and fixed: the fixtures changed the database URL after the application had been created, so the tests ran against the real database and `db.drop_all()` would have erased it; `conftest.py` now forces an in-memory database. Table 3.3 lists the principal v2 test cases with their expected and observed results.")
table_("3.3", "Test Cases and Results",
       ["Test ID", "Test Case", "Expected Result", "Observed"],
       [["TC-01", "Embedding matrix shape and norm", "N × 384, every row unit length", "Pass (4,988 × 384)"],
        ["TC-02", "Synonym similarity (\"bank heist\" vs \"robbery\")", "Cosine clearly higher than TF-IDF (which is ~0)", "Pass"],
        ["TC-03", "New user, no likes", "Valid recommendations (cold start)", "Pass"],
        ["TC-04", "One like", "Taste vector equals that movie's embedding", "Pass"],
        ["TC-05", "Unlike a movie", "Centroid equals mean of remaining likes", "Pass"],
        ["TC-06", "λ = 0 with a mood selected", "Ranking identical to no-mood ranking", "Pass (also identical in the full evaluation)"],
        ["TC-07", "Liked movies in results", "None returned", "Pass"],
        ["TC-08", "Diversity cap", "No cluster exceeds max(2, k/3)", "Pass"],
        ["TC-09", "Mood shift direction", "Mood-aligned movies rise in rank as λ increases", "Pass"],
        ["TC-10", "λ outside [0, 1]", "HTTP 400 with clear message", "Pass (also for unknown mood)"],
        ["TC-11", "Explanation fields", "taste, mood contribution and nearest liked present", "Pass"],
        ["TC-12", "Request latency (5,000 movies)", "Well under 1 s on CPU", f"Pass: {LAT['api_median_ms']} ms median (API)"]],
       [900, 3000, 3300, 1800], align=["center", "left", "left", "left"], sz=20)
sec("3.4  Deployment and Installation Procedure")
P("Table 3.4 gives the procedure to install and run the system on a new machine.")
table_("3.4", "Deployment and Installation Procedure",
       ["Step", "Action", "Notes"],
       [["1", "Clone the repository and create a virtual environment", "python -m venv venv; activate it"],
        ["2", "Install dependencies: pip install -r requirements.txt", "Includes sentence-transformers>=3.0.0"],
        ["3", "Copy .env.example to .env and set SECRET_KEY, TMDB_API_KEY", "TMDB key only needed to fetch metadata"],
        ["4", "python scripts/setup_database.py, then scripts/load_movies.py", "Creates tables and loads movies"],
        ["5", "python scripts/build_embeddings.py", "Downloads model once; writes movie_embeddings.npy and mood_vectors.npy"],
        ["6", "python run.py", "Starts Flask on port 5000"],
        ["7", "Open http://localhost:5000, sign up, like movies", "Dashboard shows mood selector and λ slider"]],
       [800, 4600, 3600], align=["center", "left", "left"])

# =========================================================== CHAPTER 4
chapter(4, "TECHNOLOGY STACK")
sec("4.1  Software Components")
P("MovieMind is a pure software project; no special hardware is used. Figure 4.1 shows the layered architecture of the technology stack, from the presentation layer down to the data sources.")
figure_("4.1", "Layered Software Architecture", "fig_4_5_layers.png")
h("4.1.1  Sentence Transformer (all-MiniLM-L6-v2)")
P("The encoder maps text to a 384-dimensional vector. It is a distilled model [2] with six Transformer layers, trained to produce sentence embeddings suitable for semantic similarity [1]. It was chosen because it is small (about 80 MB), runs on a CPU and gives a good speed-quality trade-off for semantic search. Table 4.1 lists its main properties.")
table_("4.1", "Embedding Model Properties",
       ["Property", "Value"],
       [["Model", "sentence-transformers/all-MiniLM-L6-v2"],
        ["Output dimension", "384"],
        ["Transformer layers", "6"],
        ["Approximate size", "~80 MB"],
        ["Hardware needed", "CPU is sufficient"],
        ["Query-time model use", "None (embeddings are pre-computed)"],
        ["Storage for 4,988 movies", "4,988 × 384 × 4 bytes ≈ 7.66 MB"]],
       [3000, 6000])
P("Figure 4.2 shows the encoding pipeline.")
figure_("4.2", "Movie Encoding Pipeline", "fig_4_1_encoding.png")
h("4.1.2  NumPy and scikit-learn")
P("NumPy performs the vector arithmetic: centroids, mean differences, normalisation and the matrix–vector product that yields all cosine similarities. scikit-learn provides the TF-IDF vectoriser and cosine-similarity utilities used by the v1 baseline and the k-means clustering used by the diversity filter [12].")
h("4.1.3  Flask, SQLAlchemy and SQLite")
P("Flask serves the REST API and HTML pages and Flask-Login manages sessions. SQLAlchemy maps Python classes to SQLite tables. SQLite needs no server and suits a single-machine project; the same models can be pointed at another database through the `DATABASE_URL` setting.")
h("4.1.4  Front End")
P("The interface is a purpose-built design system written in plain CSS and vanilla JavaScript; Bootstrap and stock icon fonts were removed. It uses a warm paper-and-ink palette with one accent pair that has a fixed meaning (teal for the user’s taste, vermilion for mood), a serif display face (Fraunces), a grotesk body face (Instrument Sans) and a monospaced face (JetBrains Mono) for numbers, hairline rules instead of shadows and gradients, an 8-px spacing scale and three corner radii. Icons are a small set of custom inline SVG shapes. The dashboard is built around two controls: a typographic mood list and a λ **ruler** (0 to 1, step 0.05, default 0.3) beside a small “vibe map” that shows the search position moving from the user’s taste toward the chosen mood as the ruler moves. Each recommendation card shows its explanation sentence and a two-part bar (teal = taste match, vermilion = what the mood added). Text meets a minimum 5.0:1 contrast ratio, controls are keyboard-operable, and the layout reflows to a phone width.")
h("4.1.5  Data Sources")
P("Movie metadata (overview, tagline, cast, keywords, director, poster) comes from the TMDB API [14]. Ratings used for evaluation come from MovieLens [3]. The MovieLens `ml-latest-small` release included in the project contains 100,836 ratings by 610 users on 9,742 movies; 4,988 of these movies have enriched TMDB metadata and form the recommendation catalogue.")
sec("4.2  Software Tools")
table_("4.2", "Tools Used in the Project",
       ["Tool", "Role in Project"],
       [["Python 3.10+ / venv", "Implementation and environment isolation"],
        ["sentence-transformers", "Loading all-MiniLM-L6-v2 and encoding text"],
        ["VS Code / Claude Code", "Editing and code assistance"],
        ["pytest", "Unit and integration tests"],
        ["Jupyter", "Exploratory analysis and evaluation notebooks"],
        ["Git", "Version control"],
        ["draw.io / Pillow", "Diagrams used in this report"]],
       [3000, 6000])
sec("4.3  Communication Protocol Stack")
P("The browser and server communicate over HTTP using JSON for API calls. The back end calls TMDB over HTTPS only during data loading. Inside the server, calls between modules are ordinary Python function calls; the large matrices are loaded once at start-up and reused. Table 4.3 summarises the layers.")
table_("4.3", "Communication Layers and Measured Latency",
       ["Layer", "Medium", "Protocol / Format", "Measured"],
       [["Browser → Flask", "Local network / localhost", "HTTP, JSON, session cookie", "Not separately measured"],
        ["Flask → recommender", "In-process", "Python call", "Included below"],
        ["Recommender compute", "In-memory NumPy", "Matrix–vector product, sort, diversity", f"{R['latency_compute']['median_ms']:.2f} ms median, {R['latency_compute']['p95_ms']:.2f} ms p95"],
        ["Whole /api/recommend request", "Flask test client", "Auth, SQLite reads, ranking, explanations, JSON", f"{LAT['api_median_ms']} ms median, {LAT['api_p95_ms']} ms p95"],
        ["Loader → TMDB", "Internet", "HTTPS REST (offline step)", "Not on request path"]],
       [2200, 2000, 2800, 2000])
sec("4.4  Resource and Cost Estimation")
P("Every component is open source and runs locally, so the monetary cost is zero. Table 4.4 estimates the computing resources.")
table_("4.4", "Resource Estimate",
       ["Item", "Estimate", "Basis"],
       [["Embedding matrix", f"{META['embedding_mb']} MB", "4,988 × 384 float32 (measured)"],
        ["Mood vectors", f"{META['mood_vectors_kb']} KB", "10 × 384 float32 (measured)"],
        ["Application memory", f"{int(LAT['rss_mb_peak'])} MB peak", "Flask + NumPy + scikit-learn with matrices loaded (measured)"],
        ["Per-user taste vector", "≈ 3 KB as JSON", "384 floats"],
        ["Encoding model on disk", "≈ 80 MB", "Model card"],
        ["One-time encoding time", f"{META['encode_seconds']} s", "CPU only, 4,988 movies (measured)"],
        ["Per-request compute", "O(N × 384) ≈ 1.9 M multiply-adds", "N = 4,988"],
        ["Licence / API cost", "None", "Open-source components; TMDB free tier for metadata"]],
       [2800, 2800, 3400])
sec("4.5  Comparison with Alternative Embedding Choices")
table_("4.5", "Comparison with Alternative Representations",
       ["Representation", "Dimension", "Strength", "Drawback", "Verdict"],
       [["TF-IDF", "Thousands (sparse)", "Fast, simple", "No synonyms, no semantics", "Baseline (v1)"],
        ["all-MiniLM-L6-v2 (chosen)", "384", "Small, CPU-friendly, good quality", "English-focused", "Best fit"],
        ["Larger Sentence Transformer", "768 and above", "Possibly richer", "Larger, slower encoding", "Possible future upgrade"],
        ["Hosted embedding API", "Varies", "Strong quality", "Paid, needs network, privacy concerns", "Rejected"]],
       [2300, 1500, 2000, 2000, 1200], sz=20)
sec("4.6  Privacy, Security and Reliability")
P("Passwords are stored only as salted hashes. All user-specific endpoints require login. The only personal data stored is the username, e-mail and the list of liked movies; the taste vector is a numerical summary of those likes. Because the model and data run locally, no viewing data leaves the machine. For reliability, the matrices are loaded once and validated at start-up (row count must equal the number of encoded movies), and the application falls back to popularity-ranked results if the embeddings are missing.")
sec("4.7  Scalability")
P("A full scan costs O(N × 384) per request, which is trivial for thousands of movies and still practical for hundreds of thousands. Beyond that, an approximate nearest-neighbour index (for example FAISS or HNSW) can replace the full scan without changing the λ-shifted search, because only the query vector changes.")
sec("4.8  Overall Technical Justification")
P("The stack was chosen to meet four needs: semantic quality (Sentence Transformer), transparency (vector geometry instead of a learned black box), low cost (local, free components) and ease of extension (single recommendation path). Each of these also supports the evaluation, because every design choice can be switched off, for example λ = 0 for \"no mood\" or TF-IDF for \"no embeddings\", to isolate its effect.")

# =========================================================== CHAPTER 5
chapter(5, "SCREENSHOTS")
sec("5.1  System Configuration")
P("Table 5.1 records the configuration used for the v2 system, so that the screenshots and results that follow can be interpreted precisely.")
table_("5.1", "System Configuration Parameters",
       ["Parameter", "Value"],
       [["Embedding model", "all-MiniLM-L6-v2 (384-d)"],
        ["Catalogue size", "4,988 movies"],
        ["Moods", "10 (happy, sad, excited, relaxed, romantic, thrilled, nostalgic, adventurous, thoughtful, scared)"],
        ["Default λ", "0.3 (range 0.0 to 1.0)"],
        ["Recommendations per request (k)", "10"],
        ["Diversity cap", "max(2, k/3) per cluster"],
        ["Similarity", "Cosine"],
        ["Backend / DB", "Flask 3, SQLite"]],
       [3200, 5800])
sec("5.2  Application Screenshots")
P("[[The three images below are placeholders for screenshots of the running application; replace them with captures taken from the final v2 build.]]")
figure_("5.1", "Login and Sign-up Page [placeholder]", "ph_5_1_login.png", 5.8)
figure_("5.2", "Dashboard with Mood Selector and λ Slider [placeholder]", "ph_5_2_dashboard.png", 5.8)
figure_("5.3", "Recommendations with Geometric Explanations [placeholder]", "ph_5_3_recs.png", 5.8)
sec("5.3  Console Output")
P("The offline build prints the shapes, timing and exemplar counts. The following is the real output of `python scripts/build_embeddings.py` on the project machine (CPU only):")
add(code_block("""Encoding 4988 movies with all-MiniLM-L6-v2 ...
  embeddings (4988, 384) float32 in 137.6s  (7.66 MB)
  mood exemplar counts: {"happy": 617, "sad": 403, "excited": 535, "relaxed": 481,
    "romantic": 420, "thrilled": 682, "nostalgic": 500, "adventurous": 345,
    "thoughtful": 501, "scared": 456}
  20 clusters; sizes min/max = 133/433
Saved to data/models"""))
sec("5.4  System Response and Validation")
P("The screenshots together show the behaviour verified during development:")
B("the dashboard accepts a mood and a λ value and re-ranks results when either changes;",
  "each recommendation shows its explanation (taste match, mood effect, nearest liked movie);",
  "setting λ to 0 produces the same list regardless of the selected mood;",
  "the offline script reports the matrix shapes and time taken.")

# =========================================================== CHAPTER 6
chapter(6, "SAMPLE CODING")
P("This chapter lists the core Python code of the v2 recommendation pipeline. The listings are extracted automatically from the project source files, so they are exactly the code that produced the results in Chapter 7. The code is organised in the same modules as Chapter 3.")
sec("6.1  Embeddings and Mood Vectors")
add(code_block(src("app/ml/embeddings.py", ["movie_text", "encode_texts", "mood_exemplar_rows", "compute_mood_vectors", "compute_clusters"], "app/ml/embeddings.py (offline functions)")))
sec("6.2  Embedding Store and Mood-Shifted Search")
add(code_block(src("app/ml/embeddings.py", ["unit", "centroid", "search_position", "scores"], "app/ml/embeddings.py (EmbeddingStore methods)")))
sec("6.3  User Taste Centroid")
add(code_block(src("app/ml/taste_updater.py", ["refresh_embedding_taste"], "app/ml/taste_updater.py")))
sec("6.4  Recommendation Engine")
add(code_block(src("app/ml/embedding_recommender.py", ["get_recommendations", "_diverse"], "app/ml/embedding_recommender.py (EmbeddingRecommender)")))
sec("6.5  Geometric Explanations")
add(code_block(src("app/ml/explainer.py", ["geometric_explanation"], "app/ml/explainer.py")))
sec("6.6  REST Endpoint")
add(code_block(src("app/routes/recommendation.py", ["get_recommendations"], "app/routes/recommendation.py")))
sec("6.7  API Quick Reference")
table_("6.1", "Recommendation Endpoint Parameters (liked movies are always excluded)",
       ["Parameter", "Type", "Default", "Meaning"],
       [["mood", "string", "none", "One of the ten moods; omitted means no mood shift; unknown mood returns 400"],
        ["lambda", "float 0.0–1.0", "0.3", "Strength of the mood shift; outside the range returns 400"],
        ["limit", "integer", "10", "Number of recommendations (1–50)"]],
       [1800, 1800, 1200, 4200])

# =========================================================== CHAPTER 7
chapter(7, "RESULT AND DISCUSSION")
D = R["data"]
sec("7.1  Experimental Setup")
P(f"The evaluation is an offline experiment on the MovieLens `ml-latest-small` dataset [3], restricted to the {D['catalogue']:,} catalogue movies that have enriched metadata. These carry {D['ratings_on_catalogue']:,} ratings from all {D['users_total']} users. For each user, ratings are ordered by time: the earliest 80% form the training history and the latest 20% the test set. A test movie is relevant if the user rated it 4.0 or higher, and a training movie counts as liked on the same rule. All movies the user rated in training are excluded from every system's candidate list. Users with fewer than {R['config']['min_train_liked']} liked training movies or no relevant test movie are skipped, which leaves **{D['users_evaluated']} of {D['users_total']} users**. Everything is deterministic (fixed random seed) and reproducible with `python scripts/evaluate.py`.")
P("Systems compared (top-10 lists): **POP** the movies most liked in the training data, as a reference; **B1** the v1 engine (TF-IDF and genre hybrid similarity, genre profile updated by exponential moving average, mood as a genre boost); **B2** semantic centroid with no mood (λ = 0); **B3** semantic centroid with mood applied as a v1-style genre boost; and **Proposed** the mood-shifted search with λ = 0.3.")
P(f"MovieLens contains no mood labels, so a proxy is needed. For each user the __proxy mood__ is the mood direction closest (by cosine) to the drift of their recent viewing: the mean embedding of their last {R['config']['recent_window']} liked training movies minus the mean embedding of all their liked training movies. It uses no test data. Two controls are reported: a __random mood__ for each user, and, as a clearly labelled **diagnostic that cannot be achieved in practice**, an __oracle mood__ chosen from the user's held-out relevant movies, which gives an upper bound on what a perfectly known mood could do. Metrics are HR@10, NDCG@10, Diversity@10 (one minus the mean pairwise cosine of the list) and Serendipity@10 (relevant hits whose genres miss the user's three most-watched genres, divided by 10). 95% confidence intervals come from {R['config']['bootstrap']:,} bootstrap resamples of users.")
sec("7.2  Result")
P(f"The headline result is negative for the proposed method under the proxy mood. At λ = 0.3 the mood-shifted search reaches HR@10 = {f3(PRO['hr'])} and NDCG@10 = {f3(PRO['ndcg'])}, against {f3(B2['hr'])} and {f3(B2['ndcg'])} for the same semantic search with no mood (B2). The paired difference over the same users is HR@10 {PAIR['hr']['diff']:+.3f} (95% CI {PAIR['hr']['lo']:+.3f} to {PAIR['hr']['hi']:+.3f}; {PAIR['hr']['users_better']} users better, {PAIR['hr']['users_worse']} worse, {PAIR['hr']['users_equal']} unchanged), so the loss is statistically distinguishable from zero. A random mood performs about as well as the proxy mood (HR@10 {f3(CTL['hr'])}), which shows the proxy carries little usable information about what the user goes on to like.")
P(f"Two further results stand out. First, the simple popularity reference (HR@10 = {f3(POP['hr'])}) is far above every content-based system, and the v1 engine (B1, {f3(B1['hr'])}) is above the semantic centroid (B2, {f3(B2['hr'])}). Second, when the mood is known exactly (oracle diagnostic) the mechanism works: HR@10 rises from {f3(SWEEP[(0.0,'oracle')]['hr'])} at λ = 0 to {f3(SWEEP[(0.5,'oracle')]['hr'])} at λ = 0.5 and {f3(SWEEP[(0.7,'oracle')]['hr'])} at λ = 0.7, with an interior optimum rather than a monotonic trend. The oracle uses the test items, so these values show that a correctly chosen direction helps, not that the system achieves them.")
sec("7.3  Performance Metrics")
rows = []
nm = {POP['system']: "POP: most liked in training (reference)", B1['system']: "B1: TF-IDF + genre (v1)", B2['system']: "B2: semantic centroid, no mood (λ = 0)",
      B3['system']: "B3: semantic centroid + genre-table mood", PRO['system']: "Proposed: mood-shifted, λ = 0.3, proxy mood", CTL['system']: "Control: λ = 0.3, random mood"}
for x in R["systems"]:
    rows.append([nm[x['system']], f"{f3(x['hr'])} ({f3(x['hr_lo'])}–{f3(x['hr_hi'])})", f"{f3(x['ndcg'])}", f3(x['div']), f3(x['seren'])])
table_("7.1", f"Recommendation Quality at k = 10 ({D['users_evaluated']} users; 95% bootstrap CI for HR)",
       ["Method", "HR@10 (95% CI)", "NDCG@10", "Diversity@10", "Serendipity@10"], rows, [3600, 2300, 1100, 1300, 1400], align=["left", "center", "center", "center", "center"], sz=20)
table_("7.2", "Paired Difference, Proposed (λ = 0.3, proxy mood) minus B2 (same users)",
       ["Metric", "Mean difference", "95% CI", "Users better / worse / equal"],
       [[k.upper() if k in ("hr",) else {"ndcg": "NDCG", "div": "Diversity", "seren": "Serendipity"}.get(k, k.upper()),
         f"{v['diff']:+.4f}", f"{v['lo']:+.4f} to {v['hi']:+.4f}", f"{v['users_better']} / {v['users_worse']} / {v['users_equal']}"] for k, v in PAIR.items()],
       [1800, 2000, 2800, 2400], align=["left", "center", "center", "center"])
sw_rows = []
for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 1.0):
    sw_rows.append([f"{lam:.1f}"] + [f3(SWEEP[(lam, src_)]['hr']) for src_ in ("proxy", "random", "oracle")] + [f3(SWEEP[(lam, 'proxy')]['div'])])
table_("7.3", "Effect of λ on HR@10 for Three Mood Sources (oracle uses test items; upper bound only)",
       ["λ", "Proxy mood", "Random mood", "Oracle mood", "Diversity (proxy)"], sw_rows, [1000, 2000, 2000, 2000, 2000], align=["center"] * 5)
Q = R["diagnostics"]["quality_prior"]
table_("7.4", "Diagnostics (not proposed methods): Quality Prior and Diversity Filter on the Semantic Centroid",
       ["Variant", "HR@10", "NDCG@10"],
       [[f"B2 + quality prior, weight {w}", f3(Q[w]['hr']), f3(Q[w]['ndcg'])] for w in ("0.0", "0.05", "0.15", "0.3")]
       + [["B2 without the diversity filter", f3(R['diagnostics']['no_diversity_filter']['hr']), f3(R['diagnostics']['no_diversity_filter']['ndcg'])]],
       [5000, 2000, 2000], align=["left", "center", "center"])
table_("7.5", "Latency and Resource Measurements",
       ["Metric", "Value", "How measured"],
       [["Recommender compute (median / p95)", f"{R['latency_compute']['median_ms']:.2f} / {R['latency_compute']['p95_ms']:.2f} ms", f"score + sort + diversity, {R['latency_compute']['n']} users"],
        ["Whole /api/recommend request (median / p95)", f"{LAT['api_median_ms']} / {LAT['api_p95_ms']} ms", f"Flask test client, {LAT['n']} requests, 5 likes, includes DB and explanations"],
        ["One-time encoding of the catalogue", f"{META['encode_seconds']} s", "build_embeddings.py, CPU only"],
        ["Application memory (peak)", f"{int(LAT['rss_mb_peak'])} MB", "Flask + NumPy + scikit-learn with matrices loaded"]],
       [3600, 2200, 3200])
figure_("7.1", "HR@10 and NDCG@10 versus λ for proxy, random and oracle moods (shaded: 95% CI)", "fig_7_1_lambda.png", 6.2)
figure_("7.2", "Comparison of Baselines and Proposed Method (95% bootstrap CI)", "fig_7_2_methods.png", 6.2)
sec("7.4  Discussion")
P("The four hypotheses stated in advance were checked against the measured results. Two of them are **not supported** and one is supported only in a restricted sense.")
B(f"**H1, semantic embeddings beat TF-IDF: not supported on accuracy.** The semantic centroid (B2, HR@10 {f3(B2['hr'])}) is below the v1 engine (B1, {f3(B1['hr'])}). The diagnostic in Table 7.4 explains why: v1 also scores each movie by its average rating, a popularity-like prior that correlates with whether a user later rates a film 4 or higher, whereas embeddings capture what a film is like, not how well it is regarded. Giving B2 the same kind of prior (weight 0.3, chosen after the fact and not tuned) raises it to {f3(Q['0.3']['hr'])}, above B1. Embeddings do match meaning better (Section 1.1, test TC-02), but meaning alone does not predict ratings.",
  f"**H2, a moderate mood shift does not hurt accuracy: not supported with the proxy mood.** HR@10 falls from {f3(SWEEP[(0.0,'proxy')]['hr'])} at λ = 0 to {f3(SWEEP[(1.0,'proxy')]['hr'])} at λ = 1 and the proxy mood is no better than a random one. Diversity rises only slightly ({f3(SWEEP[(0.0,'proxy')]['div'])} to {f3(SWEEP[(1.0,'proxy')]['div'])}).",
  f"**H3, mood as a direction beats mood as a genre filter: not demonstrated.** B3 ({f3(B3['hr'])}) and the proposed method ({f3(PRO['hr'])}) both sit below B2, and their intervals overlap, so no ordering can be claimed between them.",
  f"**H4, the best λ is interior: supported only for the oracle mood.** With the oracle mood HR@10 peaks at λ ≈ 0.5 to 0.7 ({f3(SWEEP[(0.5,'oracle')]['hr'])} to {f3(SWEEP[(0.7,'oracle')]['hr'])}) and falls again toward λ = 1; with the proxy and random moods there is no interior optimum.")
P("What the experiment does and does not show. The machinery behaves as designed: λ = 0 reproduces the no-mood ranking exactly, mood vectors move rankings toward their exemplars, and a correctly chosen mood improves held-out hit rate. What the offline data cannot show is that a mood __the user chooses__ improves their experience, because in this dataset the mood is unobserved and the proxy for it is weak. In the application the viewer states the mood and the strength, so the proxy problem is specific to offline evaluation; whether the shifted lists are preferred is a question for a user study, not for ratings accuracy, since a deliberate change of mood is by construction a move away from past taste. The popularity result is a reminder that content-only methods should be combined with a quality or collaborative signal before being judged on accuracy.")
sec("7.5  Limitations of the Current Prototype")
B("**Weak mood proxy:** the recent-likes drift did not outperform a random mood, so the evaluation cannot show the benefit of mood; explicit mood labels or session data are needed.",
  "**Subjective mood exemplars:** exemplar movies are chosen by TMDB keyword rules (several hundred per mood), which reflect the designer's judgement; crowd-sourced mood labels would be more reliable.",
  "**One mood at a time:** real emotional states are blends; multi-mood blending is future work.",
  "**No quality or collaborative signal in the recommender:** content embeddings alone trail a popularity reference and the v1 engine on hit rate (Table 7.1).",
  "**English-centred encoder:** the model is trained mainly on English text.",
  "**Offline, small dataset:** MovieLens `ml-latest-small` has 610 users, of whom 531 qualified; it is a development dataset and not suited to claims about large-scale performance [3]. No user study was run.",
  "**Post-hoc diagnostics:** the oracle mood and the quality-prior variants were added after the first results were seen. They are reported as explanations, not as tuned methods, and should be replicated on separate data.")

# =========================================================== CHAPTER 8
chapter(8, "CONCLUSION AND FUTURE ENHANCEMENT")
sec("8.1  Conclusion")
P("This project built MovieMind v2, a recommender in which movies, users and moods share one semantic embedding space and a recommendation is the set of movies nearest to unit(user_taste) + λ × mood_direction. The design replaces genre dictionaries and hard-coded mood tables with Sentence-Transformer embeddings, a centroid user model and data-derived mood directions, adds a user-controllable mood intensity, and produces explanations from the same geometry. The system was implemented end to end (offline pipeline, taste updates, recommender, API, a redesigned interface with a λ ruler) and verified by 73 automated tests; a request takes a median of " + str(LAT['api_median_ms']) + " ms on a CPU.")
P(f"The offline evaluation gave a mixed and honest picture. With a mood that is known exactly, the λ-shifted search improved the held-out hit rate from {f3(SWEEP[(0.0,'oracle')]['hr'])} to as much as {f3(SWEEP[(0.5,'oracle')]['hr'])}, so the mechanism is sound. With the only mood signal available in MovieLens, a proxy from recent likes, the shift did not help (HR@10 {f3(PRO['hr'])} against {f3(B2['hr'])} without mood), and content embeddings alone trailed a popularity reference ({f3(POP['hr'])}) and the v1 engine ({f3(B1['hr'])}), which carries a rating prior. The claims that are therefore supported are the engineering and design contributions: a unified space, mood as a continuous directional shift, a user-controllable λ and geometry-based explanations. The claim that mood modulation improves recommendation quality remains **untested** and needs real mood data or a user study.")
sec("8.2  Applications")
B("**Streaming and video-on-demand platforms:** a mood slider on the home screen that re-ranks a personalised list.",
  "**Digital libraries and cinemas:** content-based suggestions for catalogues with little rating data.",
  "**Education and research:** a transparent reference system for studying affect-aware recommendation.",
  "**Other media domains:** the same λ-shifted search can be applied to music, books or podcasts by changing the encoder and mood exemplars.")
sec("8.3  Future Enhancement")
fut = [("8.3.0  Quality and Collaborative Signals", "Combine the semantic search with a rating prior and a collaborative embedding. The diagnostics in Chapter 7 show that a quality term alone closes most of the gap to v1, so this is the most promising first step."),
       ("8.3.1  Multi-Mood Blending", "Allow blends such as 60% excited and 40% nostalgic, so that the search position is the taste vector plus a weighted sum of mood directions."),
       ("8.3.2  Learning λ per User", "Learn the best λ for each person from whether they like mood-shifted recommendations, instead of using one global default."),
       ("8.3.3  Hybrid Collaborative and Semantic Embeddings", "Combine the content centroid with a collaborative embedding learned from ratings, for example in a two-tower model."),
       ("8.3.4  Temporal Mood Decay", "Let λ fade gradually after a mood is selected, reflecting how a transient emotional state passes."),
       ("8.3.5  Multilingual Support", "Use a multilingual encoder so that non-English descriptions are represented properly."),
       ("8.3.6  Crowd-Sourced Mood Labels", "Replace keyword-based exemplars with labels collected from viewers, and recompute directions periodically."),
       ("8.3.7  User Study and Real Mood Data", "Collect explicit mood choices and satisfaction from real participants, which is the only way to test whether mood-shifted lists are preferred and to replace the weak offline proxy."),
       ("8.3.8  Cross-Domain Application", "Apply the framework to music, books or podcasts.")]
for t, s in fut:
    add(h3(t), p_body(s))
sec("8.4  Societal and Ethical Considerations")
P("**Privacy.** Because encoding and recommendation run locally and the stored taste vector is only a numeric summary of likes, the design is privacy-friendly compared with services that send viewing data to third parties. Passwords are stored as hashes and personal endpoints need authentication.")
P("**Filter bubbles.** A pure-taste recommender can keep showing similar content. λ and the diversity filter let the user choose how far to move from their usual neighbourhood, which gives the user control rather than hiding the trade-off.")
P("**Bias.** Embeddings inherit biases of their training text and of the catalogue, which over-represents popular and English-language films. Results should be audited before any real deployment.")
P("**Emotional inference.** The system uses a mood the user selects; it does not infer emotion from behaviour or sensors, which avoids intrusive affect detection.")
P("**Environmental cost.** All heavy computation is a one-time CPU encoding of a few thousand short texts; query-time cost is a single matrix product, so the energy footprint is small.")
sec("8.5  Deployment Considerations")
P("A production deployment would need an approximate nearest-neighbour index for large catalogues, incremental encoding of newly added movies, monitoring of recommendation quality and latency, and periodic recomputation of mood directions. A phased rollout, beginning with an opt-in mood slider and comparing engagement with the existing list, would show whether the approach helps real viewers before it is widened.")

# =========================================================== CHAPTER 9
chapter(9, "REFERENCES")
refs = [
    'N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in Proc. 2019 Conf. Empirical Methods in Natural Language Processing and 9th Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019, pp. 3982–3992, doi: 10.18653/v1/D19-1410.',
    'W. Wang, F. Wei, L. Dong, H. Bao, N. Yang and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," arXiv:2002.10957, 2020.',
    'F. M. Harper and J. A. Konstan, "The MovieLens datasets: History and context," ACM Trans. Interact. Intell. Syst., vol. 5, no. 4, Art. 19, Dec. 2015, doi: 10.1145/2827872.',
    'P. Lops, M. de Gemmis and G. Semeraro, "Content-based recommender systems: State of the art and trends," in Recommender Systems Handbook, F. Ricci, L. Rokach, B. Shapira and P. B. Kantor, Eds. New York, NY, USA: Springer, 2011, pp. 73–105, doi: 10.1007/978-0-387-85820-3_3.',
    'O. Barkan and N. Koenigstein, "Item2Vec: Neural item embedding for collaborative filtering," in Proc. IEEE 26th Int. Workshop on Machine Learning for Signal Processing (MLSP), Salerno, Italy, 2016. arXiv:1603.04259.',
    'X. He, L. Liao, H. Zhang, L. Nie, X. Hu and T.-S. Chua, "Neural collaborative filtering," in Proc. 26th Int. Conf. World Wide Web (WWW), Perth, Australia, 2017, pp. 173–182, doi: 10.1145/3038912.3052569.',
    'T. Hasan and R. Bunescu, "A survey of affective recommender systems: Modeling attitudes, emotions, and moods for personalization," arXiv:2508.20289 [cs.IR], Aug. 2025.',
    'K. Järvelin and J. Kekäläinen, "Cumulated gain-based evaluation of IR techniques," ACM Trans. Inf. Syst., vol. 20, no. 4, pp. 422–446, Oct. 2002, doi: 10.1145/582415.582418.',
    'M. Kaminskas and D. Bridge, "Diversity, serendipity, novelty, and coverage: A survey and empirical analysis of beyond-accuracy objectives in recommender systems," ACM Trans. Interact. Intell. Syst., vol. 7, no. 1, 2016, doi: 10.1145/2926720.',
    'T. Mikolov, W.-t. Yih and G. Zweig, "Linguistic regularities in continuous space word representations," in Proc. 2013 Conf. North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Atlanta, GA, USA, 2013, pp. 746–751.',
    'T. Mikolov, K. Chen, G. Corrado and J. Dean, "Efficient estimation of word representations in vector space," arXiv:1301.3781, 2013.',
    'F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.',
    'Pallets Projects, "Flask documentation." [Online]. Available: https://flask.palletsprojects.com',
    'The Movie Database (TMDB), "TMDB API documentation." [Online]. Available: https://www.themoviedb.org',
]
for i, r in enumerate(refs, 1):
    add(p_body(f"[{i}]  {r}", after=120))
add(p_empty())
add(h2("Patent / Paper Publication Proof"))
secs.append("Patent / Paper Publication Proof")
P("[[Insert proof of paper submission or publication here, if applicable. Remove this page otherwise.]]")
add(p_empty())
add(p_body("**GLOSSARY OF TERMS**", jc="center"))
gloss = [("Embedding", "A list of numbers (here 384) that represents the meaning of a text so that similar meanings are close together."),
         ("Sentence Transformer", "A neural network that maps a sentence or paragraph to a single embedding vector."),
         ("Cosine similarity", "A measure of how closely two vectors point in the same direction; 1 means identical direction."),
         ("Centroid", "The mean of a set of vectors; here, the average of liked-movie embeddings, used as the user's taste."),
         ("Mood direction", "A unit vector from the average movie toward the average of a mood's exemplar movies."),
         ("λ (lambda)", "The weight, between 0 and 1, of the mood direction when it is added to the taste vector."),
         ("TF-IDF", "A weighting of words by frequency in a document and rarity across documents; ignores word meaning."),
         ("Cold start", "The situation in which a new user or item has no history to base a recommendation on."),
         ("Hit Rate@K", "The fraction of users for whom at least one relevant item appears in the top K."),
         ("NDCG@K", "A ranking metric that rewards relevant items placed higher in the top K list."),
         ("Diversity", "How different the recommended items are from each other."),
         ("Serendipity", "How often relevant recommendations are also unexpected for the user.")]
for t, s in gloss:
    add(p_body(f"**{t}:**  {s}", after=120))

ABBR = [("NLP", "Natural Language Processing"), ("SBERT", "Sentence-BERT"), ("MiniLM", "Mini Language Model (distilled Transformer)"),
        ("TF-IDF", "Term Frequency – Inverse Document Frequency"), ("CF", "Collaborative Filtering"), ("HR", "Hit Rate"),
        ("NDCG", "Normalised Discounted Cumulative Gain"), ("TMDB", "The Movie Database"), ("API", "Application Programming Interface"),
        ("REST", "Representational State Transfer"), ("JSON", "JavaScript Object Notation"), ("HTTP", "Hypertext Transfer Protocol"),
        ("ORM", "Object-Relational Mapping"), ("SQL", "Structured Query Language"), ("UI", "User Interface"),
        ("CPU", "Central Processing Unit"), ("GPU", "Graphics Processing Unit"), ("ANN", "Approximate Nearest Neighbour"),
        ("UML", "Unified Modeling Language"), ("CSV", "Comma-Separated Values"), ("ML", "Machine Learning"),
        ("IDE", "Integrated Development Environment"), ("TC", "Test Case"), ("λ", "Lambda – mood-shift weight")]
