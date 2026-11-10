import re, time, math
from collections import Counter
from io import StringIO

import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from Bio import SeqIO, AlignIO
from Bio.Align import PairwiseAligner, substitution_matrices

EBI_EMAIL = "your_real_email@gmail.com"   # <-- put your real email here

BLUE_SCALE = ["#eff6ff", "#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
CHART_COLORS = ["#1d4ed8", "#06b6d4", "#3b82f6", "#0ea5e9", "#1e3a8a", "#38bdf8"]

st.set_page_config(page_title="OrthoQuest", page_icon="🧬", layout="wide")

# ======================= DESIGN =======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

/* ---- Page background ---- */
.stApp {
    font-family: 'Poppins', sans-serif;
    background:
        radial-gradient(circle at 8% 12%, rgba(59,130,246,0.18) 0, transparent 35%),
        radial-gradient(circle at 92% 18%, rgba(6,182,212,0.18) 0, transparent 30%),
        radial-gradient(circle at 70% 95%, rgba(30,58,138,0.12) 0, transparent 40%),
        #f0f7ff;
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
h1, h2, h3 { color: #1e3a8a !important; font-family: 'Poppins', sans-serif !important; }

/* ---- Hero banner (Home page) ---- */
.hero {
    position: relative; overflow: hidden; padding: 38px 40px; border-radius: 24px;
    background: linear-gradient(120deg, #0b1f4d, #1d4ed8, #0891b2, #1d4ed8, #0b1f4d);
    background-size: 300% 300%; animation: heroShift 14s ease infinite;
    box-shadow: 0 14px 40px rgba(29,78,216,0.35); margin-bottom: 26px;
}
@keyframes heroShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero::before, .hero::after {
    content: ""; position: absolute; border-radius: 50%; background: rgba(255,255,255,0.08);
}
.hero::before { width: 320px; height: 320px; right: -80px; top: -120px; }
.hero::after  { width: 200px; height: 200px; right: 180px; bottom: -110px; }
.hero-title { color: white; font-size: 2.8rem; font-weight: 700; letter-spacing: 0.5px; }
.hero-sub { color: #cffafe; font-size: 1.08rem; margin-top: 6px; max-width: 760px; }
.hero-tags { margin-top: 18px; }
.hero-tags span {
    display: inline-block; background: rgba(255,255,255,0.15); color: white;
    padding: 6px 14px; border-radius: 999px; margin: 4px 6px 0 0; font-size: 0.85rem;
    border: 1px solid rgba(255,255,255,0.25);
}
.dna { display: inline-block; animation: float 3s ease-in-out infinite; }
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

/* ---- Page banners (other pages) ---- */
.page-banner {
    display: flex; align-items: center; gap: 18px; padding: 22px 28px; border-radius: 20px;
    margin-bottom: 22px; position: relative; overflow: hidden;
    background: linear-gradient(110deg, #1e3a8a, #2563eb 60%, #06b6d4);
    box-shadow: 0 10px 28px rgba(37,99,235,0.28);
}
.page-banner::after {
    content: ""; position: absolute; width: 220px; height: 220px; border-radius: 50%;
    right: -60px; top: -90px; background: rgba(255,255,255,0.1);
}
.pb-icon {
    font-size: 2.1rem; background: rgba(255,255,255,0.18); width: 64px; height: 64px;
    display: flex; align-items: center; justify-content: center; border-radius: 16px;
}
.pb-title { color: white; font-size: 1.7rem; font-weight: 700; }
.pb-sub { color: #cffafe; font-size: 0.98rem; }

/* ---- Cards: charts, tables, metrics, inputs ---- */
[data-testid="stPlotlyChart"] {
    background: rgba(255,255,255,0.88); border-radius: 18px; padding: 10px;
    border: 1px solid #dbeafe; box-shadow: 0 8px 24px rgba(30,58,138,0.08);
}
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden; border: 1px solid #bfdbfe;
    box-shadow: 0 6px 18px rgba(30,58,138,0.06);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff, #eff6ff); border: 1px solid #bfdbfe;
    border-top: 5px solid #2563eb; border-radius: 16px; padding: 16px 20px;
    box-shadow: 0 8px 20px rgba(30,58,138,0.08);
}
[data-testid="stMetricValue"] { color: #1d4ed8; font-weight: 700; }
.stTextArea textarea {
    border-radius: 14px !important; border: 2px solid #bfdbfe !important; background: white !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important; box-shadow: 0 0 0 4px rgba(37,99,235,0.15) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border-radius: 14px; border: 2px dashed #93c5fd; background: rgba(239,246,255,0.85);
}
[data-testid="stExpander"] {
    border-radius: 14px; border: 1px solid #bfdbfe; background: rgba(255,255,255,0.85);
}

/* ---- Buttons ---- */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(90deg, #1d4ed8, #06b6d4); color: white !important;
    border: none; border-radius: 12px; padding: 10px 24px; font-weight: 600;
    box-shadow: 0 6px 16px rgba(29,78,216,0.3); transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 22px rgba(29,78,216,0.4); color: white !important;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f4d 0%, #1e3a8a 55%, #0e7490 100%);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div { color: #e0f2fe; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 10px 12px; margin-bottom: 4px; width: 100%;
    transition: all 0.2s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.15); transform: translateX(3px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #2563eb, #06b6d4); border-color: transparent;
    box-shadow: 0 6px 16px rgba(6,182,212,0.35);
}
[data-testid="stSidebar"] .side-logo { font-size: 1.6rem; font-weight: 700; color: white; }
[data-testid="stSidebar"] .side-sub { color: #93c5fd; font-size: 0.85rem; margin-bottom: 12px; }
[data-testid="stSidebar"] .side-head { color: #93c5fd; font-size: 0.8rem; font-weight: 600;
    letter-spacing: 1px; text-transform: uppercase; margin-top: 6px; }
.badge { padding: 8px 12px; border-radius: 10px; margin: 6px 0; font-size: 0.88rem; font-weight: 500; }
[data-testid="stSidebar"] .badge.ok {
    background: rgba(34,197,94,0.18); border: 1px solid rgba(34,197,94,0.5); color: #dcfce7;
}
[data-testid="stSidebar"] .badge.wait {
    background: rgba(250,204,21,0.15); border: 1px solid rgba(250,204,21,0.45); color: #fef9c3;
}
[data-testid="stSidebar"] .side-note { font-size: 0.8rem; color: #bae6fd; margin-top: 14px; line-height: 1.45; }

/* ---- Footer ---- */
.footer {
    text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 40px;
    padding: 16px; border-top: 1px solid #bfdbfe;
}
</style>
""", unsafe_allow_html=True)

HERO = """
<div class="hero">
  <div class="hero-title"><span class="dna">🧬</span> OrthoQuest</div>
  <div class="hero-sub">Distinguishing orthologous and paralogous proteins and analysing
  how evolution shapes protein function</div>
  <div class="hero-tags">
    <span>🔗 UniProt</span><span>🧬 Clustal Omega</span><span>🌳 Phylogenetics</span>
    <span>🏷️ Gene Ontology</span><span>🧩 Pfam domains</span>
  </div>
</div>
"""

def banner(icon, title, subtitle):
    st.markdown(f"""
    <div class="page-banner">
      <div class="pb-icon">{icon}</div>
      <div><div class="pb-title">{title}</div><div class="pb-sub">{subtitle}</div></div>
    </div>""", unsafe_allow_html=True)

def style_fig(fig, height=None):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(239,246,255,0.6)",
                      font=dict(family="Poppins, sans-serif", color="#1e3a8a"),
                      title_font=dict(size=18, color="#1e3a8a"),
                      margin=dict(t=60, l=20, r=20, b=20))
    if height:
        fig.update_layout(height=height)
    return fig

def footer():
    st.markdown('<div class="footer">🧬 OrthoQuest · Comparative protein evolution & function analysis</div>',
                unsafe_allow_html=True)

# ======================= HELPER FUNCTIONS =======================
@st.cache_data
def fetch_fasta(acc):
    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta")
    return r.text if r.ok else ""

def short_name(rec):
    parts = rec.id.split("|")
    return parts[2] if len(parts) >= 3 else rec.id

def organism(rec):
    m = re.search(r"OS=(.+?) OX=", rec.description)
    return m.group(1) if m else "Unknown"

aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5

def percent_identity(s1, s2):
    aln = aligner.align(s1, s2)[0]
    r1, r2 = aln[0], aln[1]
    matches = sum(1 for a, b in zip(r1, r2) if a == b and a != "-")
    return round(matches / len(r1) * 100, 1)

@st.cache_data
def identity_matrix(names, seqs):
    m = pd.DataFrame(index=names, columns=names, dtype=float)
    for i in range(len(seqs)):
        for j in range(i, len(seqs)):
            v = percent_identity(seqs[i], seqs[j])
            m.iloc[i, j] = v
            m.iloc[j, i] = v
    return m

@st.cache_data(show_spinner=False)
def run_clustalo(fasta_text):
    url = "https://www.ebi.ac.uk/Tools/services/rest/clustalo"
    r = requests.post(f"{url}/run", data={"email": EBI_EMAIL, "sequence": fasta_text,
                                          "stype": "protein", "outfmt": "fa"})
    job = r.text.strip()
    if not r.ok or not job.startswith("clustalo-"):
        raise RuntimeError(r.text[:300])
    for _ in range(120):
        status = requests.get(f"{url}/status/{job}").text.strip()
        if status == "FINISHED":
            return requests.get(f"{url}/result/{job}/aln-fasta").text
        if status in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError("Clustal Omega job failed: " + status)
        time.sleep(5)
    raise RuntimeError("Clustal Omega took too long. Please try again.")

def conservation_table(aln_text):
    aln = AlignIO.read(StringIO(aln_text), "fasta")
    n = len(aln)
    rows = []
    for i in range(aln.get_alignment_length()):
        column = aln[:, i]
        residues = [c for c in column if c != "-"]
        gap_fraction = 1 - len(residues) / n
        entropy = 0
        for c in Counter(residues).values():
            p = c / len(residues)
            entropy -= p * math.log2(p)
        score = (1 - entropy / math.log2(20)) * (1 - gap_fraction)
        label = "Conserved" if score >= 0.8 else "Moderate" if score >= 0.5 else "Variable"
        rows.append({"Position": i + 1, "Residues": column,
                     "Conservation": round(score, 3), "Class": label})
    return pd.DataFrame(rows), n

def need_records():
    if "records" not in st.session_state:
        st.info("👈 Start on the **🏠 Home** page: enter UniProt IDs and click **Fetch sequences**.")
        st.stop()
    records = st.session_state["records"]
    return records, [short_name(r) for r in records], [str(r.seq) for r in records]

# ======================= SIDEBAR =======================
PAGES = ["🏠 Home", "🧾 Sequences", "🔥 Similarity", "🧬 Alignment & Conservation",
         "🌳 Phylogenetic Tree", "🔀 Orthologs & Paralogs", "⚙️ Function Analysis", "📄 Report"]

with st.sidebar:
    st.markdown('<div class="side-logo"><span class="dna">🧬</span> OrthoQuest</div>'
                '<div class="side-sub">Protein evolution explorer</div>', unsafe_allow_html=True)
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    badges = '<div class="side-head">Status</div>'
    if "records" in st.session_state:
        badges += f'<div class="badge ok">✅ {len(st.session_state["records"])} proteins loaded</div>'
    else:
        badges += '<div class="badge wait">⏳ No proteins loaded yet</div>'
    if "aln_text" in st.session_state:
        badges += '<div class="badge ok">✅ Alignment ready</div>'
    else:
        badges += '<div class="badge wait">⏳ Alignment not run yet</div>'
    badges += ('<div class="side-note">Results indicate <i>possible</i> functional divergence, '
               'not experimentally proven changes.</div>')
    st.markdown(badges, unsafe_allow_html=True)

# ======================= PAGES =======================
if page == "🏠 Home":
    st.markdown(HERO, unsafe_allow_html=True)
    st.subheader("📥 Input proteins")
    example = "P69905, P68871, P02144, P01942, P02088, P04247, P01994, P02112, P02197"
    ids_text = st.text_area("UniProt IDs (separated by commas, spaces or new lines)", example)
    uploaded = st.file_uploader("...or upload a FASTA file instead", type=["fasta", "fa", "txt"])

    if st.button("🚀 Fetch sequences", type="primary"):
        records, failed = [], []
        if uploaded is not None:
            records = list(SeqIO.parse(StringIO(uploaded.getvalue().decode()), "fasta"))
        else:
            ids = [x for x in re.split(r"[,\s]+", ids_text) if x]
            with st.spinner("Fetching sequences from UniProt..."):
                for acc in ids:
                    text = fetch_fasta(acc)
                    if text:
                        records.extend(SeqIO.parse(StringIO(text), "fasta"))
                    else:
                        failed.append(acc)
        st.session_state["failed"] = failed
        if len(records) < 3:
            st.error("Please provide at least 3 proteins.")
        else:
            st.session_state["records"] = records
            st.session_state.pop("aln_text", None)   # clear old alignment
            st.rerun()

    if st.session_state.get("failed"):
        st.warning("Could not find these IDs: " + ", ".join(st.session_state["failed"]))
    if "records" in st.session_state:
        st.success(f"✅ {len(st.session_state['records'])} proteins loaded. "
                   "Open **🧾 Sequences** in the sidebar to continue.")
    footer()

elif page == "🧾 Sequences":
    banner("🧾", "Sequences", "Proteins retrieved for analysis")
    records, names, seqs = need_records()
    table = pd.DataFrame({"Protein": names,
                          "Organism": [organism(r) for r in records],
                          "Length": [len(s) for s in seqs]})
    c1, c2, c3 = st.columns(3)
    c1.metric("Proteins", len(records))
    c2.metric("Organisms", table["Organism"].nunique())
    c3.metric("Average length", int(table["Length"].mean()))
    st.dataframe(table, hide_index=True)
    fig0 = px.bar(table, x="Protein", y="Length", color="Organism",
                  title="Protein length by organism",
                  color_discrete_sequence=CHART_COLORS)
    st.plotly_chart(style_fig(fig0))
    footer()

elif page == "🔥 Similarity":
    banner("🔥", "Pairwise Similarity", "Percent identity between every pair of proteins")
    records, names, seqs = need_records()
    with st.spinner("Comparing all pairs..."):
        matrix = identity_matrix(names, seqs)
    fig = px.imshow(matrix, text_auto=True, zmin=0, zmax=100,
                    color_continuous_scale=BLUE_SCALE,
                    title="Pairwise sequence identity (%)")
    st.plotly_chart(style_fig(fig, height=600))
    st.download_button("⬇️ Download similarity table (CSV)",
                       matrix.to_csv(), "identity_matrix.csv")
    footer()

elif page == "🧬 Alignment & Conservation":
    banner("🧬", "Alignment & Conservation",
           "Multiple sequence alignment with Clustal Omega and per-position conservation")
    records, names, seqs = need_records()
    if st.button("🧬 Run alignment"):
        fasta_text = "".join(f">{n}\n{s}\n" for n, s in zip(names, seqs))
        try:
            with st.spinner("Aligning with Clustal Omega... this can take up to a minute"):
                st.session_state["aln_text"] = run_clustalo(fasta_text)
            st.rerun()
        except Exception as e:
            st.error(f"The alignment service gave an error: {e}")

    aln_file = st.file_uploader("Plan B: upload an aligned FASTA file instead",
                                type=["fasta", "fa", "txt"], key="aln_upload")
    if aln_file is not None:
        st.session_state["aln_text"] = aln_file.getvalue().decode()

    if "aln_text" in st.session_state:
        aln_text = st.session_state["aln_text"]
        cons, n = conservation_table(aln_text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Alignment length", len(cons))
        c2.metric("Conserved positions", int((cons["Class"] == "Conserved").sum()))
        c3.metric("Variable positions", int((cons["Class"] == "Variable").sum()))

        fig2 = px.bar(cons, x="Position", y="Conservation", color="Class",
                      hover_data=["Residues"],
                      color_discrete_map={"Conserved": "#1d4ed8",
                                          "Moderate": "#38bdf8",
                                          "Variable": "#f43f5e"},
                      title="Conservation score per alignment position (Shannon entropy)")
        st.plotly_chart(style_fig(fig2))

        fully = cons[cons["Residues"].apply(lambda col: len(set(col)) == 1 and "-" not in col)]
        st.subheader(f"🎯 Fully conserved in all {n} proteins")
        st.dataframe(pd.DataFrame({"Position": fully["Position"],
                                   "Amino acid": fully["Residues"].str[0]}),
                     hide_index=True)

        with st.expander("Show the alignment"):
            st.code(aln_text)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download alignment (FASTA)", aln_text, "alignment.fasta")
        d2.download_button("⬇️ Download conservation scores (CSV)",
                           cons.to_csv(index=False), "conservation_scores.csv")
    footer()

elif page == "🌳 Phylogenetic Tree":
    banner("🌳", "Phylogenetic Tree",
           "Neighbour-Joining tree with 100 bootstrap replicates")
    records, names, seqs = need_records()
    if "aln_text" not in st.session_state:
        st.info("👈 Run the alignment first on the **🧬 Alignment & Conservation** page.")
        st.stop()

    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from Bio import Phylo
    from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
    from Bio.Phylo.Consensus import bootstrap_trees

    @st.cache_data(show_spinner=False)
    def build_tree_newick(aln_text, n_boot=100):
        aln = AlignIO.read(StringIO(aln_text), "fasta")
        for rec in aln:
            parts = rec.id.split("|")
            rec.id = parts[2] if len(parts) >= 3 else rec.id
        constructor = DistanceTreeConstructor(DistanceCalculator("blosum62"), "nj")
        tree = constructor.build_tree(aln)
        tree.root_at_midpoint()
        replicates = list(bootstrap_trees(aln, n_boot, constructor))

        def splits(t):
            nm = frozenset(x.name for x in t.get_terminals())
            out = set()
            for c in t.get_nonterminals():
                leaves = frozenset(x.name for x in c.get_terminals())
                if 1 < len(leaves) < len(nm):
                    out.add(leaves)
                    out.add(nm - leaves)
            return out

        rep_splits = [splits(r) for r in replicates]
        all_names = frozenset(x.name for x in tree.get_terminals())
        for clade in tree.get_nonterminals():
            leaves = frozenset(x.name for x in clade.get_terminals())
            if leaves == all_names:
                clade.confidence = None
            else:
                clade.confidence = round(100 * sum(leaves in s for s in rep_splits) / len(rep_splits))
            clade.name = None
        handle = StringIO()
        Phylo.write(tree, handle, "newick")
        return handle.getvalue()

    with st.spinner("Building the tree and running 100 bootstrap replicates..."):
        nwk = build_tree_newick(st.session_state["aln_text"])
    st.session_state["tree"] = (st.session_state["aln_text"], nwk)

    tree = Phylo.read(StringIO(nwk), "newick")
    supports = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
    c1, c2, c3 = st.columns(3)
    c1.metric("Proteins in tree", len(tree.get_terminals()))
    c2.metric("Bootstrap replicates", 100)
    c3.metric("Lowest branch support", f"{min(supports):.0f}%" if supports else "–")

    palette = ["#1d4ed8", "#0891b2", "#1e3a8a", "#0ea5e9", "#4f46e5", "#0d9488"]
    species_list = sorted({t.name.split("_")[-1] for t in tree.get_terminals()})
    sp_color = {sp: palette[i % len(palette)] for i, sp in enumerate(species_list)}
    for c in tree.find_clades():
        c.color = "#1e3a8a"

    fig, ax = plt.subplots(figsize=(10, max(4, 0.6 * len(tree.get_terminals()))))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fbff")
    Phylo.draw(tree, axes=ax, do_show=False,
               label_func=lambda c: c.name if c.is_terminal() else "",
               label_colors=lambda label: sp_color.get(label.split("_")[-1], "#1e3a8a"))
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.set_xlabel("Branch length (evolutionary distance)", color="#1e3a8a")
    ax.set_title("Numbers on branches = bootstrap support (%)", color="#1e3a8a", fontsize=11)
    for s in ax.spines.values():
        s.set_color("#bfdbfe")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.pyplot(fig)

    st.markdown("**Species colours:** " + " ".join(
        f'<span style="background:{col};color:white;padding:4px 12px;'
        f'border-radius:999px;margin-right:6px;">{sp}</span>'
        for sp, col in sp_color.items()), unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download tree image (PNG)", buf.getvalue(), "phylogenetic_tree.png")
    d2.download_button("⬇️ Download tree (Newick)", nwk, "phylogenetic_tree.nwk")
    footer()

elif page == "🔀 Orthologs & Paralogs":
    banner("🔀", "Orthologs & Paralogs",
           "Species-overlap classification of every protein pair")
    records, names, seqs = need_records()
    tree_info = st.session_state.get("tree")
    if not tree_info or tree_info[0] != st.session_state.get("aln_text"):
        st.info("👈 Build the tree first on the **🌳 Phylogenetic Tree** page.")
        st.stop()

    import itertools
    from Bio import Phylo
    tree = Phylo.read(StringIO(tree_info[1]), "newick")
    leaves = [t.name for t in tree.get_terminals()]
    if any("_" not in n for n in leaves):
        st.warning("Some protein names have no species ending (like _HUMAN). "
                   "The classification needs the species, so results may be wrong for these.")

    def species(n):
        return n.split("_")[-1]

    def event(clade):
        seen = set()
        for child in clade.clades:
            sp = {species(t.name) for t in child.get_terminals()}
            if seen & sp:
                return "Duplication"
            seen |= sp
        return "Speciation"

    def node_support(clade):
        if clade.confidence is not None:
            return clade.confidence
        values = [c.confidence for c in clade.clades if c.confidence is not None]
        return min(values) if values else None

    rows, rel = [], {}
    for a, b in itertools.combinations(leaves, 2):
        lca = tree.common_ancestor(a, b)
        ev = event(lca)
        sup = node_support(lca)
        r = "Ortholog" if ev == "Speciation" else "Paralog"
        rel[(a, b)] = rel[(b, a)] = r
        rows.append({"Protein 1": a, "Protein 2": b, "Event": ev, "Relationship": r,
                     "Support (%)": sup,
                     "Confidence": "High" if sup is not None and sup >= 70 else "Low"})
    df = pd.DataFrame(rows)
    st.session_state["pairs"] = df

    c1, c2, c3 = st.columns(3)
    c1.metric("Protein pairs", len(df))
    c2.metric("Orthologs", int((df["Relationship"] == "Ortholog").sum()))
    c3.metric("Paralogs", int((df["Relationship"] == "Paralog").sum()))

    z = [[2 if a == b else (1 if rel[(a, b)] == "Ortholog" else 0) for b in leaves] for a in leaves]
    text = [["–" if a == b else rel[(a, b)][0] for b in leaves] for a in leaves]
    scale = [[0, "#67e8f9"], [0.333, "#67e8f9"], [0.333, "#1d4ed8"],
             [0.666, "#1d4ed8"], [0.666, "#e2e8f0"], [1, "#e2e8f0"]]
    fig = px.imshow(z, x=leaves, y=leaves, zmin=0, zmax=2, color_continuous_scale=scale,
                    title="Relationship map (O = ortholog, P = paralog)")
    fig.update_traces(text=text, texttemplate="%{text}",
                      hovertemplate="%{y} – %{x}: %{text}<extra></extra>")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(style_fig(fig, height=600))

    st.subheader("🌿 Evolutionary events in the tree")
    ev_rows = [{"Event": event(c), "Support (%)": node_support(c),
                "Proteins below this point": ", ".join(t.name for t in c.get_terminals())}
               for c in tree.get_nonterminals()]
    st.dataframe(pd.DataFrame(ev_rows), hide_index=True)

    st.subheader("📋 All protein pairs")
    st.dataframe(df, hide_index=True)
    st.download_button("⬇️ Download classification (CSV)",
                       df.to_csv(index=False), "ortholog_paralog.csv")
    footer()
elif page == "⚙️ Function Analysis":
    banner("⚙️", "Function Analysis",
           "Evolution-to-Function Impact Analyzer: GO terms, domains and functional conservation")
    records, names, seqs = need_records()
    pairs = st.session_state.get("pairs")
    if pairs is None or set(pairs["Protein 1"]) | set(pairs["Protein 2"]) != set(names):
        st.info("👈 Open the **🔀 Orthologs & Paralogs** page first.")
        st.stop()

    import os

    @st.cache_resource(show_spinner=False)
    def load_go():
        from goatools.obo_parser import GODag
        path = "data/go-basic.obo"
        if not os.path.exists(path):
            os.makedirs("data", exist_ok=True)
            r = requests.get("https://purl.obolibrary.org/obo/go/go-basic.obo")
            with open(path, "wb") as f:
                f.write(r.content)
        return GODag(path, prt=None)

    @st.cache_data(show_spinner=False)
    def fetch_annotation(acc):
        experimental = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP",
                        "HTP", "HDA", "HMP", "HGI", "HEP"}
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json")
        if not r.ok:
            return None
        go_all, go_exp, pfam = {}, [], {}
        for x in r.json().get("uniProtKBCrossReferences", []):
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            if x["database"] == "GO":
                go_all[x["id"]] = props.get("GoTerm", "")
                if props.get("GoEvidenceType", "").split(":")[0] in experimental:
                    go_exp.append(x["id"])
            elif x["database"] == "Pfam":
                pfam[x["id"]] = props.get("EntryName", "")
        return {"go": go_all, "go_experimental": go_exp, "pfam": pfam}

    acc_of = {}
    for r in records:
        parts = r.id.split("|")
        if len(parts) >= 3:
            acc_of[parts[2]] = parts[1]
    leaves = list(dict.fromkeys(list(pairs["Protein 1"]) + list(pairs["Protein 2"])))
    missing = [n for n in leaves if n not in acc_of]
    if missing:
        st.warning("These proteins have no UniProt accession, so their functions "
                   "cannot be fetched: " + ", ".join(missing))
        st.stop()

    with st.spinner("Loading the Gene Ontology (the first time can take a minute)..."):
        dag = load_go()
    with st.spinner("Fetching GO terms and Pfam domains from UniProt..."):
        ann = {n: fetch_annotation(acc_of[n]) for n in leaves}
    if any(v is None for v in ann.values()):
        st.error("Could not fetch annotations for some proteins. Please try again.")
        st.stop()

    ROOTS = {"GO:0003674", "GO:0008150", "GO:0005575"}

    def expand(terms):
        out = set()
        for t in terms:
            if t in dag:
                out.add(dag[t].id)
                out |= dag[t].get_all_parents()
        return out - ROOTS

    def overlap(a, b):
        a, b = set(a), set(b)
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    def functions(go):
        return {v[2:] for v in go.values() if v.startswith("F:")}

    matrix = identity_matrix(names, seqs)
    rows = []
    for _, p in pairs.iterrows():
        a, b = p["Protein 1"], p["Protein 2"]
        A, B = ann[a], ann[b]
        go_sim = overlap(expand(A["go"]), expand(B["go"]))
        dom_sim = overlap(A["pfam"], B["pfam"])
        score = round(0.7 * go_sim + 0.3 * dom_sim, 3)
        weak = min(len(A["go_experimental"]), len(B["go_experimental"])) < 3
        if weak:
            status = "Inconclusive (limited annotation)"
        elif score >= 0.8:
            status = "Conserved"
        elif score >= 0.5:
            status = "Partially diverged"
        else:
            status = "Possibly diverged"
        rows.append({
            "Protein 1": a, "Protein 2": b, "Relationship": p["Relationship"],
            "Same species": "Yes" if a.split("_")[-1] == b.split("_")[-1] else "No",
            "Seq identity (%)": matrix.loc[a, b],
            "GO similarity": round(go_sim, 3), "Domain similarity": round(dom_sim, 3),
            "FCS": score, "Status": status,
            "Evidence": "Weak (mostly predicted)" if weak else "Experimental",
            "Only in Protein 1": "; ".join(sorted(functions(A["go"]) - functions(B["go"]))),
            "Only in Protein 2": "; ".join(sorted(functions(B["go"]) - functions(A["go"]))),
        })
    fcs = pd.DataFrame(rows)
    st.session_state["fcs"] = fcs

    exp_df = fcs[fcs["Evidence"] == "Experimental"]

    def avg(data, rel):
        sub = data[data["Relationship"] == rel]["FCS"]
        return f"{sub.mean():.3f}" if len(sub) else "–"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ortholog FCS (experimental)", avg(exp_df, "Ortholog"))
    c2.metric("Paralog FCS (experimental)", avg(exp_df, "Paralog"))
    c3.metric("Experimental pairs", len(exp_df))
    c4.metric("Inconclusive pairs", int((fcs["Evidence"] != "Experimental").sum()))

    conj = []
    for label, data in [("All pairs", fcs), ("Experimental only", exp_df)]:
        for rel in ["Ortholog", "Paralog"]:
            sub = data[data["Relationship"] == rel]
            if len(sub):
                conj.append({"Set": label, "Relationship": rel,
                             "Average FCS": round(sub["FCS"].mean(), 3), "Pairs": len(sub)})
    conj = pd.DataFrame(conj)
    st.session_state["conj"] = conj
    fig = px.bar(conj, x="Set", y="Average FCS", color="Relationship", barmode="group",
                 text="Average FCS", hover_data=["Pairs"],
                 color_discrete_map={"Ortholog": "#1d4ed8", "Paralog": "#06b6d4"},
                 title="Ortholog conjecture: are orthologs more functionally conserved than paralogs?")
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(style_fig(fig))

    M = pd.DataFrame(1.0, index=leaves, columns=leaves)
    for _, r in fcs.iterrows():
        M.loc[r["Protein 1"], r["Protein 2"]] = r["FCS"]
        M.loc[r["Protein 2"], r["Protein 1"]] = r["FCS"]
    fig2 = px.imshow(M, text_auto=".2f", zmin=0, zmax=1, color_continuous_scale=BLUE_SCALE,
                     title="Functional Conservation Score (FCS) between proteins")
    st.plotly_chart(style_fig(fig2, height=600))

    st.subheader("🚩 Possible functional divergence")
    flagged = fcs[fcs["Status"].isin(["Partially diverged", "Possibly diverged"])]
    if flagged.empty:
        st.success("No experimentally supported functional divergence was flagged.")
    else:
        st.caption("Only pairs with experimental GO evidence are flagged. "
                   "Open a pair to see which functions differ.")
        for _, r in flagged.sort_values("FCS").iterrows():
            with st.expander(f"{r['Protein 1']} ↔ {r['Protein 2']}  ·  {r['Relationship']}  ·  "
                             f"FCS {r['FCS']:.2f}  ·  {r['Status']}"):
                st.markdown(f"**Only in {r['Protein 1']}:** {r['Only in Protein 1'] or '–'}")
                st.markdown(f"**Only in {r['Protein 2']}:** {r['Only in Protein 2'] or '–'}")
                if r["Same species"] == "Yes":
                    st.caption("Same-species pair: shared experiments can make annotations look more similar.")

    st.subheader("🧩 Domain architecture")
    all_domains = sorted({f"{k} ({v})" for n in leaves for k, v in ann[n]["pfam"].items()})
    if all_domains:
        dom_table = pd.DataFrame([
            {"Protein": n, **{d: ("✅" if d.split(" ")[0] in ann[n]["pfam"] else "—")
                              for d in all_domains}} for n in leaves])
        st.dataframe(dom_table, hide_index=True)
        variable = [d for d in all_domains
                    if not all(d.split(" ")[0] in ann[n]["pfam"] for n in leaves)]
        if variable:
            note = ("Possible domain gain or loss: " + ", ".join(variable) +
                    " is present in only some of the proteins.")
            st.warning(note)
        else:
            note = ("All proteins share the same domain architecture, so no domain "
                    "gain or loss was detected in this set.")
            st.info(note)
    else:
        note = "No Pfam domains were found for these proteins."
        st.info(note)
    st.session_state["domain_note"] = note

    st.subheader("🏷️ Annotations per protein")
    annot = pd.DataFrame([{
        "Protein": n, "GO terms": len(ann[n]["go"]),
        "Experimental GO": len(ann[n]["go_experimental"]),
        "Pfam domains": ", ".join(f"{k} ({v})" for k, v in ann[n]["pfam"].items()) or "–",
        "Molecular functions": "; ".join(sorted(functions(ann[n]["go"])))} for n in leaves])
    st.session_state["annot"] = annot
    st.dataframe(annot, hide_index=True)

    st.subheader("📋 All protein pairs")
    st.dataframe(fcs, hide_index=True)
    st.download_button("⬇️ Download functional conservation (CSV)",
                       fcs.to_csv(index=False), "functional_conservation.csv")
    st.info("Results indicate *possible* functional divergence, not experimentally proven "
            "changes. Pairs where either protein has fewer than 3 experimental GO terms "
            "are marked **Inconclusive**.")
    footer()

elif page == "📄 Report":
    banner("📄", "Report", "Summary of all analyses, ready to download")
    records, names, seqs = need_records()
    import base64, datetime
    from io import BytesIO
    import streamlit.components.v1 as components

    def matches(df):
        return df is not None and set(df["Protein 1"]) | set(df["Protein 2"]) == set(names)

    sections, missing = [], []
    sections.append(("Methods", "<ul>"
        "<li><b>Sequence retrieval:</b> UniProt REST API.</li>"
        "<li><b>Pairwise similarity:</b> global alignment (BLOSUM62, gap open −10, extend −0.5), percent identity.</li>"
        "<li><b>Multiple sequence alignment:</b> Clustal Omega (EMBL-EBI web service).</li>"
        "<li><b>Conservation:</b> Shannon entropy per alignment column, weighted by gap fraction.</li>"
        "<li><b>Phylogeny:</b> Neighbour-Joining tree from BLOSUM62 distances, midpoint-rooted, 100 bootstrap replicates.</li>"
        "<li><b>Ortholog/paralog classification:</b> species-overlap method on the gene tree.</li>"
        "<li><b>Functional Conservation Score:</b> 0.7 × GO similarity (ancestor-expanded overlap coefficient) "
        "+ 0.3 × Pfam domain overlap. Pairs with fewer than 3 experimental GO terms are marked inconclusive.</li>"
        "</ul>"))

    prot = pd.DataFrame({"Protein": names, "Organism": [organism(r) for r in records],
                         "Length": [len(s) for s in seqs]})
    sections.append(("1. Input proteins",
                     f"<p>{len(records)} proteins from {prot['Organism'].nunique()} organisms.</p>"
                     + prot.to_html(index=False, border=0)))

    m = identity_matrix(names, seqs)
    vals = [m.iloc[i, j] for i in range(len(names)) for j in range(i + 1, len(names))]
    sections.append(("2. Pairwise sequence similarity",
                     f"<p>Percent identity ranged from {min(vals)}% to {max(vals)}%.</p>"
                     + m.to_html(border=0)))

    aln_text = st.session_state.get("aln_text")
    if aln_text:
        cons, n = conservation_table(aln_text)
        fully = cons[cons["Residues"].apply(lambda col: len(set(col)) == 1 and "-" not in col)]
        fully_list = ", ".join(f"{p} ({c[0]})" for p, c in zip(fully["Position"], fully["Residues"]))
        sections.append(("3. Multiple sequence alignment and conservation",
            f"<p>Alignment length: <b>{len(cons)}</b> positions. "
            f"Conserved: <b>{int((cons['Class'] == 'Conserved').sum())}</b>, "
            f"moderate: <b>{int((cons['Class'] == 'Moderate').sum())}</b>, "
            f"variable: <b>{int((cons['Class'] == 'Variable').sum())}</b>.</p>"
            f"<p><b>Fully conserved in all {n} proteins:</b> {fully_list or 'none'}</p>"))
    else:
        missing.append("🧬 Alignment & Conservation")

    tree_info = st.session_state.get("tree")
    if tree_info and tree_info[0] == aln_text:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from Bio import Phylo
        tree = Phylo.read(StringIO(tree_info[1]), "newick")
        supports = [c.confidence for c in tree.get_nonterminals() if c.confidence is not None]
        fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(names))))
        Phylo.draw(tree, axes=ax, do_show=False,
                   label_func=lambda c: c.name if c.is_terminal() else "")
        ax.set_yticks([])
        ax.set_ylabel("")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        img = base64.b64encode(buf.getvalue()).decode()
        sections.append(("4. Phylogenetic tree",
            f"<p>Lowest bootstrap support: <b>{min(supports):.0f}%</b> "
            f"(numbers on branches are bootstrap percentages).</p>"
            f"<img src='data:image/png;base64,{img}' style='max-width:100%'>"))
    else:
        missing.append("🌳 Phylogenetic Tree")

    pairs = st.session_state.get("pairs")
    if matches(pairs):
        sections.append(("5. Ortholog and paralog classification",
            f"<p><b>{(pairs['Relationship'] == 'Ortholog').sum()}</b> ortholog pairs and "
            f"<b>{(pairs['Relationship'] == 'Paralog').sum()}</b> paralog pairs "
            f"out of {len(pairs)}.</p>" + pairs.to_html(index=False, border=0)))
    else:
        missing.append("🔀 Orthologs & Paralogs")

    fcs = st.session_state.get("fcs")
    if matches(fcs):
        flagged = fcs[fcs["Status"].isin(["Partially diverged", "Possibly diverged"])]
        flag_html = "".join(
            f"<li><b>{r['Protein 1']} ↔ {r['Protein 2']}</b> ({r['Relationship']}, FCS {r['FCS']:.2f}): "
            f"only in {r['Protein 1']}: {r['Only in Protein 1'] or '–'}; "
            f"only in {r['Protein 2']}: {r['Only in Protein 2'] or '–'}</li>"
            for _, r in flagged.iterrows()) or "<li>None flagged.</li>"
        cols = ["Protein 1", "Protein 2", "Relationship", "Seq identity (%)",
                "FCS", "Status", "Evidence"]
        sections.append(("6. Evolution-to-Function Impact Analysis",
            "<h3>Ortholog conjecture test</h3>"
            + st.session_state["conj"].to_html(index=False, border=0)
            + "<h3>Possible functional divergence</h3><ul>" + flag_html + "</ul>"
            + f"<h3>Domain architecture</h3><p>{st.session_state.get('domain_note', '')}</p>"
            + "<h3>Annotations per protein</h3>"
            + st.session_state["annot"].to_html(index=False, border=0)
            + "<h3>Functional conservation per pair</h3>"
            + fcs[cols].to_html(index=False, border=0)))
    else:
        missing.append("⚙️ Function Analysis")

    sections.append(("Limitations",
        "<div class='note'><ul>"
        "<li>Results indicate <i>possible</i> functional divergence, not experimentally proven changes.</li>"
        "<li>GO annotations are uneven: well-studied species have many more experimental terms, "
        "and same-species proteins often share annotations from the same experiments.</li>"
        "<li>Ortholog/paralog calls depend on correct species labels and on the accuracy of the gene tree.</li>"
        "<li>Neighbour-Joining is a fast distance method; maximum-likelihood trees may resolve difficult cases better.</li>"
        "</ul></div>"))

    REPORT_CSS = """<style>
    body { font-family: Poppins, Arial, sans-serif; color: #0f172a; background: #f0f7ff; margin: 0; padding: 30px; }
    .wrap { max-width: 1000px; margin: auto; background: white; border-radius: 18px; padding: 36px;
            box-shadow: 0 10px 30px rgba(30,58,138,0.12); }
    .head { background: linear-gradient(120deg, #1e3a8a, #2563eb, #06b6d4); color: white;
            padding: 26px 30px; border-radius: 14px; margin-bottom: 24px; }
    .head h1 { margin: 0; }
    h2 { color: #1e3a8a; border-bottom: 3px solid #dbeafe; padding-bottom: 6px; margin-top: 30px; }
    h3 { color: #1d4ed8; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 10px 0; }
    th { background: #1d4ed8; color: white; padding: 6px 8px; text-align: left; }
    td { padding: 5px 8px; border-bottom: 1px solid #dbeafe; }
    tr:nth-child(even) td { background: #f5f9ff; }
    .note { background: #eff6ff; border-left: 5px solid #2563eb; padding: 12px 16px; border-radius: 8px; }
    </style>"""
    body = "".join(f"<h2>{t}</h2>{c}" for t, c in sections)
    now = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
    report = (f"<html><head><meta charset='utf-8'><title>OrthoQuest Report</title>{REPORT_CSS}</head>"
              f"<body><div class='wrap'><div class='head'><h1>🧬 OrthoQuest Report</h1>"
              f"<div>Generated on {now}</div></div>{body}</div></body></html>")

    if missing:
        st.warning("Not included yet: " + ", ".join(missing) +
                   ". Open those pages first to add them to the report.")
    st.download_button("⬇️ Download report (HTML)", report, "OrthoQuest_report.html",
                       mime="text/html")
    st.caption("Tip: open the downloaded report in your browser and press Ctrl + P → "
               "Save as PDF to get a PDF.")
    components.html(report, height=900, scrolling=True)
    footer()
else:
    icon, title = page.split(" ", 1)
    banner(icon, title, "Coming soon")
    need_records()
    st.info("🚧 This page will be added in the next step.")
    footer()