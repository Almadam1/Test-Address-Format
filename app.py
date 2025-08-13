import streamlit as st
import pandas as pd
import re
import io
import time
from openai import OpenAI, RateLimitError

# =========================
# OpenAI client (v1 syntax)
# =========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
# Config / Options
# =========================
DEFAULT_MAX_ROWS = 100  # default cap for testing; user can change in UI

# Compound names to preserve (extendable via sidebar input)
DEFAULT_COMPOUNDS = [
    "northwood", "southgate", "eastwood", "westfield",
    "northridge", "southridge", "eastview", "westpark",
    "mount carmel", "green valley", "ridgeview", "lake shore",
    "pine hill", "maple grove", "cedar point", "oak ridge",
    "willow bend", "river road", "sunset boulevard", "highland avenue",
    "park place", "garden lane", "college drive", "union square",
    "harbor view", "brook stone", "fox run", "mill creek",
    "spring meadow", "pleasant hill", "fair view", "meadow brook",
    "stone bridge", "bridge way", "silver spring", "golden gate",
]

# =========================
# Helpers: compound handling
# =========================
def protect_compounds(s: str, compounds: list[str]) -> str:
    s_low = str(s)
    for name in compounds:
        token = name.replace(" ", "")
        parts = name.split()
        patterns = [rf"\b{name}\b"]
        if len(parts) == 2:
            patterns.append(rf"\b{re.escape(parts[0])}\s+{re.escape(parts[1])}\b")
        for pat in patterns:
            s_low = re.sub(pat, token, s_low, flags=re.IGNORECASE)
    return s_low

def restore_compounds(s: str, compounds: list[str]) -> str:
    out = str(s)
    for name in compounds:
        token = name.replace(" ", "")
        canonical = token.title()
        out = re.sub(rf"\b{re.escape(token)}\b", canonical, out, flags=re.IGNORECASE)
        parts = name.split()
        if len(parts) == 2:
            first, last = parts[0], parts[1]
            out = re.sub(rf"\b(?:n|north)\s+{re.escape(last)}\b", canonical, out, flags=re.IGNORECASE)
            out = re.sub(rf"\b{re.escape(first)}\s+{re.escape(last)}\b", canonical, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# =========================
# Rule-based cleaner
# =========================
def rule_clean(address: str) -> str:
    if address is None or str(address).strip() == "":
        return ""
    s = str(address).lower().strip()

    # --- Units & hash handling FIRST so we don't lose the signal by stripping '#'
    # Convert "#5C" (or " # 5C") to "apt 5c"
    s = re.sub(r'(?:^|\s)#\s*([a-z0-9-]+)\b', r' apt \1', s, flags=re.IGNORECASE)
    # Normalize explicit unit synonyms
    s = re.sub(r'\b(apartment|apt)\b\.?', 'apt', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(suite|ste)\b\.?', 'ste', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(room|rm)\b', 'rm', s, flags=re.IGNORECASE)
    s = re.sub(r'\b(unit|floor|fl)\b', 'apt', s, flags=re.IGNORECASE)

    # Remove stray punctuation
    s = re.sub(r"[.,#]", "", s)

    # Directionals
    directionals = {
        "north": "n", "south": "s", "east": "e", "west": "w",
        "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    }
    for word, abbr in directionals.items():
        s = re.sub(rf"\b{word}\b", abbr, s)

    # Street types (subset; extend as needed)
    street_types = {
        "avenue": "ave", "av": "ave", "aven": "ave", "avenu": "ave", "avnue": "ave",
        "boulevard": "blvd", "boul": "blvd", "boulv": "blvd",
        "street": "st", "str": "st", "stret": "st", "strt": "st",
        "drive": "dr", "driv": "dr", "drv": "dr",
        "road": "rd", "lane": "ln", "terrace": "ter", "terr": "ter",
        "place": "pl", "court": "ct",
        "circle": "cir", "circl": "cir", "circ": "cir",
        "parkway": "pkwy", "pkway": "pkwy", "parkwy": "pkwy", "pky": "pkwy",
        "junction": "jct", "jctn": "jct", "junctn": "jct", "junctions": "jcts",
        "mount": "mt", "mountain": "mtn", "mountin": "mtn",
        "heights": "hts", "highway": "hwy", "expressway": "expy",
    }
    for word, abbr in street_types.items():
        s = re.sub(rf"\b{word}\b", abbr, s)

    # Infer unlabeled trailing unit (e.g., "... Ave 5C" -> "... Ave Apt 5C")
    if not re.search(r'\b(apt|ste|rm)\b', s, flags=re.IGNORECASE):
        s = re.sub(
            r'\b(ave|st|blvd|dr|rd|ln|ter|pl|ct|cir|pkwy|hwy|way|loop|trl|expy)\b\s+([a-z0-9-]{1,6})$',
            r'\1 apt \2',
            s,
            flags=re.IGNORECASE
        )

    # PO Box normalization
    s = re.sub(r'\b(pobox|pob|po box|po#|box)\s*(\w+)', r'PO Box \2', s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s).strip()
    return s.title()

# =========================
# LLM correction (address line only)
# =========================
def llm_correct(address_line: str, compounds: list[str]) -> str:
    pre = protect_compounds(address_line, compounds)

    sys_prompt = (
        "You are a data formatting assistant. "
        "Format the following U.S. address LINE ONLY (no city/state/ZIP). "
        "Rules: abbreviate directions (North→N, Southwest→SW); use USPS street abbreviations (St, Ave, Blvd, Rd, Dr, Ln, Ter, Pl, Ct, Cir, Pkwy, Hwy, etc.); "
        "format PO Boxes as 'PO Box ###'; normalize unit designators (Apt, Ste, Rm); "
        "if the line ends with an unlabeled unit like '5C' (or was '#5C'), convert to 'Apt 5C'; "
        "remove special characters; title case; single spaces only; "
        "DO NOT include city/state/ZIP; DO NOT split compound street names like Northwood/Eastwood/Mount Carmel. "
        "Return one single line only."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": pre},
            ],
            temperature=0.2,
        )
        out = resp.choices[0].message.content.strip()
    except RateLimitError:
        time.sleep(5)
        return llm_correct(address_line, compounds)
    except Exception as e:
        return f"[LLM ERROR] {e}"

    out = restore_compounds(out, compounds)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# =========================
# Canonicalization for comparison
# =========================
def canonicalize_for_compare(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[.,#]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# UI: Modes (Evaluation / Production)
# =========================
st.title("📬 Address Cleaner & Formatter (CSV-only)")
st.write("Rules + LLM pipeline: merge lines → apply deterministic rules → apply LLM with your standard → output ONE corrected address line. Choose **Evaluation** to compare vs ground-truth, or **Production** to generate corrected CSV.")

st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["Evaluation (compare vs. correct CSV)", "Production (generate corrected CSV)"])
max_rows = st.sidebar.number_input("Max rows to process", min_value=1, max_value=5000, value=DEFAULT_MAX_ROWS, step=1)
use_llm = st.sidebar.checkbox("Use LLM enhancement (after rules)", value=True)
compound_extra = st.sidebar.text_area("Extra compound street names to preserve (comma-separated)", value="")
sort_by_id = st.sidebar.checkbox("Sort results by ID (when ID is used)", value=True)

compounds = DEFAULT_COMPOUNDS.copy()
if compound_extra.strip():
    compounds.extend([c.strip().lower() for c in compound_extra.split(",") if c.strip()])

def normalize_id_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
    )

# =========================
# EVALUATION MODE
# =========================
if mode.startswith("Evaluation"):
    st.header("🧪 Evaluation: Raw vs. Correct CSV")
    raw_file = st.file_uploader("Upload RAW addresses (CSV)", type=["csv"], key="raw_eval")
    correct_file = st.file_uploader("Upload CORRECT addresses (CSV)", type=["csv"], key="correct_eval")

    if raw_file and correct_file:
        raw_df = pd.read_csv(raw_file)
        cor_df = pd.read_csv(correct_file)

        st.subheader("Select Columns")
        raw_cols = list(raw_df.columns)
        cor_cols = list(cor_df.columns)

        raw_line1_col = st.selectbox("Raw: Address Line 1 column", options=raw_cols, index=min(0, len(raw_cols)-1), key="re_l1")
        raw_line2_col = st.selectbox("Raw: Address Line 2 column (optional)", options=["<none>"] + raw_cols, index=0, key="re_l2")
        cor_addr_col = st.selectbox("Correct: Address column", options=cor_cols, index=min(0, len(cor_cols)-1), key="re_coraddr")

        id_col_options = ["<none>"] + [c for c in raw_cols if c in cor_cols]
        id_col = st.selectbox("Optional: ID column present in BOTH files (for joining)", options=id_col_options, index=0, key="re_id")

        raw_df = raw_df.head(int(max_rows)).copy()
        cor_df = cor_df.head(int(max_rows)).copy()

        # Matching summary (normalized IDs to match join behavior)
        if id_col != "<none>":
            raw_df["_id_norm"] = normalize_id_series(raw_df[id_col])
            cor_df["_id_norm"] = normalize_id_series(cor_df[id_col])

            if sort_by_id:
                raw_df = raw_df.sort_values("_id_norm").reset_index(drop=True)
                cor_df = cor_df.sort_values("_id_norm").reset_index(drop=True)

            raw_ids = set(raw_df["_id_norm"].dropna().tolist())
            cor_ids = set(cor_df["_id_norm"].dropna().tolist())
            matched_ids = raw_ids & cor_ids
            only_raw = sorted(list(raw_ids - cor_ids))
            only_cor = sorted(list(cor_ids - raw_ids))

            c1, c2, c3 = st.columns(3)
            c1.metric("Matched IDs", len(matched_ids))
            c2.metric("Only in RAW", len(only_raw))
            c3.metric("Only in CORRECT", len(only_cor))

            if only_raw or only_cor:
                st.info(
                    "Unmatched examples — RAW only: "
                    + ", ".join(map(str, only_raw[:10]))
                    + (" …" if len(only_raw) > 10 else "")
                    + " | CORRECT only: "
                    + ", ".join(map(str, only_cor[:10]))
                    + (" …" if len(only_cor) > 10 else "")
                )
                or_buf = io.StringIO(); pd.DataFrame({id_col: only_raw}).to_csv(or_buf, index=False); or_buf.seek(0)
                oc_buf = io.StringIO(); pd.DataFrame({id_col: only_cor}).to_csv(oc_buf, index=False); oc_buf.seek(0)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ IDs only in RAW (CSV)", data=or_buf.getvalue(), file_name="ids_only_in_raw.csv", mime="text/csv")
                d2.download_button("⬇️ IDs only in CORRECT (CSV)", data=oc_buf.getvalue(), file_name="ids_only_in_correct.csv", mime="text/csv")
        else:
            matched_n = min(len(raw_df), len(cor_df))
            extra_raw = max(0, len(raw_df) - matched_n)
            extra_cor = max(0, len(cor_df) - matched_n)
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows aligned by order", matched_n)
            c2.metric("Extra in RAW", extra_raw)
            c3.metric("Extra in CORRECT", extra_cor)

        # Build RawAddress
        if raw_line2_col != "<none>" and raw_line2_col in raw_df.columns:
            raw_df["RawAddress"] = raw_df.apply(
                lambda r: f"{r[raw_line1_col]} {r[raw_line2_col]}".strip() if pd.notnull(r[raw_line2_col]) and str(r[raw_line2_col]).strip() != "" else str(r[raw_line1_col]),
                axis=1,
            )
        else:
            raw_df["RawAddress"] = raw_df[raw_line1_col].astype(str)

        # Ensure normalized IDs exist before processing for keyed predictions
        if id_col != "<none>":
            raw_df["_id_norm"] = normalize_id_series(raw_df[id_col])
            cor_df["_id_norm"] = normalize_id_series(cor_df[id_col])

        st.subheader("Processing")
        placeholder = st.empty()
        results = []
        total = len(raw_df)
        for i, raw in enumerate(raw_df["RawAddress"].astype(str).tolist(), start=1):
            interim = rule_clean(raw)
            pred = llm_correct(interim, compounds) if use_llm else interim
            pred = re.sub(r"\s+", " ", pred).strip()
            results.append(pred)
            placeholder.progress(min(i / max(1, total), 1.0))

        # Predictions keyed by normalized ID when available
        if id_col != "<none>":
            pred_df = pd.DataFrame({"_id_norm": raw_df["_id_norm"].tolist(), "PredictedAddress": results})
        else:
            pred_df = pd.DataFrame({"PredictedAddress": results})

        # Align and compare
        if id_col != "<none>":
            base_cols = [id_col, "_id_norm", "RawAddress"]
            combo = (
                raw_df[base_cols]
                  .reset_index(drop=True)
                  .merge(
                      cor_df[[id_col, "_id_norm", cor_addr_col]].reset_index(drop=True),
                      on="_id_norm",
                      how="inner",
                      suffixes=("_RAW", "_COR")
                  )
            )
            combo = combo.merge(pred_df, on="_id_norm", how="left")
            combo = combo.rename(columns={cor_addr_col: "CorrectAddress"})
            if sort_by_id:
                combo = combo.sort_values("_id_norm").reset_index(drop=True)
            st.caption(f"Joined rows used for accuracy: {len(combo)}")
        else:
            matched_n = min(len(raw_df), len(cor_df))
            raw_sub = raw_df.head(matched_n)
            cor_sub = cor_df.head(matched_n)
            combo = pd.concat([
                raw_sub[["RawAddress"]].reset_index(drop=True),
                cor_sub[[cor_addr_col]].rename(columns={cor_addr_col: "CorrectAddress"}).reset_index(drop=True),
                pred_df.head(matched_n).reset_index(drop=True),
            ], axis=1)
            st.caption(f"Rows aligned by order and used for accuracy: {matched_n}")

        # Canonical comparison
        combo["_pred_norm"] = combo["PredictedAddress"].apply(canonicalize_for_compare)
        combo["_corr_norm"] = combo["CorrectAddress"].apply(canonicalize_for_compare)
        combo["ExactMatch"] = combo["_pred_norm"] == combo["_corr_norm"]

        accuracy = (combo["ExactMatch"].sum() / max(1, len(combo))) * 100.0
        st.success(f"✅ Accuracy: {accuracy:.2f}% (exact normalized match)")

        # Outputs (include clean ID when available)
        if id_col != "<none>":
            combo["ID"] = combo["_id_norm"]

        # Preview
        st.subheader("Results Preview")
        preview_cols = [c for c in ["ID" if "ID" in combo.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress", "ExactMatch"] if c in combo.columns]
        st.dataframe(combo[preview_cols].head(50), use_container_width=True)

        # Non-exact matches
        non_matches = combo.loc[~combo["ExactMatch"]].copy()
        st.subheader(f"❗ Non-exact matches: {len(non_matches)}")
        if len(non_matches) > 0:
            nm_cols = [c for c in ["ID" if "ID" in non_matches.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress"] if c in non_matches.columns]
            st.dataframe(non_matches[nm_cols].head(25), use_container_width=True)
            nm_buf = io.StringIO(); non_matches[nm_cols].to_csv(nm_buf, index=False)
            st.download_button("⬇️ Download Non-Exact Matches (CSV)", data=nm_buf.getvalue(), file_name="address_non_exact_matches.csv", mime="text/csv")

        # Full evaluation CSV
        csv_buf = io.StringIO()
        out_cols = [c for c in ["ID" if "ID" in combo.columns else None, "RawAddress", "PredictedAddress", "CorrectAddress", "ExactMatch"] if c in combo.columns]
        combo[out_cols].to_csv(csv_buf, index=False)
        st.download_button("📥 Download Evaluation Results (CSV)", data=csv_buf.getvalue(), file_name="address_evaluation_results.csv", mime="text/csv")

# =========================
# PRODUCTION MODE
# =========================
else:
    st.header("🚀 Production: Generate Corrected Addresses CSV")
    raw_file_p = st.file_uploader("Upload RAW addresses (CSV)", type=["csv"], key="raw_prod")
    if raw_file_p:
        raw_df = pd.read_csv(raw_file_p)
        raw_cols = list(raw_df.columns)

        st.subheader("Select Columns")
        raw_line1_col = st.selectbox("Raw: Address Line 1 column", options=raw_cols, index=min(0, len(raw_cols)-1), key="rp_l1")
        raw_line2_col = st.selectbox("Raw: Address Line 2 column (optional)", options=["<none>"] + raw_cols, index=0, key="rp_l2")
        id_col = st.selectbox("Optional: ID column (e.g., CWID)", options=["<none>"] + raw_cols, index=0, key="rp_id")

        raw_df = raw_df.head(int(max_rows)).copy()

        # Build RawAddress
        if raw_line2_col != "<none>" and raw_line2_col in raw_df.columns:
            raw_df["RawAddress"] = raw_df.apply(
                lambda r: f"{r[raw_line1_col]} {r[raw_line2_col]}".strip() if pd.notnull(r[raw_line2_col]) and str(r[raw_line2_col]).strip() != "" else str(r[raw_line1_col]),
                axis=1,
            )
        else:
            raw_df["RawAddress"] = raw_df[raw_line1_col].astype(str)

        # Normalize ID for clean output
        if id_col != "<none>":
            raw_df["ID"] = normalize_id_series(raw_df[id_col])

        st.subheader("Processing")
        placeholder = st.empty()
        results = []
        total = len(raw_df)
        for i, raw in enumerate(raw_df["RawAddress"].astype(str).tolist(), start=1):
            interim = rule_clean(raw)
            pred = llm_correct(interim, compounds) if use_llm else interim
            pred = re.sub(r"\s+", " ", pred).strip()
            results.append(pred)
            placeholder.progress(min(i / max(1, total), 1.0))

        # Build output CSV (ID + CorrectedAddress, plus Raw if helpful)
        out_df = pd.DataFrame({"CorrectedAddress": results})
        if id_col != "<none>":
            out_df = pd.concat([raw_df[["ID"]].reset_index(drop=True), out_df], axis=1)
        out_df = pd.concat([out_df, raw_df[["RawAddress"]].reset_index(drop=True)], axis=1)

        st.subheader("Preview (first 50)")
        st.dataframe(out_df.head(50), use_container_width=True)

        csv_buf = io.StringIO(); out_df.to_csv(csv_buf, index=False)
        st.download_button("⬇️ Download Corrected Addresses (CSV)", data=csv_buf.getvalue(), file_name="corrected_addresses.csv", mime="text/csv")

    st.caption("In Production mode, we don't need the ground-truth file. The app outputs the corrected address line per record using the Rules → LLM pipeline.")
