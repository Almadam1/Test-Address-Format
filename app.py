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
        # protect canonical and split variants
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
        # restore protected token
        out = re.sub(rf"\b{re.escape(token)}\b", canonical, out, flags=re.IGNORECASE)
        # also repair splits like "North Wood"/"N Wood" -> "Northwood"
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
    s = str(address).lower()

    # Remove special characters
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
        "road": "rd",
        "lane": "ln",
        "terrace": "ter", "terr": "ter",
        "place": "pl",
        "court": "ct",
        "circle": "cir", "circl": "cir", "circ": "cir",
        "parkway": "pkwy", "pkway": "pkwy", "parkwy": "pkwy", "pky": "pkwy",
        "junction": "jct", "jctn": "jct", "junctn": "jct", "junctions": "jcts",
        "mount": "mt", "mountain": "mtn", "mountin": "mtn",
        "heights": "hts", "highway": "hwy", "expressway": "expy",
    }
    for word, abbr in street_types.items():
        s = re.sub(rf"\b{word}\b", abbr, s)

    # Units
    s = re.sub(r"\b(apartment|apt)\b", "apt", s)
    s = re.sub(r"\b(ste|suite)\b", "ste", s)
    s = re.sub(r"\b(room|rm)\b", "rm", s)
    s = re.sub(r"\b(floor|fl)\b", "apt", s)

    # PO Box
    s = re.sub(r"\b(pobox|pob|po box|po#|box)\s*(\w+)", r"PO Box \2", s, flags=re.IGNORECASE)

    s = re.sub(r"\s+", " ", s).strip()
    return s.title()

# =========================
# LLM correction (address line only)
# =========================
def llm_correct(address_line: str, compounds: list[str]) -> str:
    # Pre-protect compounds
    pre = protect_compounds(address_line, compounds)

    sys_prompt = (
        "You are a data formatting assistant. "
        "Format the following U.S. address LINE ONLY (no city/state/ZIP). "
        "Rules: abbreviate directions (North→N, Southwest→SW); use USPS street abbreviations (St, Ave, Blvd, Rd, Dr, Ln, Ter, Pl, Ct, Cir, Pkwy, Hwy, etc.); "
        "format PO Boxes as 'PO Box ###'; normalize unit designators (Apt, Ste, Rm); remove special characters (# . ,); "
        "title case words; ensure single spaces only; and DO NOT split compound street names like Northwood/Eastwood/Mount Carmel. "
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

    # Post-repair compounds and whitespace
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
    # Remove punctuation we don't care about; collapse spaces
    s = re.sub(r"[.,#]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

# =========================
# Streamlit UI
# =========================
st.title("📬 Address Cleaner & Evaluator (CSV-only)")
st.write("Upload **raw addresses** and **correct addresses** as CSV files, run the formatter, and get accuracy metrics.")

# Sidebar options
st.sidebar.header("Options")
max_rows = st.sidebar.number_input("Max rows to process", min_value=1, max_value=5000, value=DEFAULT_MAX_ROWS, step=1)
use_llm = st.sidebar.checkbox("Use LLM enhancement (after rules)", value=True)
compound_extra = st.sidebar.text_area("Extra compound street names to preserve (comma-separated)", value="")

compounds = DEFAULT_COMPOUNDS.copy()
if compound_extra.strip():
    compounds.extend([c.strip().lower() for c in compound_extra.split(",") if c.strip()])

# File uploaders (CSV ONLY)
raw_file = st.file_uploader("Upload RAW addresses (CSV)", type=["csv"], key="raw")
correct_file = st.file_uploader("Upload CORRECT addresses (CSV)", type=["csv"], key="correct")

if raw_file and correct_file:
    # Read CSVs
    raw_df = pd.read_csv(raw_file)
    cor_df = pd.read_csv(correct_file)

    # Column selection UI
    st.subheader("Select Columns")
    raw_cols = list(raw_df.columns)
    cor_cols = list(cor_df.columns)

    raw_line1_col = st.selectbox("Raw: Address Line 1 column", options=raw_cols, index=min(0, len(raw_cols)-1))
    raw_line2_col = st.selectbox("Raw: Address Line 2 column (optional)", options=["<none>"] + raw_cols, index=0)
    cor_addr_col = st.selectbox("Correct: Address column", options=cor_cols, index=min(0, len(cor_cols)-1))

    id_col_options = ["<none>"] + [c for c in raw_cols if c in cor_cols]
    id_col = st.selectbox("Optional: ID column present in BOTH files (for joining)", options=id_col_options, index=0)

    # Limit rows for testing (limit independently so unmatched counts reflect within-cap)
    raw_df = raw_df.head(int(max_rows))
    cor_df = cor_df.head(int(max_rows))

    # --- Matching summary banner ---
    if id_col != "<none>":
        raw_ids = set(raw_df[id_col].dropna().astype(str))
        cor_ids = set(cor_df[id_col].dropna().astype(str))
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
            # Downloads for unmatched ID lists (CSV)
            or_buf = io.StringIO()
            pd.DataFrame({id_col: only_raw}).to_csv(or_buf, index=False)
            or_buf.seek(0)
            oc_buf = io.StringIO()
            pd.DataFrame({id_col: only_cor}).to_csv(oc_buf, index=False)
            oc_buf.seek(0)
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

    # Process RawAddress → Predicted
    st.subheader("Processing")
    placeholder = st.empty()
    results = []
    for raw in raw_df["RawAddress"].astype(str).tolist():
        # Apply rules first
        interim = rule_clean(raw)
        # Optionally LLM enhance
        if use_llm:
            pred = llm_correct(interim, compounds)
        else:
            pred = interim
        # Final trim
        pred = re.sub(r"\s+", " ", pred).strip()
        results.append(pred)
        placeholder.progress(min(len(results) / max(1, len(raw_df)), 1.0))

    pred_df = pd.DataFrame({"PredictedAddress": results})

    # Align and compare
    if id_col != "<none>":
        combo = (
            raw_df[[id_col, "RawAddress"]]
            .reset_index(drop=True)
            .merge(cor_df[[id_col, cor_addr_col]], on=id_col, how="inner")
        )
        combo = pd.concat([combo.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    else:
        matched_n = min(len(raw_df), len(cor_df))
        raw_sub = raw_df.head(matched_n)
        cor_sub = cor_df.head(matched_n)
        combo = pd.concat([
            raw_sub[["RawAddress"]].reset_index(drop=True),
            cor_sub[[cor_addr_col]].rename(columns={cor_addr_col: "CorrectAddress"}).reset_index(drop=True),
            pred_df.head(matched_n).reset_index(drop=True),
        ], axis=1)

    # Canonical comparison
    corr_col_name = "CorrectAddress" if "CorrectAddress" in combo.columns else cor_addr_col
    combo["_pred_norm"] = combo["PredictedAddress"].apply(canonicalize_for_compare)
    combo["_corr_norm"] = combo[corr_col_name].apply(canonicalize_for_compare)
    combo["ExactMatch"] = combo["_pred_norm"] == combo["_corr_norm"]

    accuracy = (combo["ExactMatch"].sum() / max(1, len(combo))) * 100.0

    st.success(f"✅ Accuracy: {accuracy:.2f}% (exact normalized match)")

    # Show sample and full table toggles
    st.subheader("Results Preview")
    st.dataframe(combo[[c for c in ["RawAddress", "PredictedAddress", corr_col_name, "ExactMatch"] if c in combo.columns]].head(50), use_container_width=True)

    # Download full results as CSV
    csv_buf = io.StringIO()
    out_cols = [col for col in ["RawAddress", "PredictedAddress", corr_col_name, "ExactMatch"] if col in combo.columns]
    combo[out_cols].to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Evaluation Results (CSV)",
        data=csv_buf.getvalue(),
        file_name="address_evaluation_results.csv",
        mime="text/csv",
    )
