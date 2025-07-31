import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import openai
import re
import os
import time
import io
from openai import RateLimitError

# === CONFIGURATION ===
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Known compound street names to preserve
compound_names = [
    "northwood", "southgate", "eastwood", "westfield",
    "northridge", "southridge", "eastview", "westpark"
]

# === RULE-BASED CLEANUP ===
def clean_address(address):
    if pd.isnull(address):
        return ""

    address = address.lower()

    # Preserve compound names before abbreviation
    for name in compound_names:
        address = re.sub(rf'\b{name}\b', name.replace(" ", ""), address)

    # Remove special characters
    address = re.sub(r'[.,#]', '', address)

    # Directionals
    directionals = {
        'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
        'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw'
    }
    for word, abbr in directionals.items():
        address = re.sub(rf'\b{word}\b', abbr, address)

    # Street types
    street_types = {
        'avenue': 'ave', 'av': 'ave', 'aven': 'ave', 'avenu': 'ave', 'avnue': 'ave',
        'boulevard': 'blvd', 'blvd': 'blvd', 'boul': 'blvd', 'boulv': 'blvd',
        'street': 'st', 'str': 'st', 'stret': 'st', 'strt': 'st',
        'drive': 'dr', 'driv': 'dr', 'drv': 'dr',
        'road': 'rd', 'rd': 'rd',
        'lane': 'ln', 'ln': 'ln',
        'terrace': 'ter', 'terr': 'ter',
        'place': 'pl', 'pl': 'pl',
        'court': 'ct', 'ct': 'ct',
        'circle': 'cir', 'circl': 'cir', 'circ': 'cir',
        'parkway': 'pkwy', 'pkway': 'pkwy', 'parkwy': 'pkwy', 'pkwy': 'pkwy', 'pky': 'pkwy',
        'junction': 'jct', 'jctn': 'jct', 'junctn': 'jct', 'junctions': 'jcts',
        'mount': 'mt', 'mountain': 'mtn', 'mountin': 'mtn',
        'heights': 'hts', 'highway': 'hwy', 'expressway': 'expy'
    }
    for word, abbr in street_types.items():
        address = re.sub(rf'\b{word}\b', abbr, address)

    # Apt/Suite normalization
    address = re.sub(r'\b(apartment|apt)\b', 'apt', address)
    address = re.sub(r'\b(ste|suite)\b', 'ste', address)
    address = re.sub(r'\b(room|rm)\b', 'rm', address)
    address = re.sub(r'\b(floor|fl)\b', 'apt', address)

    # PO Box normalization
    address = re.sub(r'\b(pobox|pob|po box|po#|box)\s*(\w+)', r'PO Box \2', address)

    address = re.sub(r'\s+', ' ', address).strip()
    return address.title()

# === LLM CORRECTION ===
def correct_with_llm(prompt_address):
    system_prompt = """
    You are a data formatting assistant. Format the following U.S. address line strictly according to institutional standards:
    - Abbreviate directions (North → N, Southwest → SW, etc.)
    - Use USPS-style street abbreviations (e.g., St, Ave, Blvd, etc.)
    - Format PO Boxes as "PO Box ###"
    - Normalize unit designators (Apt, Ste, Rm)
    - Remove special characters (e.g., #, ., ,)
    - Capitalize appropriately
    - Ensure no trailing or multiple spaces exist
    - Do NOT separate compound names like Northwood, Southridge, etc.
    Return the corrected address as a single line only.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_address}
            ],
            temperature=0.3,
        )
        result = response.choices[0].message.content.strip()
        return re.sub(r'\s+', ' ', result)
    except RateLimitError:
        time.sleep(5)
        return correct_with_llm(prompt_address)
    except Exception as e:
        return f"[LLM ERROR] {str(e)}"

# === STREAMLIT UI ===
st.title("📬 Address Cleaner & Formatter")
st.write("Upload a dataset with `AddrLine1` and `AddrLine2`. The app will return corrected address lines.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.head(100)
    df['RawAddress'] = df.apply(
        lambda row: f"{row['AddrLine1']} {row['AddrLine2']}".strip()
        if pd.notnull(row['AddrLine2']) else row['AddrLine1'], axis=1
    )

    df['Cleaned'] = df['RawAddress'].apply(clean_address)

    st.subheader("📄 Original vs. Corrected Address Line (Live)")
    output_placeholder = st.empty()
    results = []

    for i, raw in enumerate(df['RawAddress']):
        corrected = correct_with_llm(raw)
        results.append({"RawAddress": raw, "CorrectedAddress": corrected})
        temp_df = pd.DataFrame(results)
        output_placeholder.dataframe(temp_df, use_container_width=True)

    final_df = pd.DataFrame(results)
    st.success("✅ Address line correction complete!")

    excel_buffer = io.BytesIO()
    final_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)

    st.download_button(
        label="📥 Download Corrected Addresses",
        data=excel_buffer,
        file_name="corrected_addresses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
