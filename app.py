import streamlit as st
import pandas as pd
from rapidfuzz import fuzz, process
import openai
import re
import os

# === CONFIGURATION ===
openai.api_key = st.secrets["OPENAI_API_KEY"]  # Set this in Streamlit secrets

# === RULE-BASED CLEANUP ===
def clean_address(address):
    if pd.isnull(address):
        return ""

    address = address.lower()

    # Remove special characters
    address = re.sub(r'[.,#]', '', address)

    # Directionals
    directionals = {
        'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
        'northeast': 'ne', 'northwest': 'nw', 'southeast': 'se', 'southwest': 'sw'
    }
    for word, abbr in directionals.items():
        address = re.sub(rf'\b{word}\b', abbr, address)

    # Street types - expanded list based on Word doc (sample subset)
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
        # Extend as needed
    }
    for word, abbr in street_types.items():
        address = re.sub(rf'\b{word}\b', abbr, address)

    # Apt/Suite normalization
    address = re.sub(r'\b(apartment|apt)\b', 'apt', address)
    address = re.sub(r'\b(ste|suite)\b', 'ste', address)
    address = re.sub(r'\b(room|rm)\b', 'rm', address)
    address = re.sub(r'\b(floor|fl)\b', 'apt', address)  # floor normalized to apt

    # PO Box normalization
    address = re.sub(r'\b(pobox|pob|po box|po#|box)\s*(\w+)', r'PO Box \2', address)

    # Remove extra spaces
    address = re.sub(r'\s+', ' ', address).strip()

    # Title case
    return address.title()

# === LLM CORRECTION ===
def correct_with_llm(prompt_address):
    system_prompt = """
    Format the following mailing address according to strict database entry standards:
    - Abbreviate directions (North -> N, Southwest -> SW, etc.)
    - Use postal regulation street abbreviations (e.g., Ave, Blvd, St, etc.)
    - Write out cities and towns in full (e.g., New York City instead of NYC)
    - Use postal state abbreviations (e.g., NY, NJ, CA)
    - Include ZIP+4 when possible (e.g., 10458-1234)
    - Format PO boxes as 'PO Box 12345' or 'PO Box G'
    - Abbreviate apartment, suite, or room numbers as Apt, Ste, or Rm and place them on the second line
    - Remove special characters (e.g., # . ,)
    - Write out school and business names in full (e.g., Montclair State University instead of MSU)
    Return only the corrected mailing address as a single string.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_address}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# === STREAMLIT UI ===
st.title("📬 Address Cleaner & Formatter")
st.write("Upload a dataset with `AddrLine1` and `AddrLine2`. The app will return corrected addresses.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Combine lines
    df['RawAddress'] = df.apply(
        lambda row: f"{row['AddrLine1']} {row['AddrLine2']}".strip() 
        if pd.notnull(row['AddrLine2']) else row['AddrLine1'], axis=1
    )

    # Step 1: Rule-based cleaning
    df['Cleaned'] = df['RawAddress'].apply(clean_address)

    # Step 2: LLM cleanup for all addresses
    with st.spinner("Processing addresses with LLM..."):
        df['CorrectedAddress'] = df['RawAddress'].apply(correct_with_llm)

    st.success("✅ Address correction complete!")
    st.dataframe(df[['RawAddress', 'CorrectedAddress']])

    # Download
    st.download_button(
        label="📥 Download Corrected Addresses",
        data=df.to_excel(index=False, engine='openpyxl'),
        file_name="corrected_addresses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
