import streamlit as st
import fitz
import pandas as pd
import re
import openai
import json
import docx
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import plotly.graph_objects as go

@st.cache_resource
def ensure_punkt_tokenizer():
    import nltk
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters

    punkt_params = PunktParameters()
    
    abbrevs = [
        'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr',
        'inc', 'ltd', 'co', 'corp', 'llc', 'pvt',
        'vs', 'etc', 'e.g', 'i.e', 'viz', 'al', 'fig',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug',
        'sep', 'sept', 'oct', 'nov', 'dec', 'nos', 'vol', 'rev', 'ed',
        'st', 'no', 'dept'
    ]

    punkt_params.abbrev_types = set(abbrevs)
    tokenizer = PunktSentenceTokenizer(punkt_params)
    return tokenizer
    
# === OpenAI Key ===
openai.api_key = st.secrets["openai"]["api_key"]

# === UI Styling ===
st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: bold;
    color: #2C3E50;
    padding-bottom: 1rem;
}
.sub-section {
    background-color: #FAFAFA;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    box-shadow: 0 0 8px rgba(0,0,0,0.05);
}
.highlight {
    background-color: #D6EAF8;
    padding: 0.6rem;
    border-radius: 8px;
    color: #154360;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# === Cache ===
@st.cache_data
def load_personas():
    with open("personas.json") as f:
        return json.load(f)

@st.cache_resource
def load_finbert():
    tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tok, mdl

@st.cache_data(show_spinner=False)
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return "\n".join([p.get_text() for p in fitz.open(stream=uploaded_file.read(), filetype="pdf")])
    elif uploaded_file.name.endswith(".docx"):
        return "\n".join([p.text for p in docx.Document(uploaded_file).paragraphs])
    else:
        return uploaded_file.read().decode("utf-8")

# === Sentiment Functions ===
def generate_prompt(persona, sentence, finbert_scores):
    tone_desc = f"Tone analysis suggests this sentence is perceived as:\n" \
                f"- Negative: {finbert_scores['negative'] * 100:.1f}%\n" \
                f"- Neutral: {finbert_scores['neutral'] * 100:.1f}%\n" \
                f"- Positive: {finbert_scores['positive'] * 100:.1f}%\n"

    return f"""
You are {persona['name']}, a {persona['investment_style']} investor with a {persona['risk_tolerance']} risk tolerance and a {persona['investment_horizon']} horizon.

Your tone is {persona['tone_preference']}. You prioritize: {', '.join(persona['focus_areas'])}.
Your typical concerns are: {', '.join(persona['typical_concerns'])}.
You get particularly triggered by: {', '.join(persona['sentiment_triggers'])}.

Below is a sentence from an annual report. Tone analysis is provided too. Read carefully and interpret from your viewpoint.

---
📄 Sentence:
\"\"\"{sentence}\"\"\"

{tone_desc}

---
Now respond in this exact JSON format (no free text):

{{
  "viewpoint": "Interpretation from your investor perspective, in your tone.",
  "risk_level": "High / Medium / Low — from your risk lens.",
  "rationale": "Explain WHY it matters to your investment thesis. End with: 'Would I invest based on this? Yes/No, and why.'"
}}
"""
def get_llm_sentiment(persona, sentence, finbert_scores):
    try:
        prompt = generate_prompt(persona, sentence, finbert_scores)
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600
        )
        raw = response.choices[0].message.content
        json_start = raw.find('{')
        json_end = raw.rfind('}')
        if json_start != -1 and json_end != -1:
            clean_json = raw[json_start:json_end+1]
            parsed = json.loads(clean_json)
            parsed["risk_level"] = parsed.get("risk_level", "Medium").strip().capitalize()
            return parsed
        else:
            raise ValueError("JSON structure not found in LLM response.")
    except Exception as e:
        return {
            "viewpoint": "❌ Error generating response from LLM.",
            "risk_level": "Medium",
            "rationale": f"LLM error: {str(e)}"
        }        
def get_finbert_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    labels = ['negative', 'neutral', 'positive']
    return {label: round(float(prob), 4) for label, prob in zip(labels, probs)}

def calculate_risk(finbert, llm):
    penalty = 0.4 if "negative" in llm.lower() else 0
    return round(min(finbert['negative'] + penalty, 1.0), 2)

# === Main App ===
def main():
    st.image("https://raw.githubusercontent.com/vaishnavipj/Sentiment/main/Screenshot%202025-07-17%20004956.png", width=240)
    st.markdown("<div class='main-title'>🗣️ VOICEIQ - Intelligence for your corporate voice</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Upload Annual Report (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

    if uploaded:
        text = extract_text(uploaded)
        st.success("✅ File Uploaded & Extracted")
        sentence_tokenizer = ensure_punkt_tokenizer()
        sentences = sentence_tokenizer.tokenize(text)
        personas = load_personas()
        finbert_tokenizer, model = load_finbert()

        tab1, tab2, tab3 = st.tabs(["🧠 Investor Sentiment", "📋 Compliance Check", "🔁 Redundancy"])

        # --- TAB 1: INVESTOR SENTIMENT ---
        with tab1:
            subtab1, subtab2, subtab3 = st.tabs(["📄 Sentence-based", "📚 Topic-based", "📊 Overall Sentiment"])
            # === SUBTAB 1: Sentence-Based ===
            with subtab1:
                # st.markdown("<div class='sub-section'>Sentiment analysis - Persona Based</div>", unsafe_allow_html=True)
                selected = st.multiselect("🎯 Select Personas", [p['name'] for p in personas], default=[p['name'] for p in personas])
                show_all = st.checkbox("Show all sentences (not just risky ones)", value=False)

                if st.button("▶️ Run Sentence-Based Sentiment Analysis"):
                    selected_personas = [p for p in personas if p['name'] in selected]
                    if not selected_personas:
                        st.warning("⚠️ Please select at least one persona.")
                    else:
                        progress = st.progress(0.0)
                        for i, persona in enumerate(selected_personas):
                            st.markdown(f"<div class='highlight'>👤 <strong>{persona['name']}</strong></div>", unsafe_allow_html=True)
                            st.markdown(f"🧬 <i>{persona['bio']}</i>", unsafe_allow_html=True)
                            st.markdown(f"🔎 <strong>Investment Style:</strong> {persona['investment_style']}  \n📌 <strong>Focus Areas:</strong> {', '.join(persona['focus_areas'])}", unsafe_allow_html=True)

                            low, med, high = 0, 0, 0

                            for sent in sentences[:20]:
                                finbert = get_finbert_sentiment(sent, finbert_tokenizer, model)
                                llm_result = get_llm_sentiment(persona, sent, finbert)
                                risk_level = llm_result.get("risk_level", "Medium").capitalize()
                                if risk_level == "High": high += 1
                                elif risk_level == "Low": low += 1
                                else: med += 1

                                if show_all or risk_level in ["High", "Medium"]:
                                    with st.expander(f"💬 Sentence Review –"):
                                        st.markdown(f"**📝 Sentence:** {sent}")
                                        st.markdown(f"**📊 Risk Probability for {persona['name']}:**")
                                        finbert_df = pd.DataFrame([finbert]).T.rename(columns={0: "Probability"})
                                        finbert_df["Probability"] = (finbert_df["Probability"] * 100).round(2).astype(str) + " %"
                                        st.dataframe(finbert_df, use_container_width=True)

                                        st.markdown("🧠 <strong>Persona’s Interpretation:</strong>", unsafe_allow_html=True)
                                        st.markdown(f"<div class='sub-section'>{llm_result.get('viewpoint', 'N/A')}</div>", unsafe_allow_html=True)

                                        st.markdown("📌 <strong>Why this matters to the persona:</strong>", unsafe_allow_html=True)
                                        st.info(llm_result.get("rationale", "No rationale provided."))

                            chart_df = pd.DataFrame({"Risk Level": ["Low", "Medium", "High"], "Count": [low, med, high]})
                            fig = go.Figure(data=[go.Bar(x=chart_df["Risk Level"], y=chart_df["Count"], marker_color=["green", "orange", "red"])])
                            fig.update_layout(title="📈 Risk Distribution Across Sentences", xaxis_title="Risk Level", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                            progress.progress((i + 1) / len(selected_personas))

            # === SUBTAB 2: Topic-Based ===
            with subtab2:
                selected_personas = st.multiselect("🎯 Select Personas for Topic Analysis", [p['name'] for p in personas], default=[p['name'] for p in personas])

                if st.button("▶️ Run Topic-Based Sentiment Analysis"):
                    with st.spinner("🔍 Extracting topics and analyzing risks..."):
                        try:
                            persona_insights = "\n".join([
                                f"Persona: {p['name']}\nTone: {p['tone_preference']}\nFocus: {', '.join(p['focus_areas'])}\nTriggers: {', '.join(p['sentiment_triggers'])}"
                                for p in personas if p['name'] in selected_personas
                            ])
                            topic_prompt = f"""
You are a financial analyst helping investor personas interpret an annual report.

1. Read the following annual report text.
2. Extract 8–12 clear, distinct strategic or operational topics from the report.
3. For each topic, return:
   - "topic": Clear, specific title (avoid generic headings)
   - "summary": 2–3 sentence summary tailored to investor interest
   - focus_area": Choose the best fit from the following:
     ['Profitability', 'Risk', 'Growth', 'Compliance', 'Governance', 'Innovation', 'Sustainability', 'Liquidity', 'Leverage', 'Operational Efficiency', 'Customer Acquisition', 'Digital Transformation', 'Market Share', 'Regulatory Outlook', 'Capital Allocation', 'Human Capital']
   - "top_sentences": 5 important lines that carry high tone or disclosure weight

Incorporate context from these selected investor personas:
{persona_insights}

Ensure the focus_area clearly reflects what matters most to investors.
Only respond in the following JSON format:
[
  {{
    "topic": "...",
    "summary": "...",
    "focus_area": "...",
    "top_sentences": ["...", "...", "...", "...", "..."]
  }}, ...
]

Text:
{text[:6500]}
"""
                            response = openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": topic_prompt}],
                                temperature=0.3,
                                max_tokens=1600
                            )
                            topic_json = json.loads(re.search(r"\[.*\]", response.choices[0].message.content, re.DOTALL).group(0))
                        except Exception as e:
                            st.error(f"❌ Error extracting topics: {str(e)}")
                            return

                        for persona in [p for p in personas if p['name'] in selected_personas]:
                            st.markdown(f"<div class='highlight'>👤 <strong>{persona['name']}</strong></div>", unsafe_allow_html=True)
                            for topic in topic_json:
                                st.markdown(f"### 🧩 Topic: {topic['topic']}")
                                st.markdown(f"**🔍 Summary:** {topic['summary']}")
                                st.markdown(f"**📌 Focus Area:** `{topic['focus_area']}`")

                                topic_text = " ".join(topic['top_sentences'])
                                finbert = get_finbert_sentiment(topic_text, finbert_tokenizer, model)
                                llm_result = get_llm_sentiment(persona, topic_text, finbert)

                                st.markdown("📊 **Probability Breakdown:**")
                                score_df = pd.DataFrame([finbert]).T.rename(columns={0: "Probability"})
                                score_df["Probability"] = (score_df["Probability"] * 100).round(2).astype(str) + "%"
                                st.dataframe(score_df, use_container_width=True)

                                st.markdown("📌 <strong>Why this section matters to the persona:</strong>", unsafe_allow_html=True)
                                st.info(llm_result.get("rationale", "No rationale provided."))

                                st.markdown("🧠 <strong>Persona’s Interpretation:</strong>", unsafe_allow_html=True)
                                st.markdown(f"<div class='sub-section'>{llm_result.get('viewpoint', 'N/A')}</div>", unsafe_allow_html=True)

                                st.markdown("### 📊 Top Contributing Sentences")
                                for i, sent in enumerate(topic['top_sentences']):
                                    st.markdown(f"{i+1}. {sent}")

                                fig = go.Figure(data=[
                                    go.Bar(x=list(finbert.keys()), y=[v * 100 for v in finbert.values()], marker_color=["red", "orange", "green"])
                                ])
                                fig.update_layout(
                                    title=f"🎯 Risk Sentiment Distribution for Topic: {topic['topic']}",
                                    xaxis_title="Sentiment", yaxis_title="Probability (%)",
                                    yaxis=dict(range=[0, 100])
                                )
                                st.plotly_chart(fig, use_container_width=True)
            with subtab3:
                if st.button("🧠 Generate Overall Investor Summary"):
                    with st.spinner("Analyzing investor personas and generating executive summary..."):
                        try:
                            persona_summary = "\n".join([
                                f"Persona: {p['name']}\nTone: {p['tone_preference']}\nFocus: {', '.join(p['focus_areas'])}\nBio: {p['bio']}\nTriggers: {', '.join(p['sentiment_triggers'])}" for p in personas
                            ])

                            summary_prompt = f"""
You are a capital markets expert generating a structured, executive-level **Investor Sentiment Report** based on an uploaded **Annual Report** and predefined **Investor Persona Profiles**.

---

## 🔍 OBJECTIVE:
Thoroughly analyze the uploaded report content and persona profiles to:
1. Extract the overall **sentiment**, **narrative tone**, and **perceived risks**.
2. Generate tailored insights and content recommendations for each investor persona.
3. Present the entire output in clean, well-structured **Markdown format**.

---

## 📦 OUTPUT FORMAT GUIDELINES:
Your output **must strictly follow** the markdown structure provided below.
- Use **markdown tables** for summaries.
- Use bullet points where specified.
- Use emojis for section titles as indicated.
- Ensure all table columns are filled; use concise yet specific phrases.
- Do not generate long text blocks — use structured outputs only.

---

## 📑 REQUIRED OUTPUT SECTIONS:

### 📈 Overall Sentiment
Provide a high-level snapshot:
- **Sentiment Score** (e.g., +0.44)
- **Tone**: A comma-separated list of tone descriptors (e.g., Visionary, Risk-Averse, ESG-Forward, Tech-Focused)

---

### 🎯 Narrative Tone
List **3–6 bullet points** describing the report's tone.
Examples:  
- Visionary  
- Data-Rich  
- AI-Driven  
- ESG-Aware

---

### ⚠️ Material Gaps Identified
Highlight specific risk or gap statements in this table:

| **Statement** | **Risk Category** | **Severity Flag** |
|---------------|--------------------|--------------------|
| e.g., "Remuneration not tied to ESG metrics" | Governance Risk | 🚩 |

---

### 👥 Persona-Based Summary Table
Summarize how each persona is likely to interpret the report content:

| **Persona** | **Sentiment Score** | **Emotional Triggers** | **Risk Flags** | **Tone Fit** | **Suggested Narrative Adjustment** |
|-------------|----------------------|--------------------------|----------------|--------------|------------------------------------|

> Each row should reflect persona alignment, gaps, and customized adjustments.

---

### 📊 Sentiment by Section Table
Break down sentiment tone per major section:

| **Section** | **Sentiment Tone** | **Keywords Triggering Emotion** | **Persona Impact** |
|-------------|--------------------|----------------------------------|---------------------|
| e.g., Chairman’s Letter | Visionary | “civilizational shift,” “AI-first” | All |

---

### 💡 Recommendations Table
Offer concrete narrative/content enhancements per persona:

| **Persona** | **Addition to Improve Sentiment** | **Why It Matters** |
|-------------|-----------------------------------|---------------------|

---

### 🧠 Emotion Summary Chart (use bullet points or markdown table)
Summarize emotional tone distribution across the document:

- **Optimism** – XX% (Source: e.g., CEO Letter, Customer Innovation)
- **Confidence** – XX% (Source: e.g., Financials, AI Vision)
- **Caution** – XX% (Source: e.g., ESG Targets, Scope 3)
- **Neutrality** – XX% (Source: Governance, Reporting)
- **Skepticism** – XX% (Source: Pay, Offsets, Biodiversity)

---

### 📌 Final Summary
End with a concise, executive-level summary.

#### ✔ Strengths
- Bullet 1  
- Bullet 2  
- Bullet 3  
- Bullet 4  

#### ❌ Gaps
- Bullet 1  
- Bullet 2  
- Bullet 3  
- Bullet 4  

#### 🔎 Persona Confidence Level:
List confidence (High / Medium / Low) for each persona.

---

### 📘 CONTEXT INPUTS:

**Annual Report Text**
{text[:7000]}

**Investor Persona Profiles:**
{persona_summary}
---
Please return the full response in **Markdown format only**.
"""
                            response = openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": summary_prompt}],
                                temperature=0.3,
                                max_tokens=3200
                            )
                            st.markdown(response.choices[0].message.content)
                        except Exception as e:
                            st.error(f"❌ Error generating executive summary: {str(e)}")                    

                # --- Tab 2: Compliance Check ---
        with tab2:
            st.markdown("<div class='sub-section'>📋 Compliance report against SEBI & Companies Act</div>", unsafe_allow_html=True)

            def fallback_checklist():
                csv_text = """Section,Regulatory Source,Required for,Keywords
Cover Page with FY,Best Practice,All Companies,"cover page, annual report"
Table of Contents,Best Practice,All Companies,"table of contents, index"
Chairman’s & CEO’s Letter,Best Practice,All Companies,"ceo message, chairman’s message"
Company Overview,Best Practice,All Companies,"company overview, business summary"
Board’s Report,Sec 134,All Companies,"board’s report, directors’ report"
Director’s Responsibility Statement,Sec 134(5),All Companies,"responsibility statement"
Financial Highlights,Sec 134(3),All Companies,"financial summary, highlights"
Dividend Declaration,Sec 123 / SEBI Reg 43A,All Companies,"dividend"
CSR Report,Sec 135,Eligible Companies,"csr, corporate social responsibility"
Secretarial Audit Report,Sec 204,Listed / Large Cos,"secretarial audit"
Statutory Auditor’s Report,Sec 143,All Companies,"statutory audit report"
Financial Statements – Standalone,Schedule III,All Companies,"balance sheet, p&l, income"
Notes to Financial Statements,Ind-AS / AS,All Companies,"notes to accounts"
MD&A,SEBI LODR Reg 34(3),Listed Companies,"management discussion, md&a"
Corporate Governance Report,SEBI LODR Reg 27,Listed Companies,"corporate governance"
Business Responsibility Report,SEBI BRSR Top 1000,Top 1000,"business responsibility, sustainability"
"""
                return pd.read_csv(StringIO(csv_text.strip()))

            def analyze_section(section, keywords, excerpt):
                prompt = (
                    f'Section: "{section}"\n'
                    f'--- Keywords ---\n{keywords}\n'
                    f'--- Content ---\n{excerpt[:3000]}\n\n'
                    f'Respond in JSON format:\n'
                    '{"status": "✅ Present" / "❌ Missing" / "⚠️ Incomplete", "remark": "reason"}'
                )
                try:
                    res = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    return json.loads(re.search(r"\{.*?\}", res.choices[0].message.content, re.DOTALL).group(0))
                except:
                    return {"status": "⚠️ Incomplete", "remark": "GPT error"}

            def run_compliance(text):
                checklist = fallback_checklist()
                results = []
                for _, row in checklist.iterrows():
                    result = analyze_section(row['Section'], row['Keywords'], text)
                    results.append({
                        "✅ Section": row['Section'],
                        "📘 Regulatory Source": row['Regulatory Source'],
                        "📌 Required for": row['Required for'],
                        "📂 Status in Report": result["status"],
                        "📝 Remarks": result["remark"]
                    })
                return pd.DataFrame(results)

            def render_markdown_table(df):
                header = "| ✅ Section | 📘 Regulatory Source | 📌 Required for | 📂 Status | 📝 Remarks |\n"
                divider = "|-----------|----------------------|-----------------|------------|-------------|\n"
                rows = ""
                for _, row in df.iterrows():
                    rows += f"| {row['✅ Section']} | {row['📘 Regulatory Source']} | {row['📌 Required for']} | {row['📂 Status in Report']} | {row['📝 Remarks']} |\n"
                return header + divider + rows

            if st.button("▶️ Run Compliance Check"):
                df = run_compliance(text)

                # Visual Summary
                counts = df["📂 Status in Report"].value_counts().to_dict()
                present = counts.get("✅ Present", 0)
                missing = counts.get("❌ Missing", 0)
                incomplete = counts.get("⚠️ Incomplete", 0)

                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Present", present)
                col2.metric("❌ Missing", missing)
                col3.metric("⚠️ Incomplete", incomplete)

                # Donut Chart
                fig = go.Figure(data=[go.Pie(
                    labels=["✅ Present", "❌ Missing", "⚠️ Incomplete"],
                    values=[present, missing, incomplete],
                    marker=dict(colors=["green", "red", "orange"]),
                    hole=0.4
                )])
                fig.update_layout(title="📊 Compliance Status Distribution")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### ✅ Full Compliance Checklist")
                st.markdown(render_markdown_table(df), unsafe_allow_html=True)

                missing_sections = df[df["📂 Status in Report"] == "❌ Missing"]["✅ Section"].tolist()
                if missing_sections:
                    st.markdown("### 🔴 Summary of Missing Sections:")
                    for item in missing_sections:
                        st.error(f"❌ {item}")

                    # Add reference table for missing sections
                    st.markdown("### 📚 Regulatory References for Missing Sections")

                    # Mapping sections to regulatory links
                    reference_links = {
                        "Board’s Report": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Director’s Responsibility Statement": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Financial Highlights": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Dividend Declaration": "https://caalley.com/gn/68983clcgc55147-gnd3.pdf",
                        "CSR Report": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Secretarial Audit Report": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Statutory Auditor’s Report": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Financial Statements – Standalone": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "Notes to Financial Statements": "https://www.mca.gov.in/Ministry/pdf/CompaniesAct2013.pdf",
                        "MD&A": "https://www.sebi.gov.in/legal/regulations/jun-2021/sebi-listing-obligations-and-disclosure-requirements-regulations-2015-last-amended-on-june-3-2021-_37269.html",
                        "Corporate Governance Report": "https://www.sebi.gov.in/legal/regulations/jul-2024/securities-and-exchange-board-of-india-listing-obligations-and-disclosure-requirements-regulations-2015-last-amended-on-july-10-2024-_84817.html",
                        "Business Responsibility Report": "https://www.sebi.gov.in/legal/regulations/jul-2024/securities-and-exchange-board-of-india-listing-obligations-and-disclosure-requirements-regulations-2015-last-amended-on-july-10-2024-_84817.html"
                    }

                    # Create and display table
                    ref_table = "| Missing Section | Regulatory Reference |\n"
                    ref_table += "|------------------|-----------------------|\n"
                    for section in missing_sections:
                        link = reference_links.get(section, "https://caalley.com/gn/68983clcgc55147-gnd3.pdf")
                        ref_table += f"| {section} | [View Reference]({link}) |\n"

                    st.markdown(ref_table, unsafe_allow_html=True)
                # if missing_sections:
                #     st.markdown("### 🔴 Summary of Missing Sections:")
                #     for item in missing_sections:
                #         st.error(f"❌ {item}")

        # === TAB 3: Redundancy Detection ===
        with tab3:
            st.markdown("<div class='sub-section'>🔁 Redundancy Detector & Section Rewrites</div>", unsafe_allow_html=True)

            def chunk_by_section(text):
                pattern = re.compile(r"(?<=\n)([0-9A-Z]{0,2}[.\)]?\s*)?(?P<title>[A-Z][A-Za-z0-9 /:&,-]{4,60})(?=\n)")
                matches = list(pattern.finditer(text))
                chunks = {}
                for i, m in enumerate(matches):
                    title = m.group("title").strip()
                    start, end = m.end(), matches[i+1].start() if i+1 < len(matches) else len(text)
                    chunk = text[start:end].strip()
                    if len(chunk.split()) >= 10:
                        chunks[title] = chunk
                return chunks

            def get_embedding(text):
                res = openai.embeddings.create(input=[text[:700]], model="text-embedding-3-small")
                return res.data[0].embedding

            def detect_redundancy(chunks, max_sections=6):
                selected = sorted(chunks.items(), key=lambda x: len(x[1]), reverse=True)[:max_sections]
                titles = [k for k, _ in selected]
                texts = [v for _, v in selected]
                embeddings = [get_embedding(t[:750]) for t in texts]
                rows = []

                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                        if sim > 0.85:
                            combined = texts[i][:800] + "\n---\n" + texts[j][:800]
                            prompt = (
                                f"The following two sections may be redundant:\n\n"
                                f"Section A: {titles[i]}\nSection B: {titles[j]}\n\n"
                                f"Content:\n{combined}\n\n"
                                f"Suggest:\n- issue\n- fix\n- restructure\n\n"
                                f"Reply in JSON format."
                            )
                            try:
                                res = openai.chat.completions.create(
                                    model="gpt-4",
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.4,
                                    max_tokens=400
                                )
                                out = json.loads(re.search(r"\{.*?\}", res.choices[0].message.content, re.DOTALL).group(0))
                            except:
                                out = {"issue": "Parse error", "fix": "-", "restructure": "-"}

                            rows.append({
                                "Redundant Areas": f"{titles[i]} & {titles[j]}",
                                "Current Issue": out["issue"],
                                "Recommended Fix": out["fix"],
                                "Revised Structure / Notes": out["restructure"]
                            })
                return pd.DataFrame(rows)

            if st.button("▶️ Run Redundancy Detection"):
                with st.spinner("🔍 Detecting redundancy using GPT + embeddings..."):
                    chunks = chunk_by_section(text)
                    df = detect_redundancy(chunks)

                st.metric("🔁 High-Similarity Redundant Pairs", len(df))

                if df.empty:
                    st.success("✅ No significant redundancy found.")
                else:
                    st.markdown("### 🔁 Redundancy Table")
                    st.dataframe(df, use_container_width=True)

                    st.markdown("### 🪜 Section-Wise Restructuring Plan")
                    for i, row in df.iterrows():
                        with st.expander(f"🔹 {row['Redundant Areas']}"):
                            st.markdown(f"**🧠 Current Issue:** {row['Current Issue']}")
                            st.markdown(f"**🔧 Recommended Fix:** {row['Recommended Fix']}")
                            st.markdown(f"**📐 Revised Structure / Notes:** {row['Revised Structure / Notes']}")

# ✅ Run the app
if __name__ == "__main__":
    main()      
