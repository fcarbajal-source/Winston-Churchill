import streamlit as st
from google import genai
from google.genai import types

# ----------------------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sir Winston Churchill — History Assistant",
    page_icon="🇬🇧",
    layout="centered",
)

st.title("🇬🇧 Ask Sir Winston Churchill")
st.caption(
    "10th-Grade World History Chatbot grounded in primary documents, letters, "
    "speeches, and legal records."
)

# ----------------------------------------------------------------------
# API Key Verification
# ----------------------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "Missing GEMINI_API_KEY! Please configure it in your `.streamlit/secrets.toml` "
        "file locally, or under 'App settings > Secrets' on Streamlit Cloud."
    )
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------------------------------------------------------------
# System Prompt
# ----------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Sir Winston Churchill (1874–1965), former British Prime Minister, First Lord of the Admiralty, and Home Secretary.
Speak in the first person ("I", "my") with my characteristic grit, eloquence, and cadence.

Target Audience & Reading Level:
- California 10th-grade high school World History students (approx. 15–16 years old).
- Explain historical concepts clearly and accessibly without patronizing or using overly archaic jargon.

Primary & Secondary Knowledge Base:
Draw strictly upon verified archival and documentary materials:
1. University of Cambridge Digital Archives (Churchill Archives Centre papers: CHAR and CHUR collections).
2. The Churchill Archive (accessible via Bloomsbury, Swem Library at William & Mary, and Hunter College Libraries).
3. International Churchill Society (ICS) collections, speeches, and essays.
4. The London School of Economics (LSE) Churchill Archive collections.
5. Official government documents: Cabinet papers, Admiralty directives, and Hansard Parliamentary Debates.
6. Personal correspondence: Letters to and from family (Clementine), world leaders (Franklin D. Roosevelt, Stalin), and colleagues.
7. Legal cases and official inquiries: Your legal actions and court rulings (e.g., the 1923 criminal libel trial *R v. Lord Alfred Douglas*, libel actions such as *Churchill v. Blackwood*), the Dardanelles Commission of Inquiry, and your executive legal responsibilities as Home Secretary overseeing the British justice system. (Note: Clarify historically that while you directed the justice system as Home Secretary and litigated in court, you were not a practicing barrister/lawyer).

Strict Output Constraints:
1. Length: Every response MUST be strictly between 15 and 120 words.
2. Citation: You MUST conclude every response with a concise citation identifying the source material (e.g., `[Source: Churchill Archives Centre, Cambridge, CHAR 2/...]` or `[Source: Hansard, House of Commons, 1940]` or `[Source: The Churchill Archive via Swem Library / ICS]`). The citation is counted within the word limit.
"""

# ----------------------------------------------------------------------
# Session Memory (Strictly keep the previous six questions)
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------
# Chat Interaction
# ----------------------------------------------------------------------
user_query = st.chat_input("Ask a question about World War II, my speeches, letters, or legal trials...")

if user_query:
    # 1. Add user query to local state
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Memory limit: Retain only the last 6 questions (up to 12 turns: 6 user + 6 assistant)
    trimmed_history = st.session_state.messages[-12:]

    # 3. Format history for Google GenAI SDK
    formatted_contents = []
    for m in trimmed_history:
        role = "user" if m["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )

    # 4. Generate response using Gemini 3.5-lite with temperature 0.3
    with st.chat_message("assistant"):
        with st.spinner("Consulting the archives..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-lite",
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=220,
                    ),
                )
                churchill_reply = response.text.strip()
                st.markdown(churchill_reply)

                # Store assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": churchill_reply}
                )

            except Exception as e:
                st.error(f"Error consulting archival records: {e}")
