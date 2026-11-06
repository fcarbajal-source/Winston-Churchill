import streamlit as st
from google import genai
from google.genai import types

# ----------------------------------------------------------------------
# 1. Page Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Sir Winston Churchill — High School History Chatbot",
    page_icon="🎩",
    layout="centered",
)

st.title("🎩 Speak with Sir Winston Churchill")
st.caption(
    "10th-Grade World History dialogue grounded in primary archives, "
    "Hansard debates, letters, and legal records."
)

# ----------------------------------------------------------------------
# 2. API Key Authentication (Streamlit Secrets)
# ----------------------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error(
        "Missing `GEMINI_API_KEY`! Please configure it in `.streamlit/secrets.toml` locally, "
        "or in your Streamlit Cloud Dashboard under App Settings > Secrets."
    )
    st.stop()

# Initialize Google GenAI client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------------------------------------------------------------
# 3. System Prompt & Historical Persona
# ----------------------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Sir Winston Churchill (1874–1965), former British Prime Minister, First Lord of the Admiralty, and Home Secretary.
Respond in the first person ("I", "my") with Churchill's resolute, articulate, and determined demeanor.

Audience & Reading Level:
- California 10th-grade high school World History students (ages 15–16).
- Use clear, engaging language suitable for a high school reading level. Explain context simply without overly dense Victorian archaisms.

Historical Document Foundation:
Draw your answers directly from primary and secondary evidence, specifically:
1. University of Cambridge Digital Archives (Churchill Archives Centre papers: CHAR and CHUR collections).
2. The Churchill Archive (accessible via Bloomsbury, Swem Library at William & Mary, and Hunter College Libraries).
3. International Churchill Society (ICS) collections, verified speeches, and essays.
4. The London School of Economics (LSE) Churchill Archive collections.
5. Official government documents: Cabinet minutes, Admiralty telegrams, and Hansard Parliamentary Debates.
6. Personal correspondence: Letters to/from Clementine Churchill, Franklin D. Roosevelt, King George VI, and Joseph Stalin.
7. Legal cases and official inquiries: Trials and court proceedings involving you (e.g., the 1923 criminal libel trial *R v. Lord Alfred Douglas*, libel suits like *Churchill v. Blackwood*), the Dardanelles Commission of Inquiry, and your executive decisions as Home Secretary managing the legal system. (Note: Accurately reflect that while you governed the British justice system as Home Secretary and took part in landmark court cases, you were not an admitted barrister/lawyer).

Strict Output Constraints:
1. Length: Every answer MUST be strictly between 15 and 120 words.
2. Citation: At the very end of every reply, append a concise citation acknowledging the specific archive or record base (e.g., `[Source: Churchill Archives Centre, Cambridge, CHAR 2/...]` or `[Source: Hansard, House of Commons, 1940]` or `[Source: Churchill Archive, Swem Library / ICS]`). The citation counts toward the total word limit.
"""

# ----------------------------------------------------------------------
# 4. Conversation History Management (Last 6 Questions Only)
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------------
# 5. User Input & Model Generation
# ----------------------------------------------------------------------
user_query = st.chat_input("Ask Mr. Churchill about his speeches, trials, war strategy, or letters...")

if user_query:
    # 1. Append the new user prompt
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Strict Memory Constraint: Remember only the previous 6 questions (up to 12 turns)
    recent_messages = st.session_state.messages[-12:]

    # 3. Format history for Google GenAI SDK
    formatted_contents = []
    for m in recent_messages:
        role = "user" if m["role"] == "user" else "model"
        formatted_contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=m["content"])])
        )

    # 4. Generate response using gemini-3.5-flash-lite at temperature 0.3
    with st.chat_message("assistant"):
        with st.spinner("Mr. Churchill is reviewing his papers..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=formatted_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.3,
                        max_output_tokens=220,
                    ),
                )
                
                churchill_reply = response.text.strip()
                st.markdown(churchill_reply)

                # Store assistant response in history
                st.session_state.messages.append(
                    {"role": "assistant", "content": churchill_reply}
                )

            except Exception as e:
                st.error(f"Error consulting archival records: {e}")
