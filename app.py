import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Nemotron Chat", page_icon="💬", layout="centered")
st.title("Nemotron AI Chat")
st.caption("Powered by NVIDIA NIM Cloud Endpoints")

# Fetch the API key silently from Streamlit Secrets / Environment
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("API Key is missing from the server secrets.")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Base system prompts
NORMAL_PROMPT = "You are a highly capable AI assistant powered by the Nemotron model. Provide accurate, helpful, and concise answers."
CAVEMAN_PROMPT = """Respond terse like smart caveman. All technical substance stay. Only fluff die.
Rules:
1. Answer first. [thing] [action] [reason]. [next step].
2. Kill ceremony. No greetings, hedging, or pleasantries.
3. Short words. Standard acronyms fine.
4. Articles optional. Never drop not/never/no/only/except.
5. One idea per sentence. 20 words max.
6. Payload verbatim. Code, paths, and errors exact.
7. Tool runs: bounded status.
8. User's language.
9. Never perform caveman. No "me think" or decorative emoji."""

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "caveman_mode" not in st.session_state:
    st.session_state.caveman_mode = False

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Message Nemotron..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    cmd = prompt.strip().lower()
    
    # Logic: Status check
    if cmd == "/caveman status":
        status_msg = "Caveman mode: active" if st.session_state.caveman_mode else "Caveman mode: unknown"
        st.session_state.messages.append({"role": "assistant", "content": status_msg})
        st.chat_message("assistant").write(status_msg)
        st.stop()
        
    # Logic: Switch off
    elif cmd in ["stop caveman", "normal mode"]:
        st.session_state.caveman_mode = False
        off_msg = "Caveman mode disabled. Returning to normal behavior."
        st.session_state.messages.append({"role": "assistant", "content": off_msg})
        st.chat_message("assistant").write(off_msg)
        st.stop()
        
    # Logic: Switch on
    elif cmd in ["/caveman", "caveman mode", "talk like caveman"]:
        st.session_state.caveman_mode = True
        on_msg = "Active."
        st.session_state.messages.append({"role": "assistant", "content": on_msg})
        st.chat_message("assistant").write(on_msg)
        st.stop()

    # Determine which system prompt to inject for this API call
    active_system_prompt = CAVEMAN_PROMPT if st.session_state.caveman_mode else NORMAL_PROMPT
    
    # Build payload with the correct system prompt at the top
    api_messages = [{"role": "system", "content": active_system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b",
                messages=api_messages,
                temperature=0.2 if st.session_state.caveman_mode else 0.5,
                max_tokens=4096,
                stream=True,
                timeout=15.0 # Prevents the app from hanging silently
            )
            
            # Robust fallback: Handle both streamed and non-streamed responses safely
            if hasattr(response, 'choices'):
                full_response = response.choices[0].message.content
                message_placeholder.markdown(full_response)
            else:
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices and chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                # Render the final text without the cursor
                message_placeholder.markdown(full_response)
            
            # Save the successful response to the session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Connection Error: {e}. If this is a timeout, the model is temporarily offline on NVIDIA's cloud.")