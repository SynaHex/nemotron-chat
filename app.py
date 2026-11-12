
import os
import streamlit as st

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    RateLimitError,
    APIStatusError,
)

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="Nemotron Chat",
    page_icon="💬",
    layout="centered"
)

st.title("Nemotron AI Chat")
st.caption("Powered by NVIDIA NIM Cloud Endpoints")

MODEL = "nvidia/nemotron-3-super-120b-a12b"

BASE_URL = "https://integrate.api.nvidia.com/v1"

# ============================================================
# 2. NVIDIA API CONFIGURATION
# ============================================================

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error(
        "NVIDIA API Key is missing. "
        "Add NVIDIA_API_KEY to your .env file or environment."
    )
    st.stop()

@st.cache_resource
def get_nvidia_client(key):
    return OpenAI(
        base_url=BASE_URL,
        api_key=key,
        timeout=120.0,
        max_retries=1
    )

client = get_nvidia_client(api_key)

# ============================================================
# 3. SYSTEM PROMPTS
# ============================================================

NORMAL_PROMPT = """
You are a highly capable AI assistant powered by the Nemotron model.

Provide accurate, helpful, logical, and concise answers.

Use structured formatting when appropriate.
Explain technical concepts clearly.
Prioritize correctness and practical solutions.
"""

CAVEMAN_PROMPT = """
Respond in a terse, direct style while preserving all technical substance.

Rules:
1. Answer first. State the solution and reason.
2. Avoid unnecessary greetings and pleasantries.
3. Prefer short, clear words.
4. Never remove important technical details.
5. Keep sentences concise.
6. Preserve code, paths, and error messages exactly.
7. Explain tool execution and results clearly.
8. Respond in the user's language.
9. Do not use phrases such as "me think".
10. Avoid decorative emojis.
"""

# ============================================================
# 4. SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "caveman_mode" not in st.session_state:
    st.session_state.caveman_mode = False

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.title("Settings")

    st.write("### Model Information")

    st.caption(f"Model: {MODEL}")

    st.divider()

    st.write("### Response Style")

    if st.session_state.caveman_mode:
        st.success("Caveman Mode: Active")
    else:
        st.info("Normal Mode: Active")

    st.divider()

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# 6. RENDER CONVERSATION HISTORY
# ============================================================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 7. CHAT INPUT
# ============================================================

prompt = st.chat_input("Message Nemotron...")

if prompt:

    cmd = prompt.strip().lower()

    # ========================================================
    # 8. CUSTOM COMMAND HANDLING
    # ========================================================

    command_response = None

    if cmd == "/caveman status":

        status = (
            "Caveman mode: active"
            if st.session_state.caveman_mode
            else "Caveman mode: inactive"
        )

        command_response = status

    elif cmd in ["stop caveman", "normal mode"]:

        st.session_state.caveman_mode = False

        command_response = (
            "Caveman mode disabled. "
            "Returning to normal behavior."
        )

    elif cmd in [
        "/caveman",
        "caveman mode",
        "talk like caveman"
    ]:

        st.session_state.caveman_mode = True

        command_response = "Caveman mode activated."

    # Display and store custom commands without sending them to AI.

    if command_response is not None:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "command": True
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": command_response,
            "command": True
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.markdown(command_response)

    # ========================================================
    # 9. NORMAL AI CHAT PROCESSING
    # ========================================================

    else:

        # Store user message.

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Select active system prompt.

        active_system_prompt = (
            CAVEMAN_PROMPT
            if st.session_state.caveman_mode
            else NORMAL_PROMPT
        )

        # Build API messages.
        # Exclude custom commands from model conversation.

        api_messages = [
            {
                "role": "system",
                "content": active_system_prompt
            }
        ] + [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in st.session_state.messages
            if not msg.get("command", False)
        ]

        # ====================================================
        # 10. NVIDIA API REQUEST AND STREAMING
        # ====================================================

        with st.chat_message("assistant"):

            message_placeholder = st.empty()

            full_response = ""

            try:

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=api_messages,
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=4096,
                    stream=True
                )

                # Process streamed response.

                for chunk in response:

                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    content = delta.content

                    if content:

                        full_response += content

                        message_placeholder.markdown(
                            full_response + "▌"
                        )

                # Display final response.

                message_placeholder.markdown(full_response)

                # Save successful assistant response.

                if full_response:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

                else:

                    st.warning(
                        "Nemotron returned no visible response."
                    )

            # =================================================
            # 11. ERROR HANDLING
            # =================================================

            except AuthenticationError:

                st.session_state.messages.pop()

                st.error(
                    "401 Authentication Error: "
                    "Check your NVIDIA API key."
                )

            except PermissionDeniedError:

                st.session_state.messages.pop()

                st.error(
                    "403 Permission Error: "
                    "Your NVIDIA account may not have access "
                    "to this API or model."
                )

            except NotFoundError as e:

                st.session_state.messages.pop()

                st.error(
                    f"404 Not Found: {e}\n\n"
                    "Check the model identifier, hosted endpoint "
                    "availability, and NVIDIA account provisioning."
                )

            except RateLimitError:

                st.session_state.messages.pop()

                st.error(
                    "429 Rate Limit Error: "
                    "NVIDIA API usage limit reached."
                )

            except APITimeoutError:

                st.session_state.messages.pop()

                st.error(
                    "Request Timeout: "
                    "NVIDIA did not respond within the configured "
                    "timeout period."
                )

            except APIConnectionError as e:

                st.session_state.messages.pop()

                st.error(
                    f"API Connection Error: {e}"
                )

            except APIStatusError as e:

                st.session_state.messages.pop()

                st.error(
                    f"NVIDIA API Error: HTTP {e.status_code}\n\n"
                    f"Details: {e}"
                )

            except Exception as e:

                st.session_state.messages.pop()

                st.error(
                    f"Unexpected Application Error: {e}"
                )
