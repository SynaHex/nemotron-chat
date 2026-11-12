
import os
import streamlit as st
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

# NVIDIA API configuration
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA API key is missing.")
    st.stop()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=1
)

MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Generate response
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

        for chunk in response:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Display final answer content, not internal reasoning
            content = delta.content

            if content:
                full_response += content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        if full_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })
        else:
            st.warning("The model returned no visible answer.")

    except AuthenticationError:
        st.error("401: NVIDIA API authentication failed. Check your API key.")

    except PermissionDeniedError:
        st.error("403: NVIDIA API access is restricted.")

    except NotFoundError as e:
        st.error(f"404: NVIDIA model or function not found. Details: {e}")

    except RateLimitError:
        st.error("429: NVIDIA API rate limit reached.")

    except APITimeoutError:
        st.error("Request timed out. Try again later.")

    except APIConnectionError as e:
        st.error(f"API connection failed: {e}")

    except APIStatusError as e:
        st.error(
            f"NVIDIA API error: HTTP {e.status_code}\n\n{e.response.text}"
        )

    except Exception as e:
        st.error(f"Unexpected application error: {e}")
