# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
from enum import Enum
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.async_job import AsyncJob
import time

st.set_page_config(layout="wide")
session = get_active_session()

if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = {}


class SystemMessages(Enum):
    DEFAULT = """Act as a Subject matter expert. 
                 Provide simple explanations.
                 Do Not return System Roles"""


def reset():
    st.session_state["prompt_history"] = {}


def submit_prompt(prompt: str) -> None:
    """Submits prompt to LLM using `to_pandas() method with `block=false` argument. Returns a Snowpark Async Job"""
    for idx, model in enumerate(st.session_state["selected_models"]):
        if not st.session_state["prompt_history"].get(idx):
            st.session_state["prompt_history"][idx] = []
        try:
            with st.spinner("Awaiting response"):
                comp_prompt = compose_prompt(st.session_state[prompt])
                response = session.sql(
                    "select snowflake.cortex.complete(?, ?) as RESPONSE",
                    params=[model, comp_prompt],
                ).to_pandas(block=False)
                st.session_state["prompt_history"][idx].append(
                    dict(prompt=st.session_state[prompt], response=response)
                )
        except Exception as e:
            st.exception(e)


def compose_prompt(prompt: str) -> str:
    """Composes a prompt text adding system roles."""
    full_prompt = [{"role": "system", "content": SystemMessages.DEFAULT.value}]
    user_prompt = [{"role": "user", "content": prompt}]
    full_prompt.extend(user_prompt)
    return str(full_prompt)


models = sorted(
    [
        "llama3-8b",
        "llama3-70b",
        "snowflake-arctic",
        "reka-core",
        "reka-flash",
        "mistral-large",
        "mixtral-8x7b",
        "llama2-70b-chat",
        "mistral-7b",
        "gemma-7b",
        "llama3.1-405b",
    ]
)


pending_reponses = 0
selected_models = st.multiselect(
    "LLM Model",
    options=models,
    key="selected_models",
    max_selections=4,
    on_change=reset,
)
menu = st.columns((1, 5))
with menu[0].container(border=True):
    show_all = st.toggle("Show all", value=False)
if selected_models:
    with st.container(border=True):
        reponses_columns = st.columns(len(selected_models))
        for column, reponses in st.session_state["prompt_history"].items():
            with reponses_columns[column]:
                st.subheader(selected_models[column], anchor=False, divider=True)
                model_responses = (
                    st.session_state["prompt_history"][column]
                    if show_all
                    else st.session_state["prompt_history"][column][-1:]
                )
                for row, chat in enumerate(model_responses):
                    if chat.get("prompt"):
                        with st.chat_message("user"):
                            st.write(chat.get("prompt"))
                    if chat.get("response"):
                        if isinstance(chat.get("response"), AsyncJob):
                            if chat.get("response").is_done():
                                try:
                                    model_responses[row]["response"] = chat.get(
                                        "response"
                                    ).result()["RESPONSE"][0]
                                except Exception as e:
                                    if "unknown model" in e.__repr__():
                                        model_responses[row][
                                            "response"
                                        ] = "`Model Not Available`"
                                    else:
                                        "`Unknown Error`"
                                with st.chat_message("ai"):
                                    st.write(chat.get("response"))
                            else:
                                with st.chat_message("ai"):
                                    pending_reponses += 1
                                    st.write("Awaiting response...")
                        else:
                            with st.chat_message("ai"):
                                st.write(chat.get("response"))
if pending_reponses > 0:
    time_left = 5
    prog_cont = menu[1].empty()
    for percent_complete in range(5):

        prog_cont.info(
            f"**Pending prompts, will check for progress in :orange[{time_left-percent_complete}] seconds**"
        )
        time.sleep(1)
        prog_cont.empty()
    st.rerun()

st.chat_input(
    "Ask me anything",
    on_submit=submit_prompt,
    key="prompt_input",
    args=["prompt_input"],
)
