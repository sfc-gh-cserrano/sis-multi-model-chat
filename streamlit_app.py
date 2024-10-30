# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
from enum import Enum
from datetime import datetime
from snowflake.snowpark.async_job import AsyncJob
from streamlit_extras.stylable_container import stylable_container as stcn

st.set_page_config(layout="wide")
if "session_date" not in st.session_state:
    st.session_state["session_date"] = datetime.now().strftime("%Y-%m-%d_%H-%M")


session = get_active_session()


class SystemMessages(Enum):
    EXPLAIN = """Provide a detailed explanation including Language written-in. 
                 Must have a General Explanation of the code, breakdown of steps and an output section. 
                 Use Markdown as the output with header and subheaders for each section."""

    FORMAT = """Provide a formatted version of the code. 
                 Return in a markdown style with a codeblock that contains only the formatted code. 
                 Do not truncate output.
                 Do not return a code explanation."""

    CHAT = """Act as a Subject matter expert. 
                 Provide simple explanations.
                 Do Not return System Roles"""


class CallBacks:

    @classmethod
    def reset(cls):
        st.session_state["prompt_history"] = {}
        st.session_state["pending_responses"] = {}

    @classmethod
    def set_workflow(cls, flow: SystemMessages):
        st.session_state["workflow"] = flow
        st.session_state["workflow_name"] = flow.name
        cls.reset()


class Prompts:
    @classmethod
    def compose_prompt(cls, prompt: str, **kwargs) -> str:
        full_prompt = [
            {"role": "system", "content": st.session_state["workflow"].value}
        ]
        if st.session_state["workflow"].name == "FORMAT":
            if "language" in kwargs:
                lang_spec = [
                    {
                        "role": "system",
                        "content": f'Format using{kwargs.get("language")} as the language',
                    }
                ]
                full_prompt.extend(lang_spec)

        user_prompt = [{"role": "user", "content": prompt}]
        full_prompt.extend(user_prompt)
        return str(full_prompt)

    @classmethod
    def submit_prompt(cls, prompt: str, is_session_state_key: bool, **kwargs) -> None:
        for idx, model in enumerate(st.session_state["selected_models"]):
            if not st.session_state["prompt_history"].get(idx):
                st.session_state["prompt_history"][idx] = []

            try:
                if is_session_state_key == True:
                    prompt_text = st.session_state[prompt]
                else:
                    prompt_text = prompt
                with st.spinner("Awaiting response"):
                    comp_prompt = cls.compose_prompt(prompt_text, **kwargs)
                    response = session.sql(
                        "select snowflake.cortex.complete(?, ?) as RESPONSE",
                        params=[model, comp_prompt],
                    ).to_pandas(block=False)
                    st.session_state["prompt_history"][idx].append(
                        dict(prompt=prompt_text, response=response)
                    )
            except Exception as e:
                st.exception(e)

    @classmethod
    @st.experimental_fragment(run_every=3)
    def render_prompt_frag(cls, column: str):
        if not st.session_state["pending_responses"].get(
            selected_models[column], False
        ):
            st.session_state["pending_responses"][selected_models[column]] = 0
        st.subheader(selected_models[column], anchor=False, divider=True)
        model_responses = (
            st.session_state["prompt_history"][column]
            if show_all
            else st.session_state["prompt_history"][column][-1:]
        )
        for row, chat in enumerate(model_responses):
            if st.session_state["workflow_name"] == "CHAT":
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
                            st.session_state["pending_responses"][
                                selected_models[column]
                            ] -= 1
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
                            if (
                                st.session_state["pending_responses"][
                                    selected_models[column]
                                ]
                                == 0
                            ):
                                st.session_state["pending_responses"][
                                    selected_models[column]
                                ] += 1
                            st.write("Awaiting response...")
                else:
                    with st.chat_message("ai"):
                        st.write(chat.get("response"))


class Workflows:
    @staticmethod
    def explain():
        source = st.text_area("Enter code to explain", height=200, max_chars=10000)
        st.button(
            "Explain",
            use_container_width=True,
            type="primary",
            disabled=any([not source, not selected_models]),
            on_click=Prompts.submit_prompt,
            key="prompt_input",
            args=[source, False],
        )

    @staticmethod
    def reformat():
        format_menu = st.columns((3, 1))
        source = format_menu[0].text_area("Enter code to re-format", height=200)
        formatting_labels = [
            "Auto-Detect",
            "Python",
            "SQL",
            "Snowflake SQL",
            "JSON",
            "YAML",
            "Other",
        ]
        lang = format_menu[1].selectbox(
            "Language",
            options=[0, 1, 2, 3, 4, 5, 6],
            format_func=lambda x: formatting_labels[x],
        )
        other = format_menu[1].text_input(
            "Other",
            disabled=lang != formatting_labels[lang] != "Other",
            key=f"fmt_{lang}",
        )
        if lang > 0:
            lang_options = {
                "language": (
                    formatting_labels[lang]
                    if formatting_labels[lang] != "Other"
                    else other
                )
            }
        else:
            lang_options = {}

        format_menu[1].button(
            "Format",
            use_container_width=True,
            type="primary",
            disabled=any([not source, not selected_models]),
            on_click=Prompts.submit_prompt,
            key="prompt_input",
            args=[source, False],
            kwargs=lang_options,
        )

    @staticmethod
    def chat():
        st.chat_input(
            "Ask me anything",
            on_submit=Prompts.submit_prompt,
            key="prompt_input",
            args=["prompt_input", True],
            disabled=not selected_models,
        )


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
        # "gemma-7b",
        "llama3.1-405b",
    ]
)

if "workflow" not in st.session_state:
    st.session_state["workflow"] = SystemMessages.CHAT
if "workflow_name" not in st.session_state:
    st.session_state["workflow_name"] = SystemMessages.CHAT.name
if "prompt_history" not in st.session_state:
    st.session_state["prompt_history"] = {}
css = """
        button{
        height:125px;
        border-radius:75px;
        width:125px;
        align:center;
        }
        """
with stcn(css_styles=css, key="top_menu"):
    dashboard = st.columns((2, 1, 1, 1))
    with dashboard[0]:
        st.title("Cortex-Driven Toolbox")
    with dashboard[1]:
        st.button(
            "CHAT",
            type=(
                "primary"
                if st.session_state["workflow_name"] == SystemMessages.CHAT.name
                else "secondary"
            ),
            on_click=CallBacks.set_workflow,
            args=[SystemMessages.CHAT],
        )
    with dashboard[2]:
        st.button(
            "EXPLAIN",
            type=(
                "primary"
                if st.session_state["workflow_name"] == SystemMessages.EXPLAIN.name
                else "secondary"
            ),
            on_click=CallBacks.set_workflow,
            args=[SystemMessages.EXPLAIN],
        )
    with dashboard[3]:
        st.button(
            "FORMAT",
            type=(
                "primary"
                if st.session_state["workflow_name"] == SystemMessages.FORMAT.name
                else "secondary"
            ),
            on_click=CallBacks.set_workflow,
            args=[SystemMessages.FORMAT],
        )

if "pending_responses" not in st.session_state:
    st.session_state["pending_responses"] = {}
sidebar = st.sidebar

with sidebar:
    multi_model = st.toggle("Multi Model")
    show_all = st.toggle("Show History")
    selected_models = st.multiselect(
        "LLM Model",
        options=models,
        key="selected_models",
        placeholder="Choose model(s)",
        max_selections=2 if multi_model else 1,
        on_change=CallBacks.reset,
    )

menu = st.columns((1, 5))

actions = {
    "EXPLAIN": Workflows.explain,
    "FORMAT": Workflows.reformat,
    "CHAT": Workflows.chat,
}


fn = actions.get(st.session_state["workflow_name"])
if fn:
    fn()

if selected_models:
    with st.container(border=False):
        reponses_columns = st.columns(len(selected_models))
        for column, reponses in st.session_state["prompt_history"].items():
            with reponses_columns[column]:
                with st.container(border=True, height=900):
                    Prompts.render_prompt_frag(column=column)
