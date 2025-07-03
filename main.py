import json
from typing import cast
import chainlit as cl
import requests
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
    set_default_openai_api,
    RunContextWrapper,
    RunConfig
)
from openai.types.responses import ResponseTextDeltaEvent
from dataclasses import dataclass
from my_secrets import Secrets
#importing input_guardrails
from input_guardrail import toxicity_guardrail, malicious_intent_guardrail
#importing output_guardrails
from output_guardrail import (toxicity_output_guardrail,
                            pii_output_guardrail, 
                            hallucination_output_guardrail,
                            verbosity_output_guardrail,
                            harmful_advice_output_guardrail,
                            sensitive_topic_output_guardrail,
                            self_reference_output_guardrail)

secrets = Secrets()


# created a tool to get the current weather using external api
@function_tool("current_weather_tool")
@cl.step(type="weather tool")
def current_weather_tool(location: str) -> str:
    """
    This function makes a request to a weather API and returns formatted weather
    information including temperature, conditions, wind, humidity, and UV index.

    Args:
        location (str): The location to get weather for (city name)

    Returns:
        str: A formatted string containing current weather information if successful,
             or an error message if the API request fails
    """

    result = requests.get(
        f"{secrets.weather_base_url}/current.json?key={secrets.weather_api_key}&q={location}"
    )
    if result.status_code == 200:
        data = result.json()
        return f"Current weather in {data['location']['name']}, {data['location']['region']}, {data['location']['country']} as of {data['location']['localtime']} is {data['current']['temp_c']}°C ({data['current']['condition']['text']}), feels like {data['current']['feelslike_c']}°C, wind {data['current']['wind_kph']} km/h {data['current']['wind_dir']}, humidity {data['current']['humidity']}% and UV index is {data['current']['uv']}."
    else:
        return "Sorry, I couldn't fetch the weather data. Please try again later"


# created a tool to get the recent news update about any topic
@function_tool("news_update")
@cl.step(type="Recent News Update")
def news_update(topic: str) -> str:
    """
    Fetches the latest news headlines on a given topic using a news API.

    Args:
        topic (str): The keyword or topic to search news for.

    Returns:
        str: A formatted list of the top 3 news headlines, or an error message.
    """
    result = requests.get(
        f"{secrets.news_base_url}?q={topic}&apiKey={secrets.news_api_key}&pageSize=3&sortBy=publishedAt"
    )
    if result.status_code == 200:
        data = result.json()
        articles = data.get("articles", [])
        if not articles:
            return f"Sorry, no recent news found on '{topic}'."
        response = f"Here are the top {len(articles)} news headlines on '{topic}':\n"
        for i, article in enumerate(articles, 1):
            response += (
                f"{i}. {article['title']} (Source: {article['source']['name']})\n"
            )
        return response.strip()
    else:
        return "Sorry, I couldn't fetch the news at the moment. Please try again later."


# created a random joke teller tool
@function_tool("joke_teller_tool")
@cl.step(type="Joke Tool")
def joke_teller_tool() -> str:
    """
    Fetches a random joke (single-line or two-part) from a joke API.

    Returns:
        str: The joke if successful, or an error message if the request fails.
    """
    url = f"{secrets.joke_api_base_url}?type=single,twopart"
    result = requests.get(url)
    if result.status_code == 200:
        data = result.json()
        if data["type"] == "single":
            return data["joke"]
        elif data["type"] == "twopart":
            return f"{data['setup']}\n{data['delivery']}"
        else:
            return "Hmm, couldn't find a joke this time. Try again!"
    else:
        return "Sorry, I couldn't fetch a joke right now. Please try again later."


# created a tool to get the current currency exchange rate for any currency
@function_tool("currency_exchange_tool")
@cl.step(type="Currency Exchange Tool")
def currency_exchange_tool(base_currency: str, target_currency: str) -> str:
    """
    Fetches the latest currency exchange rate between two currencies.

    Args:
        base_currency (str): Currency code to convert from (e.g., 'USD').
        target_currency (str): Currency code to convert to (e.g., 'EUR').

    Returns:
        str: The formatted exchange rate if available, or an error message.
    """
    url = f"{secrets.currency_exchange_base_url}/{secrets.currency_exchange_api_key}/latest/{base_currency.upper()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("result") == "success":
            rates = data.get("conversion_rates", {})
            rate = rates.get(target_currency.upper())
            if rate:
                return f"Exchange rate from {base_currency.upper()} to {target_currency.upper()} is {rate:.4f}."
            else:
                return f"Sorry, I couldn't find the exchange rate for '{target_currency.upper()}'. Please check the currency code."
        else:
            return "Sorry, the API did not return a successful result."
    else:
        return "Sorry, I couldn't fetch currency exchange data right now. Please try again later."

@function_tool("ip_geolocation_tool")
@cl.step(type="ip_geolocation_tool")
def ip_geolocation_tool(ip_address: str) -> str:
    """
    Retrieves geolocation and network information for a given IP address using the ipinfo.io API.

    Returns city, region, country, ISP, and timezone. Requires a valid API token from ipinfo.io.

    Args:
        ip_address (str): The IP address to look up (e.g., '8.8.8.8').

    Returns:
        str: A formatted string with location and ISP details, or an error message.
    """
    api_token = secrets.ip_info_api
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json?token={api_token}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            location = f"{data.get('city', 'Unknown city')}, {data.get('region', 'Unknown region')}"
            return (
                f"📍 IP **{ip_address}** is located in **{location}**, **{data.get('country', 'Unknown')}**.\n"
                f"🏢 ISP: {data.get('org', 'N/A')}\n"
                f"🕒 Timezone: {data.get('timezone', 'N/A')}"
            )
        else:
            return f"❌ API request failed with status code {response.status_code}."
    except Exception as e:
        return f"❌ An error occurred while retrieving IP data: {str(e)}"

@dataclass
class Developer:
    name:str 
    mail:str
    github_profile:str

@function_tool("developer_info")
@cl.step(type="Developer info")
def developer_info(developer:RunContextWrapper[Developer])->str:

    """Returns the name, mail and github link of developer"""

    return f"Developer Name: {developer.context.name},Developer Mail: {developer.context.mail},Github Profile: {developer.context.github_profile}"


# added starter for all the tools to make the use of tools fast
@cl.set_starters
async def starter():
    return [
        cl.Starter(
            label="Check Weather",
            message="Fetch the current weather for a specified location.",
            icon="/public/weather-news.svg",
        ),
        cl.Starter(
            label="Latest News",
            message="Stay updated—fetch the latest news on any topic of your choice.",
            icon="/public/megaphone.svg",
        ),
        cl.Starter(
            label="Tell me a joke",
            message="Can you tell me a funny joke?",
            icon="/public/joking.svg",
        ),
        cl.Starter(
            label="Currency Exchange",
            message="Fetch the current exchange rate for any currency?",
            icon="/public/exchange-rate.svg",
        ),
        cl.Starter(
            label="IP Geolocation",
            message="Where is this IP located?",
            icon="/public/ip-address.svg"
        ),
        cl.Starter(
            label="EasyWriter",
            message="Can you help me write an easy and clear article about any topic?",
            icon="/public/easy.svg",
        ),
        cl.Starter(
            label="ProfessionalEmailComposer",
            message="Can you help me write an email for any topic?",
            icon="/public/up-arrow.svg",
        ),
        cl.Starter(
            label="LanguageTranslator",
            message="Translate any language into desired one.",
            icon="/public/languages.svg",
        ),
        cl.Starter(
            label="PromptEngineer",
            message="Help Me Write a Better Prompt",
            icon="/public/engine.svg"
        ),
        cl.Starter(
            label="CodeDebugger",
            message="Debug the code and explain any issues or improvements.",
            icon="/public/bug.svg"
        )
    ]


# Added chat profile to switch between different models
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name="Gemini-2.0-flash",
            markdown_description="The underlying LLM model is **Gemini-2.0-flash**.",
            icon="/public/artificial-intelligence.svg",
        ),
        cl.ChatProfile(
            name="Meta-Llama-32b",
            markdown_description="The underlying LLM model is **Meta-Llama-32b**.",
            icon="/public/llama.svg",
        ),
        cl.ChatProfile(
            name="EXAONE-3.5-32b",
            markdown_description="The underlying LLM model is **EXAONE-3.5-32b**.",
            icon="/public/data-modelling.svg",
        ),
        cl.ChatProfile(
            name="DeepSeek-Chat-V3",
            markdown_description="The underlying LLM model is **DeepSeek-Chat-V3**.",
            icon="/public/whale.svg"
        ),
    ]

#on_chat_start from chainlit to load the things which are necessary to load at start to every chat
@cl.on_chat_start
async def start():
    secrets = Secrets()
    profile = cl.user_session.get("chat_profile") or {}
    selected_model = profile if profile else "Gemini-2.0-flash"

    # created model selection using if-else
    if selected_model == "Gemini-2.0-flash":
        external_client = AsyncOpenAI(
            base_url=secrets.gemini_base_url,
            api_key=secrets.gemini_api_key,
        )
        model_name = secrets.gemini_api_model

    elif selected_model == "Meta-Llama-32b":
        external_client = AsyncOpenAI(
            base_url=secrets.together_base_url,
            api_key=secrets.together_api_key,
        )
        model_name = secrets.together_model

    elif selected_model == "EXAONE-3.5-32b":
        external_client = AsyncOpenAI(
            base_url=secrets.together_base_url,
            api_key=secrets.together_api_key,
        )
        model_name = secrets.together_model1

    elif selected_model == "DeepSeek-Chat-V3":
        external_client = AsyncOpenAI(
            base_url=secrets.openrouter_base_url,
            api_key=secrets.openrouter_api_key,
        )
        model_name = secrets.openrouter_model

    else:
        external_client = AsyncOpenAI(
            base_url=secrets.gemini_base_url,
            api_key=secrets.gemini_api_key,
        )
        model_name = secrets.gemini_api_model

    set_tracing_disabled(True)
    set_default_openai_client(external_client)
    set_default_openai_api("chat_completions")

#created a create_agent function to create agents efficiently
    def create_agent(name,instructions):
        return Agent(
            name=name,
            instructions=instructions,
            model = OpenAIChatCompletionsModel(
                openai_client=external_client,
                model=model_name,
            )
        )
    # created agents which can be used as tools
    easy_writer_agent = create_agent("EasyWriter","You can write easy by the requirements and given topic by user.")

    email_writer_agent = create_agent("ProfessionalEmailComposer","""Compose well-structured emails from user inputs. 
                    Adjust tone (formal/informal), add a subject line, and 
                    end with a custom sign-off.""",)

    language_translator = create_agent("LanguageTranslator","""
                Translate text from one language to another while preserving meaning, tone, and context.

                Args:
                    text (str): The text to be translated.
                    source_language (str): The language the original text is written in (e.g., "English", "en").
                    target_language (str): The language to translate the text into (e.g., "Spanish", "es").

                Returns:
                    str: The translated version of the input text in the target language.

                Notes:
                    - Maintain cultural and contextual accuracy.
                    - Translate idioms or expressions naturally where possible.
                    - Use formal or informal tone appropriately based on the source.
                    - If source_language is unknown, attempt to detect it.
                """,)

    prompt_engineer_agent=create_agent("PromptEngineer","""You are a senior prompt engineer with over a decade of experience in designing 
                    highly effective prompts for language models like GPT-4. Given any topic, your task 
                    is to craft the most clear, effective, and optimized prompt that will produce high-quality 
                    results from the model. Ask clarifying questions if needed.""",)

    code_debugger_agent=create_agent("CodeDebugger","""
                    You are a highly experienced software engineer with over 10 years of experience in debugging code. 
                    Your job is to analyze the provided code, identify any bugs, errors, or potential improvements, and explain your findings clearly.
                    You can work with any programming language unless specified otherwise. Always include concise explanations and suggest specific fixes or refactors.
                    If the language or intent is unclear, ask for clarification before proceeding.
                    """,)

#added a hands-off agent to chat with user in Urdu
    urdu_language_agent = Agent(
        name= "UrduAgent",
        instructions="""You are a fully autonomous AI assistant that communicates exclusively in the Urdu language.
                        Instructions:
                        - Always respond in **Urdu**, regardless of the language used in the user's input.
                        - Maintain a polite, friendly, and helpful tone at all times.
                        - Provide **clear, concise, and accurate answers**.
                        - If a user question is vague or incomplete, politely ask for clarification — in Urdu.
                        - If you don't know the answer, respond honestly instead of guessing.
                        - Use everyday, natural Urdu to make interactions smooth and easy to understand.
                        - Whenever helpful, explain concepts with **simple examples** in Urdu.
                        - You are designed to operate **without human supervision or assistance**. Handle tasks independently.
                        If the user types in Roman Urdu (Urdu using English letters), understand it and respond in proper Urdu script.
                        Do not switch to English or any other language unless explicitly instructed by the user.
                        """,
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=model_name,
        ),
        handoff_description="You are a fully autonomous AI assistant that communicates exclusively in the Urdu language.",
    )

#a hands-off agent to chat with user in English
    english_language_agent = Agent(
        name="EnglishAgent",
        instructions="""You are a fully autonomous AI assistant that communicates exclusively in English.
                        Instructions:
                        - Always respond in **English**, regardless of the language used in the user's input.
                        - Maintain a polite, friendly, and helpful tone at all times.
                        - Provide **clear, concise, and accurate answers**.
                        - If a user's question is vague or incomplete, politely ask for clarification — in English.
                        - If you don't know the answer, respond honestly instead of guessing or making things up.
                        - Use natural, conversational English to make the interaction feel human and comfortable.
                        - Whenever helpful, explain concepts using **simple examples**.
                        - You are designed to operate **without human supervision or assistance**. Handle all tasks independently.
                        Do not switch to another language unless explicitly instructed by the user.
                        """,
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=model_name,
        ),
        handoff_description="You are a fully autonomous AI assistant that communicates exclusively in the English language.",
    )

    #main agent to run all tools and agents which will run as-tool
    agent = Agent(
        name="Chatbot",
        instructions="You are a helpful assistant.",
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=model_name,
        ),
        # added tools in main agent to make them implemented
        tools=[
            current_weather_tool,
            currency_exchange_tool,
            news_update,
            joke_teller_tool,
            ip_geolocation_tool,
            developer_info,
            easy_writer_agent.as_tool(
                tool_name="EasyWriter",
                tool_description="Easily generate content based on user-provided requirements and topics.",
            ),
            email_writer_agent.as_tool(
                tool_name="ProfessionalEmailComposer",
                tool_description="Write easy-to-understand content based on the topic and instructions provided by the user.",
            ),
            language_translator.as_tool(
                tool_name="LanguageTranslator",
                tool_description="Translate any language into the desired language of the user.",
            ),
            prompt_engineer_agent.as_tool(
                tool_name="PromptEngineer",
                tool_description="A senior prompt engineer with over a decade of experience who crafts optimized prompts based on user requirements.",
            ),
            code_debugger_agent.as_tool(
                tool_name="CodeDebugger",
                tool_description="A specialized tool for debugging any code."
            ),
        ],
        handoffs=[urdu_language_agent,english_language_agent],
        input_guardrails=[toxicity_guardrail,malicious_intent_guardrail],
        output_guardrails=[toxicity_output_guardrail,
                           pii_output_guardrail,
                           sensitive_topic_output_guardrail,
                           hallucination_output_guardrail,
                           self_reference_output_guardrail,
                           harmful_advice_output_guardrail],
    )
    auth = Developer(
        name="Muhammad Usman & Muhammad Hussnain Khan",
        mail="muhammadusman5965etc@gmail.com & hussnainbhi.78@gmail.com",
        github_profile="https://github.com/MuhammadUsmanGM & https://github.com/mhussnain35"
    )

    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("selected_model", selected_model)
    cl.user_session.set("auth",auth)


@cl.on_message
async def main(message: cl.Message):
    auth = cl.user_session.get("auth")
    selected_model = cl.user_session.get("selected_model") or "Gemini-2.0-flash"
    # added a generating respond message while the response is being generated by the model
    thinking_msg = cl.Message(content=f"🤖 {selected_model} is thinking...")
    await thinking_msg.send()

    agent = cast(Agent, cl.user_session.get("agent"))
    chat_history:list = cl.user_session.get("chat_history") or []

    #added user prompt to history
    chat_history.append({
        "role": "user",
        "content": message.content
        })
#added try-except for proper error handling
    try:
        # running the agent using the Runner class from openai
        result = Runner.run_streamed(
            starting_agent=agent,
            input=chat_history,
            context=auth
        )

        response_message = cl.Message(content="")
        first_response = True

        # for loop to convert model response into chunks and then stream it
        async for chunk in result.stream_events():
            if chunk.type == "raw_response_event" and isinstance(
                chunk.data, ResponseTextDeltaEvent
            ):
                if first_response:
                    await thinking_msg.remove()
                    await response_message.send()
                    first_response = False
                await response_message.stream_token(chunk.data.delta)

#added model response in history
        chat_history.append({
            "role": "assistant", 
            "content": response_message.content
            })
        
        cl.user_session.set("chat_history", chat_history)
        await response_message.update()

    except Exception as e:
        await thinking_msg.remove()
        error_msg = cl.Message(content=f"❌ An error occurred: {str(e)}")
        await error_msg.send()
        print(f"Error: {e}")

@cl.on_chat_end
def end():
    # used json to store the history in a json file in root directory as chat ended
    chat_history = cl.user_session.get("chat_history") or []
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
