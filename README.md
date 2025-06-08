# 🧠 Multi-Tool Chatbot (Chainlit + OpenRouter + Gemini)

A powerful and extensible AI assistant built with [Chainlit](https://www.chainlit.io/), integrated with multiple LLM providers (Gemini, OpenRouter), and enhanced with custom tools like:

- 🌦️ Weather Checker  
- 🗞️ News Fetcher  
- 😂 Programming Joke Teller  
- 💱 Currency Exchange Rate Lookup  
- ✍️ EasyWriter (Custom writing assistant)

---

## 🚀 Features

- ✅ Supports multiple LLMs (Gemini, OpenRouter DeepSeek, OpenRouter Mistral)
- 🔧 Modular tool-based architecture using `function_tool`
- 🧠 Dynamic profile selection and model configuration
- 💬 Streamed responses with live typing
- 🧾 Chat history saved on session end
- 🎯 Useful starter prompts for user engagement

---

## 📁 Project Structure

├── main.py # Entry point with chat logic, streaming, and tools
├── my_secrets.py # Handles environment variables securely
├── .env # API keys and config (not committed)
└── chat_history.json # Chat history output file (on session end)

## 🚆Interface Preview

![Interface](/images/Interface.png)

## 📤Sample Output

![Output](/images/Output.png)
