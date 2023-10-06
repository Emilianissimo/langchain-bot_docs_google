# LangChain chat bot

Langbot tested with azure, but should work as well for OpenAI itself, you could just config it

### Structure
- Data folder -> database for vectorized information, like chunks and etc. Creates dynamically. Better to delete each run
- Single doc -> just to read one doc and get prompt answer
- Single long doc -> to read document, split it to the chunks and send to prompt to get an answer
- Docs are just have some old docs from my laptop (checked for any sensitive info by the bot)
- Factories folder has LLMFactory, that init all components
- Services has GoogleSearchiService that uses Google Custom Search Engine to get query results
- Singletones has DocumentLoaderSingletone and SettignsSingletone for documents load and .env variables load accordingly
- Multi doc -> to read document and get the conversation with history in the terminal
- Main -> the main running file
- Constants for use

All creds in the .env.example should be copied to the new .env and filled!

### GOOGLE API
Create google api key and custom search engine ID by following those pages:
- https://console.cloud.google.com/apis/credentials -> create and take an API key
- https://programmablesearchengine.google.com/controlpanel/create -> create CSE and take an ID (identification string)

After that you will maybe have to enable CSE by following link from the error (to enable this extension for the current project).
After enabling, you will have to create CSE credentials (no need to use them into the project)
