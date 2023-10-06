import dotenv
import openai

config = dotenv.dotenv_values('.env')


class SettingsSingleTone:
    OPENAI_API_KEY:      str = config.get('OPENAI_API_KEY')
    OPENAI_BASE:         str = config.get('OPENAI_BASE')
    OPENAI_TYPE:         str = config.get('OPENAI_TYPE')
    OPENAI_VERSION:      str = config.get('OPENAI_VERSION')
    DEPLOYMENT_NAME:     str = config.get('DEPLOYMENT_NAME')
    EMBEDDED_MODEL_NAME: str = config.get('EMBEDDED_MODEL_NAME')
    # Google search API
    GOOGLE_API_KEY:      str = config.get("GOOGLE_API_KEY")
    GOOGLE_CSE_ID:       str = config.get("GOOGLE_CSE_ID")

    def __init__(self, provider='openai'):
        if provider == 'azure':
            openai.api_key = self.OPENAI_API_KEY
            openai.api_base = self.OPENAI_BASE
            openai.api_type = self.OPENAI_TYPE
            openai.api_version = self.OPENAI_VERSION
        else:
            openai.api_key = self.OPENAI_API_KEY
