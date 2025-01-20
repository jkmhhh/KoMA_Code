
import yaml
import os


def load_openai_config():
    OPENAI_CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    if OPENAI_CONFIG['OPENAI_API_TYPE'] == 'azure':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
        os.environ["OPENAI_API_VERSION"] = OPENAI_CONFIG['OPENAI_API_VERSION']
        os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['OPENAI_API_BASE']
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['OPENAI_API_KEY']
        os.environ["EMBEDDING_MODEL"] = OPENAI_CONFIG['EMBEDDING_MODEL']
    elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'openai':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
        os.environ["OPENAI_API_KEY"] = OPENAI_CONFIG['REAL_OPENAI_KEY']
        os.environ["OPENAI_API_BASE"] = OPENAI_CONFIG['OPENAI_API_BASE']
    elif OPENAI_CONFIG['OPENAI_API_TYPE'] == 'ollama':
        os.environ["OPENAI_API_TYPE"] = OPENAI_CONFIG['OPENAI_API_TYPE']
    else:
        raise ValueError("Unknown OPENAI_API_TYPE")


if __name__ == "__main__":
    pass