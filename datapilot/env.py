"""Optional .env loading so you don't have to export the key in every terminal."""
import os


def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ
