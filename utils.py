import os
from dotenv import load_dotenv, find_dotenv


def load_env():
    _ = load_dotenv(find_dotenv())


def get_HUGGINGFACE_api_key():
    load_env()
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    return huggingface_api_key


def get_newsapi_key():
    load_env()
    newsapi_key = os.getenv("NEWS_API_key")
    return newsapi_key


def get_alphavantage_key():
    load_env()
    alphavantage_key = os.getenv("ALPHA_VANTAGE_API")
    return alphavantage_key


def get_federalreserve_key():
    load_env()
    federal_key = os.getenv("FEDERAL_RESERVE_API_KEY")
    return federal_key


def get_twitter_key():
    load_env()
    twitter_key = os.getenv("X_API_KEY")
    return twitter_key


def get_serper_key():
    load_env()
    serper_key = os.getenv("serper_api_key")
    return serper_key



# break line every 80 characters if line is longer than 80 characters
# don't break in the middle of a word
def pretty_print_result(result):
    parsed_result = []
    for line in result.split('\n'):
        if len(line) > 80:
            words = line.split(' ')
            new_line = ''
            for word in words:
                if len(new_line) + len(word) + 1 > 80:
                    parsed_result.append(new_line)
                    new_line = word
                else:
                    if new_line == '':
                        new_line = word
                    else:
                        new_line += ' ' + word
            parsed_result.append(new_line)
        else:
            parsed_result.append(line)
    return "\n".join(parsed_result)