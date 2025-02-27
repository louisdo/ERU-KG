import unicodedata, re, string
from cleantext import clean
from functools import lru_cache 


@lru_cache(maxsize=10000, typed=False)
def clean_text(text: str, deep_clean: bool = False) -> str:
    """
    Cleans and preprocesses the input text.

    Parameters:
    - text (str): The input text to be cleaned.
    - deep_clean (bool, optional): If True, performs a more comprehensive cleaning using the 'clean' function from the 'clean-text' library.
      Default is False.

    Returns:
    - str: The cleaned text.
    """

    if deep_clean:
        _text = clean(
            text,
            fix_unicode=True,  # fix various unicode errors
            to_ascii=True,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=True,  # replace all email addresses with a special token
            no_phone_numbers=True,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
            no_punct=False,  # remove punctuations
            replace_with_url="<url>",
            replace_with_email="<email>",
            replace_with_phone_number="<phone>",
            lang="en",  # set to 'de' for German special handling
        )
    else:
        _text = unicodedata.normalize("NFKD", text)
        _text = re.sub(r"\s\s+", " ", _text).strip()

    return _text



@lru_cache(maxsize=16384)
def simple_tokenize(text: str, lower = False):
    """
    Tokenizes the given text into a list of tokens.
    
    Args:
        text (str): The input text to be tokenized.
        lower (bool, optional): Whether to convert the tokens to lowercase. Defaults to False.
    
    Returns:
        list: A list of tokenized tokens extracted from the input text.
    """
    # Function implementation goes here
    if not text: return []
    if not isinstance(text, str): text = str(text)
    if lower:
        res = [
            tok.strip(string.punctuation).strip("\n").lower() for tok in re.split(r"[-\,\(\)\s]+", text)
        ]
    else:
        res = [tok.strip(string.punctuation).strip("\n") for tok in re.split(r"[-\,\(\)\s]+", text)]

    return [tok for tok in res if tok]