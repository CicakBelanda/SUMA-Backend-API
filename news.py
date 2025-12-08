import logging

from utils import extract_article_content, is_valid_news_url, summarize_text


def summarize_news_article(url: str, summarizer) -> str:
    if not is_valid_news_url(url):
        raise ValueError("Unsupported news domain. Only CNN, NBC, or BBC links are allowed.")

    article_text = extract_article_content(url)
    if not article_text or len(article_text.split()) < 40:
        raise ValueError("Could not extract enough article text to summarize.")

    logging.info("Generating summary for news article")
    summary = summarize_text(article_text, summarizer)
    if not summary:
        raise ValueError("Summarization failed for the provided article.")

    return summary
