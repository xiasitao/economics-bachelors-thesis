import pytask
import pickle
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import numpy as np

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

from pathlib import Path
SOURCE_PATH = Path(__file__).parent.resolve()
ASSET_PATH = SOURCE_PATH.joinpath('..', '..', 'assets').resolve()
BUILD_PATH = SOURCE_PATH.joinpath("..", "..", "build").resolve()


def filter_tokens(doc: list) -> list:
    nltk_tokens = word_tokenize(' '.join(doc))
    return [token[0] for token in pos_tag(nltk_tokens)
        if len(token[0]) > 1
        and token[0] != 've'
        and token[1] in ('FW', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ')  # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    ]


def equilibrate_role_models(articles: pd.DataFrame, ses_data: pd.DataFrame) -> pd.DataFrame:
    """Equilibrate role models such that for each prevalent_ses the same amount of role models are present

    Args:
        articles (pd.DataFrame): all articles relevant for these role models
        ses_data (pd.DataFrame): substrate

    Returns:
        pd.DataFrame: substrate with equal amounts of role models for each prevalent ses
    """    
    ses_data = ses_data[ses_data.index.isin(articles['role_model'])]
    minimum_count = ses_data.groupby('prevalent_ses').count()['count'].min()
    ses_data = ses_data.sample(frac=1.0, random_state=42)
    ses_data['_group_index'] = ses_data.groupby('prevalent_ses').cumcount()
    ses_data = ses_data[ses_data['_group_index'] < minimum_count]
    ses_data = ses_data.drop('_group_index', axis=1)
    return ses_data


def train_lda_model(n_topics: int, dictionary: Dictionary, corpus: list) -> tuple:
    """Train an LDA model on the corpus with a predefined number of topics.

    Args:
        n_topics (int): Number of topcis to generate.

    Returns:
        tuple: model, topics (10 most significant words for each topic)
    """    
    iterations = 500
    passes = 4
    model = LdaModel(
        corpus=corpus, id2word=dictionary.id2token, random_state=42,
        iterations=iterations, passes=passes,
        num_topics=n_topics,
    )
    topics = [[topic_entry[1] for topic_entry in topic[0][0:10]] for topic in model.top_topics(corpus)]
    return model, topics


def find_topic_and_entropy(model: LdaModel, dictionary: Dictionary, doc: list) -> tuple:
    """Find the most probable topic and the topic distribution
    entropt for a document.

    Args:
        model (LdaModel): Topic model
        doc (list): Document as list of tokens

    Returns:
        tuple: topic, entropy
    """    
    topic_probabilities = np.array(model[dictionary.doc2bow(filter_tokens(doc))])
    topics = topic_probabilities[:, 0]
    probabilities = topic_probabilities[:, 1]
    topic = topics[probabilities.argmax()]
    entropy = -probabilities.dot(np.log(probabilities))
    return topic, entropy

TOPIC_MODELLING_BUILD_PATH = BUILD_PATH / 'topic_modelling/topic_modelling.pkl'
@pytask.mark.skip()
@pytask.mark.depends_on(BUILD_PATH / 'articles/articles_balanced_50.pkl')
@pytask.mark.produces(TOPIC_MODELLING_BUILD_PATH)
def task_topic_modelling(produces: Path, n_min=2, n_max=10):
    """Perform topic modelling on the 50-articles-per-role-model balanced article data set.

    Args:
        produces (Path): Destination file path
        n_min (int, optional): Smalles number of topics to examine. Defaults to 2.
        n_max (int, optional): Largest number of topics to examine.. Defaults to 10.
    """    
    articles = pd.read_pickle(BUILD_PATH / 'articles/articles_balanced_50.pkl')
    articles_en = articles[articles.language_ml == 'en']
    articles_en = articles_en[['article_id', 'content_slim']]
    articles_en['content_tokenized'] = articles_en['content_slim'].str.split(' ')
    
    # Building corpus with balanced articles
    dictionary = Dictionary(articles_en['content_tokenized'])
    dictionary.filter_extremes(no_below=5, no_above=0.3)
    dictionary[0]
    corpus = [dictionary.doc2bow(filter_tokens(article)) for article in articles_en['content_tokenized']]

    # Predicting with unique articles
    unique_articles = articles_en[['article_id', 'content_slim']].drop_duplicates().set_index('article_id', drop=True)
    unique_articles['content_tokenized'] = unique_articles['content_slim'].str.split(' ')
    all_n_topics = [i for i in range(n_min, n_max+1)]
    article_topics = pd.DataFrame(data=None, columns=[wildcard.format(n) for n in all_n_topics for wildcard in ('topic_{}', 'topic_{}_entropy')], index=unique_articles.index)
    topic_words = {}
    for n_topics in all_n_topics:
        model, these_topic_words = train_lda_model(n_topics, corpus=corpus, dictionary=dictionary)
        topic_words[n_topics] = these_topic_words
        article_topics[[f'topic_{n_topics}', f'topic_{n_topics}_entropy']] = unique_articles['content_tokenized'].parallel_apply(lambda doc: pd.Series(find_topic_and_entropy(model, dictionary, doc)))
    
    # Save topic-classified articles
    with open(produces, 'wb') as file:
        pickle.dump((topic_words, article_topics,), file)


if __name__ == '__main__':
    task_topic_modelling(TOPIC_MODELLING_BUILD_PATH, 2, 15)