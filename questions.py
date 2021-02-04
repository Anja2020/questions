import nltk
import sys
import string
import os
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            path = os.path.join(directory, file)
            with open(path, encoding='utf-8') as f:
                files[file] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []
    tokens = [word.lower() for word in nltk.word_tokenize(document)]
    stopwords = nltk.corpus.stopwords.words("english")
    for word in tokens:
        if word in string.punctuation or word in stopwords:
            continue
        else:
            words.append(word)
    words.sort()
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    words = dict()
    num_docs = len(documents)

    for doc in documents:
        for word in set(documents[doc]):
            if word in words.keys():
                words[word] += 1
            else:
                words[word] = 1

    for word in words:
        idf = math.log(num_docs / words[word])
        idfs[word] = idf
            
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    filenames = list()
    ranking = dict()

    for file in files:
        ranking[file] = 0
        for word in query:
            num_appearing = files[file].count(word)
            ranking[file] += num_appearing * idfs[word]
    filenames = sorted(ranking, key=ranking.get, reverse=True)[:n]
    return filenames


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    top_sentences = list()
    ranking = dict()

    for sentence in sentences:
        mwm = 0
        for word in query:
            if word in sentences[sentence]:
                mwm += idfs[word]
        ranking[sentence] = mwm
    top_sentences = sorted(ranking.items(), key=lambda x:x[1], reverse=True)

    for sentence1 in top_sentences:
        for sentence2 in top_sentences:
            if sentence1[0] != sentence2[0] and sentence1[1] == sentence2[1]:
               
                query_words1 = 0
                query_words2 = 0
                for word in query:
                    if word in sentence1[0]:
                        query_words1 += 1
                    if word in sentence2[0]:
                        query_words2 += 1
        
                qtd1 = query_words1 / len(sentences[sentence1[0]])
                qtd2 = query_words2 / len(sentences[sentence2[0]]) 
                if qtd2 > qtd1 and top_sentences.index(sentence1) < top_sentences.index(sentence2):
                    a, b = top_sentences.index(sentence2), top_sentences.index(sentence1)
                    top_sentences[b], top_sentences[a] = top_sentences[a], top_sentences[b]
                elif qtd1 > qtd2 and top_sentences.index(sentence1) > top_sentences.index(sentence2):
                    a, b = top_sentences.index(sentence1), top_sentences.index(sentence2)
                    top_sentences[b], top_sentences[a] = top_sentences[a], top_sentences[b]
    top_sentences = [s[0] for s in top_sentences][:n]
    return top_sentences

if __name__ == "__main__":
    main()
