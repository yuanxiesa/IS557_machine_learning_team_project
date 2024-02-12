# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn


# tokenization and keep nouns
def keep_nouns(line):
    is_noun = lambda pos: pos[:2] == 'NN'
    tokenized = word_tokenize(line)
    nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return nouns


def remove_hyperlink(nouns):
    new_nouns = []
    for noun in nouns:
        if ('http' not in noun) and ('//' not in noun) and ('@' not in noun):
            new_nouns.append(noun)
    return new_nouns


def single_pair_path_similarity(tweet_1_nouns, tweet_2_nouns):
    """
    param tweet_1_nouns: a list of nouns from one tweet
    :param tweet_2_nouns: a list of nouns from the other tweet
    :return: the averaged path similarity score between all noun pairs (wn.path_similarity)
    """
    score = 0
    num_pair = 0
    for noun_1 in tweet_1_nouns:
        for noun_2 in tweet_2_nouns:
            try:
                synset_1 = wn.synsets(noun_1)[0]
                synset_2 = wn.synsets(noun_2)[0]
            except:
                pass
            else:
                score += synset_1.path_similarity(synset_2)
                num_pair += 1

    try:
        avg_score = score / num_pair
        return avg_score
    except ZeroDivisionError:
        return float('Inf')


def compute_semantic_similarity_score(tweet_a_nouns, tweet_b_nouns):
    """

    param tweet_a_nouns: a list of nouns from one tweet
    :param tweet_b_nouns: a list of nouns from the other tweet
    :return: Sab^2/Saa/Sbb. Sab is the returned value from single_pair_path_similarity(a,b)
    """

    Saa = single_pair_path_similarity(tweet_a_nouns, tweet_a_nouns)
    Sbb = single_pair_path_similarity(tweet_b_nouns, tweet_b_nouns)
    Sab = single_pair_path_similarity(tweet_a_nouns, tweet_b_nouns)

    score = Sab * Sab / Saa / Sbb
    return score


def main():
    # use a scaled similarity measure: S(a, b)^2/(S(a,a)*S(b,b))
    tweet_1_nouns = ['boy', 'food', 'allergies', 'blood', 'transfusion']
    tweet_1 = 'boy gets food allergies from blood transfusion  http://ow.ly/lllbm\n'
    tweet_2_nouns = ['patients', 'kin', 'hospitals', 'confusion']
    tweet_2 = 'listing patients as \'next of kin\' at hospitals may cause legal confusion  http://ow.ly/liumg\n'
    tweet_3_nouns = ['canine', 'flu', 'outbreak', 'chicago', 'kills', 'dogs']
    tweet_3 = 'rare canine flu outbreak in chicago kills 5 dogs http://ow.ly/liddj\n'
    tweet_4_nouns = ['tuberculosis', 'interrupts', 'school', 'spring', 'break']
    tweet_4 = 'tuberculosis testing interrupts ohio school\'s spring break http://ow.ly/ligqs\n'

    score = compute_semantic_similarity_score(tweet_1_nouns, tweet_2_nouns)
    print(f'\nThe similarity Score between \ntweet #1 {tweet_1} and\ntweet #2 {tweet_2} is: {round(score, 4)}')

    score = compute_semantic_similarity_score(tweet_3_nouns, tweet_4_nouns)
    print(f'\nThe similarity Score between \ntweet #3 {tweet_3} and\ntweet #4 {tweet_4} is: {round(score, 4)}')

    score = compute_semantic_similarity_score(tweet_2_nouns, tweet_3_nouns)
    print(f'\nThe similarity Score between \ntweet #2 {tweet_2} and\ntweet #3 {tweet_3} is: {round(score, 4)}')

    score = compute_semantic_similarity_score(tweet_1_nouns, tweet_4_nouns)
    print(f'\nThe similarity Score between \ntweet #1 {tweet_1} and\ntweet #4 {tweet_4} is: {round(score, 4)}')


if __name__ == '__main__':
    main()
