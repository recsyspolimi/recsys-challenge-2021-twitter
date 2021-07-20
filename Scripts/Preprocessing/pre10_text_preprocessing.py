import threading

from Scripts.utilities import start_correct_cluster, read_dataset, save_dataset, parse_args
from preprocessing_utilities import dict_path, temp_output_path, dataset_path

import numpy as np
import re
from transformers import BertTokenizerFast
import text_preprocessing_utilities as text_vars
import dask.dataframe as dd
import string
import os

out_cols = ['text_is_reply', 'text_tokens_count', 'text_unknown_count', 'text_special_tokens_count',
            # 'text_exclamation_count',
            'text_questions_count', 'text_semantic_separation'
            # 'text_period_count', 'text_mention_count',
            # 'text_sentence_count',
                                    'text_newline_count',
            # 'text_tripledots_count', 'text_multidot_count',
            # 'text_multiquestion_count', 'text_multiexclamation_count', 'text_surprise_count',
            # 'text_multipunctuation_count',
            'text_separated_count','text_char_count', 'text_asking_like', 'text_asking_reply', 'text_comment_related_count',
            'text_no_comment_related_count', 'text_asking_retweet',
            # 'text_is_giveaway',
            'text_nsfw_count', 'text_kpop_count', 'text_covid_count', 'text_sports_count',
            'text_japanesetrending_count', 'text_anime_count', 'text_vtuber_count', 'text_news_count',
            'text_myanmar_count', 'text_genshin_count', 'text_nintendo_count', 'text_crypto_count',
            'text_trending_count', 'text_love_count','text_slang_count','text_games_count',

            'text_nsfw_bool', 'text_kpop_bool', 'text_covid_bool', 'text_sports_bool',
            'text_japanesetrending_bool', 'text_anime_bool', 'text_vtuber_bool', 'text_news_bool',
            'text_myanmar_bool', 'text_genshin_bool', 'text_nintendo_bool', 'text_crypto_bool',
            'text_trending_bool', 'text_love_bool', 'text_slang_bool', 'text_games_bool'
            ]
out_frame_name = 'text_features'
SPLIT_TEXT_PREPROCESS = True


def split_text(s: str):
    return np.fromstring(s, dtype=np.int64, sep="\t")


def multiple_replace(dictionary, tex):
    # https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dictionary.keys())))

    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda match_object: dictionary[match_object.string[match_object.start():match_object.end()]], tex)


# Utilities to be employed
def do_count(token):
    def f(arr):
        return np.count_nonzero(arr == token)

    return f


def do_count_all(token_list):
    if token_list:
        assert isinstance(token_list, set)

        def f(arr):
            count = 0
            for t in arr:
                if t in token_list:
                    count += 1
            return count
    else:
        def f(arr):
            return arr.size
    return f


if SPLIT_TEXT_PREPROCESS:
    def preprocess_text(arr):
        t = threading.current_thread()
        try:
            s = t.tokenizer.decode(arr)
        except:
            t.tokenizer = BertTokenizerFast.from_pretrained('./saved_tokenizer')
            s = t.tokenizer.decode(arr)

        to_replace_before_1 = {
            "[cls] rt ": "",
            
            # remove unknown tokens to do at the end => may risk to fix something that doesn't need to be fixed
        }
        to_replace_before_2 = {
            "[sep]": "",
            "[cls]": "",  # remove initial token
            "& amp ; ": "&",  # replace escaped characters
            "& lt ; ": "<",
            "& gt ; ": ">",
            " _ ": "_",  # unite words underscore separate
            " - ": "-",  # unite words separated with -
            "[unk] ": "",
            
            # remove unknown tokens to do at the end => may risk to fix something that doesn't need to be fixed
        }

        to_replace1={
            "リツイート": " リツイート ",
            "イイネ": " イイネ ",
            "いいね": " いいね ",
            "返信": " 返信 ",
            "'s": "",
            "rt": " rt ",
            "retweet": " retweet ",
            "like&follow": " like ",
            "좋아요를": " 좋아요를 ",
            "리트윗": " 리트윗 ",
            "quoteretweet": "qrt",
            "quote retweet": "qrt",
            "quote or retweet": "qrt",
            "quote tweet": "qrt",
            "retweet and quote": "qrt",
            "retweet quote": "qrt",

            "拡散希望": " 拡散希望 ",
            "回复": " 回复 ",
            "引用": " qrt ",
            "larry king" : " larryking " ,
            "dustin diamond": " dustindiamond ",  
            "jj watt": "jjwatt" ,
        }
        to_replace2 = {

            "no qrt": "negqrt",
            "dont qrt": "negqrt",
            "donot qrt": "negqrt",
            "do not qrt": "negqrt",
            "noqrt": "negqrt",
            "dontqrt": "negqrt",
            "donotqrt": "negqrt",
            "donotqrt": "negqrt",

        }
        s = s.lower()

        for match, sub in to_replace_before_1.items():
            s = s.replace(match, sub)
        for match, sub in to_replace_before_2.items():
            s = s.replace(match, sub)

        to_translate = string.punctuation + "¿" + "¡" + "؟" + "。" + "｡" + "・" + "☆" + '②' + '①' + '③' + '→' + '▼' + '／' + '＼' + '⇒' + '•'
        to_translate= to_translate+ "＆"+"（"+"）" +"‰" +"‖"+"‗"+"‣"+"‴"+"※"+"‽"+"⁈"+"⁇"+"⁉"+"⁊" 
        #some seem the same  but are UTF-8
        s = s.translate(str.maketrans('', '', to_translate))
        # s = multiple_replace(dictionary=to_replace, tex=s)
        
        for match, sub in to_replace1.items():
            s = s.replace(match, sub)
        for match, sub in to_replace2.items():
            s = s.replace(match, sub)

        s = [word for word in s.split(' ') if word != ""]

        return s


    def do_count_str(token):
        def f(s_list):
            return s_list.count(token)

        return f


    def do_count_all_str(token_list):
        if token_list:
            assert isinstance(token_list, set)

            def f(s_list):
                count = 0
                for w in s_list:
                    if w in token_list:
                        count += 1
                return count
        else:
            def f(s_list):
                return len(s_list)
        return f
else:
    def preprocess_text(arr):
        t = threading.current_thread()
        try:
            s = t.tokenizer.decode(arr)
        except:
            t.tokenizer = BertTokenizerFast.from_pretrained('./saved_tokenizer')
            s = t.tokenizer.decode(arr)

        to_replace = {
            "[SEP]": "",
            # remove SEP ==> keep a space in front to allow using a space before a character to be used to recognize a word
            "[CLS]": "",  # remove initial token
            "& amp ; ": "&",  # replace escaped characters
            "& lt ; ": "<",
            "& gt ; ": ">",
            "# ": "#",  # concatenate symbol with hashtag body
            "。": ".",  # point
            "｡": ".",  # point
            " _ ": "_",  # unite words underscore separate
            " - ": "-",  # unite words separated with -
            "[UNK] ": "",
            # remove unknown tokens to do at the end => may risk to fix something that doesn't need to be fixed
        }

        # s = multiple_replace(dictionary=to_replace, tex=s)
        for match, sub in to_replace.items():
            s = s.replace(match, sub)

        to_translate = {  # more efficient
            "¿": "",  # in spanish must match ending ? => no added info
            "¡": "",  # in spanish must match ending ! => no additional info
            "؟": "?"
        }
        s = s.translate(to_translate)

        patterns = [
            "https : / / t. co / [^\s]+ ",  # remove link
            "^ RT @[^:]*: ",  # remove initial retweet
            "@ [^\s] ",  # remove mentions
            "\.\.\.[\.]+ ",  # multiple questionMarks used as a new token
            "\?(\?| \?)+",  # multiple exclamations used as a new token
            "\!(\!| \!)+",
            "¶( ¶)+"
        ]
        start_sub_dict = {
            '.': '.multi',
            '?': '?multi',
            '!': '!multi',
            '¶': '¶'
        }
        pat = '|'.join(patterns)
        r = re.compile(pat, flags=re.UNICODE)
        s = r.sub(lambda match_object: start_sub_dict.get(match_object.string[match_object.start()], ""), s)

        # s = re.sub("(\!|\?)(multi)?((\!|\?| \!| \?)(multi)?)+", "`surpriseMarks", s)
        s = s.lower()

        return s


    def do_count_str(token):
        def f(s):
            return s.count(token)

        return f


    def do_count_all_str(token_list):
        assert isinstance(token_list, set)

        def f(s):
            count = 0
            for w in token_list:
                count += s.count(w)
            return count

        return f

if __name__ == '__main__':

    generate_dict, is_test = parse_args()
    c = start_correct_cluster(is_test, use_processes=True)

    columns = [
        "raw_feature_tweet_text_token"
    ]
    df = read_dataset(dataset_path, columns)
    df_links = read_dataset(os.path.join(temp_output_path, 'mapped_links'), ['tweet_links_count'])
    df['tweet_links_count'] = df_links['tweet_links_count']

    # Base Series to be used around
    raw_text_series = df["raw_feature_tweet_text_token"]
    splitted_text_series = raw_text_series.map(split_text, meta=("", "O"))
    preprocessed_text_series = splitted_text_series.apply(preprocess_text,
                                                          # tokenizer=tokenizer,
                                                          meta=("", "O"))
    # TODO: check why dask does not accept args=(tokenizer,) but instead tokenizer=tokenizer

    # counted_splitted_text_series = splitted_text_series.map(Counter, meta=("", "O"))  # problems of garbage collection

    # Output creation
    if not os.path.exists('./saved_tokenizer'):
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
        tokenizer.save_pretrained('./saved_tokenizer')

    out = dd.concat(
        [
            splitted_text_series.map(lambda arr: arr[0] == 101 and arr[1] == 137, meta=("", np.bool_))
                .to_frame(name='text_is_reply'),

            splitted_text_series.map(do_count_all(None), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_tokens_count'),
            splitted_text_series.map(do_count(text_vars.unk_token), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_unknown_count'),
            splitted_text_series.map(do_count_all(text_vars.special_tokens_list), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_special_tokens_count'),
            # splitted_text_series.map(do_count(text_vars.esclamation_token), meta=("", np.uint8))
            #                     .astype(np.uint8).rename('text_exclamation_count').to_frame(),
            splitted_text_series.map(do_count_all({text_vars.question_token, text_vars.rev_question}),
                                     meta=("", np.uint8))
                .astype(np.uint8).rename('text_questions_count').to_frame(),
            splitted_text_series.map(do_count_all({text_vars.period_token, text_vars.coma_token,
                                                   text_vars.period_coma_token, text_vars.period_2,
                                                   text_vars.period_3}),
                                     meta=("", np.uint8))  # " rt ", "the finger that retweets this", " rt.", " retweet"
                .astype(np.uint8).to_frame(name='text_semantic_separation'),
            # splitted_text_series.map(do_count(text_vars.period_token), meta=("", np.uint8))
            #                     .astype(np.uint8).rename('text_period_count').to_frame(),

            # preprocessed_text_series.map(do_count_str('.'), meta=("", np.uint8))
            #                         .astype(np.uint8).rename('text_sentence_count').to_frame(),
            preprocessed_text_series.map(do_count_str('¶'), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_newline_count'),
            # preprocessed_text_series.map(do_count_str('.triple'), meta=("", np.uint8))
            #                         .astype(np.uint8).rename('text_tripledots_count').to_frame(),
            # preprocessed_text_series.map(do_count_str('.multi'), meta=("", np.uint8))
            #                         .astype(np.uint8).rename('text_multidot_count').to_frame(),
            # preprocessed_text_series.map(do_count_str('?multi'), meta=("", np.uint8))
            #                         .astype(np.uint8).rename('text_multiquestion_count').to_frame(),
            # preprocessed_text_series.map(do_count_str('!multi'), meta=("", np.uint8))
            #                         .astype(np.uint8).rename('text_multiexclamation_count').to_frame(),

            preprocessed_text_series.map(do_count_all_str(None), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_separated_count'),
            preprocessed_text_series.map(lambda x: sum(map(len,x)), meta=("", np.uint16))
                .astype(np.uint16).to_frame(name='text_char_count'),

            preprocessed_text_series.map(
                do_count_all_str({"like", "イイネ", "いいね", "좋아요를", "lajk", "tykkäys", "लाइक", "лајк", "polubienie",
                                  }), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_asking_like'),
            preprocessed_text_series.map(do_count_all_str({"reply", "replies", "kommentti", "कॉमेंट", "返信", "リプした人にやる",
                                                           "antwoord", "መልስ", "الرد", "cavab", "erantzun", "উত্তর",
                                                           "odgovorite", "respon", "odpověď", "回复", "responde",
                                                           "respuestas",
                                                           "reponn", "odgovorite", "svar", "responde", "vastaus",
                                                           "répondre", "repondre", "repondez", "répondez",
                                                           "responder", "ateb", "जवाब", "rispondi", "atbildēt", "svare",
                                                           "ответ", "svara",
                                                           "antworten",
                                                           }), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_asking_reply'),
            preprocessed_text_series.map(do_count_all_str({"comment", "引用", "qrt",
                                                           }), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_comment_related_count'),  # only useful ones
            preprocessed_text_series.map(do_count_all_str({"negqrt", "negqrts"
                                                           }), meta=("", np.uint8))
                .astype(np.uint8).to_frame(name='text_no_comment_related_count'),  # only useful ones
            preprocessed_text_series.map(do_count_all_str(
                {'rt', 'rts', 'retweet', 'retweets', 'retwitta', 'retwit', 'retwittare', 'retwittate', 'retwittare',
                 'ರಿಟ್ವೀಟ್','ritwit','ritwittate', 'ritwitta',
                 'retvītot', 'pетвитни', 'റീട്വീറ്റ്', 'retweeten', 'retuitar', 'ретвитнуть', 'retuitear', 'retweets',
                 'retwits',
                 'retweeta', 'ретвит', 'retweeten', 'รีทวีต', 'retweetle', 'ретвіт', 'ٹویٹ',
                 'ritwitta', 'ռեթվիթ', 'retweeta',
                 'পুনঃটুইট', 'retuit', 'retweeter', '리트윗',
                 '轉推', '拡散希望',
                 'रीट्वीट', 'रिट्वीट',
                 }), meta=("", np.uint8))  # " rt ", "the finger that retweets this", " rt.", " retweet"
                .astype(np.uint8).to_frame(name='text_asking_retweet'),
            preprocessed_text_series.map(do_count_all_str(text_vars.NSFW))
                .astype(np.uint8).to_frame(name='text_nsfw_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.Kpop))
                .astype(np.uint8).to_frame(name='text_kpop_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.covid))
                .astype(np.uint8).to_frame(name='text_covid_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.sports))
                .astype(np.uint8).to_frame(name='text_sports_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.japaneseTrending))
                .astype(np.uint8).to_frame(name='text_japanesetrending_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.anime))
                .astype(np.uint8).to_frame(name='text_anime_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.virtualYoutube))
                .astype(np.uint8).to_frame(name='text_vtuber_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.news))
                .astype(np.uint8).to_frame(name='text_news_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.myanmar))
                .astype(np.uint8).to_frame(name='text_myanmar_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.genshinImpact))
                .astype(np.uint8).to_frame(name='text_genshin_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.nintendogames))
                .astype(np.uint8).to_frame(name='text_nintendo_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.crypto))
                .astype(np.uint8).to_frame(name='text_crypto_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.most_searched))
                .astype(np.uint8).to_frame(name='text_trending_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.love))
                .astype(np.uint8).to_frame(name='text_love_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.slang))
                .astype(np.uint8).to_frame(name='text_slang_count'),
            preprocessed_text_series.map(do_count_all_str(text_vars.games))
                .astype(np.uint8).to_frame(name='text_games_count'),

        ],
        axis=1,
        ignore_unknown_divisions=True
    )

    out['text_mention_count'] = splitted_text_series.map(do_count(text_vars.at_token), meta=("", np.uint8)) - \
                                out['text_is_reply']

    columns = [
        'text_nsfw_count', 'text_kpop_count', 'text_covid_count', 'text_sports_count',
        'text_japanesetrending_count', 'text_anime_count', 'text_vtuber_count', 'text_news_count',
        'text_myanmar_count', 'text_genshin_count', 'text_nintendo_count', 'text_crypto_count',
        'text_trending_count', 'text_love_count', 'text_slang_count', 'text_games_count'
    ]

    out_final = dd.concat(
        [out] + [
            out[c].astype(np.bool_).to_frame(c.replace("count", "bool"))
            for c in columns
        ],
        axis=1, ignore_unknown_divisions=True
    )

    # out['text_separated_count'] = out['text_separated_count'] - 4 * df['tweet_links_count']
    # out['text_char_count'] = out['text_char_count'] - 18 * df['tweet_links_count']

    # out['text_multipunctuation_count'] = out['text_multidot_count'] + out['text_multiquestion_count'] + \
    #     out['text_multiexclamation_count']

    save_dataset(temp_output_path, out_final, out_frame_name)

# LEGACY
# out = splitted_text_series.map(lambda arr: arr[0] == 101 and arr[1] == 137, meta=("", np.bool_)).rename(
#     'text_is_reply').to_frame()

# out['text_distinct_tokens_count'] = counted_splitted_text_series.map(do_count_all(None), meta=("", int))
# out['text_unknown_count'] = counted_splitted_text_series.map(do_count(text_vars.unk_token), meta=("", np.uint8))
# out['text_special_tokens_count'] = counted_splitted_text_series.map(do_count_all(text_vars.special_tokens_list),
#                                                                     meta=("", np.uint8))

# out['text_exclamation_count'] = counted_splitted_text_series.map(do_count(text_vars.esclamation_token),
#                                                                  meta=("", np.uint8))
# out['text_questions_count'] = counted_splitted_text_series.map(do_count(text_vars.question_token),
#                                                                meta=("", np.uint8))
# out['text_period_count'] = counted_splitted_text_series.map(do_count(text_vars.period_token), meta=("", np.uint8))
# out['text_mention_count'] = counted_splitted_text_series.map(do_count(text_vars.at_token), meta=("", np.uint8)) - \
#                             out['text_is_reply']

# out['text_sentence_count'] = preprocessed_text_series.map(do_count_str('.'), meta=("", np.uint8))
# out['text_newline_count'] = preprocessed_text_series.map(do_count_str('¶'), meta=("", np.uint8))
# out['text_tripledots_count'] = preprocessed_text_series.map(do_count_str('.triple'), meta=("", np.uint8))
# out['text_multidot_count'] = preprocessed_text_series.map(do_count_str('.multi'), meta=("", np.uint8))
# out['text_multiquestion_count'] = preprocessed_text_series.map(do_count_str('?multi'), meta=("", np.uint8))
# out['text_multiexclamation_count'] = preprocessed_text_series.map(do_count_str('!multi'), meta=("", np.uint8))
# out['text_surprise_count'] = preprocessed_text_series.map(do_count_str('`surpriseMarks'), meta=("", np.uint8))
# out['text_multipunctuation_count'] = out['text_multidot_count'] + out['text_multiquestion_count'] + out[
#     'text_multiexclamation_count'] + out['text_surprise_count']

# out['text_separated_count'] = preprocessed_text_series.map(
#     lambda x: len(re.findall(" [^\s]+ ", x, flags=re.UNICODE)), meta=("", np.uint8))
# out['text_separatedwords_count'] = preprocessed_text_series.map(
#     lambda x: len(re.findall(" [-_#\w:][-_#\w:\.]* ", x, flags=re.UNICODE)), meta=("", np.uint8))
#
# out['text_asking_retweet'] = preprocessed_text_series.map(
#     do_count_all_str([" rt ", "the finger that retweets this", " rt.", " retweet"]), meta=("", np.uint8))
#
# c1 = do_count_all_str({'retweet', 'rt'})
# c2 = do_count_all_str({'like', 'reply', 'comment'})
# c3 = do_count_str('follow')
#
# # I'm sorry for the unreadability of this, but at least is evaluates the least number of conditions ...
# def giveaway_test(s):
#     v1 = c1(s)
#     v2 = c2(s)
#     if v1 and v2:
#         return True
#     if not v1 and not v2:
#         return False
#     if c3(s):
#         return True
#     return False

# out['text_is_giveaway'] = preprocessed_text_series.map(giveaway_test, meta=("", np.bool_))
#
# out['text_nsfw_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.NSFW))
# out['text_kpop_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.Kpop))
# out['text_covid_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.covid))
# out['text_sports_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.sports))
# out['text_japanesetrending_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.japaneseTrending))
# out['text_anime_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.anime))
# out['text_vtuber_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.virtualYoutube))
# out['text_news_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.news))
# out['text_myanmar_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.myanmar))
# out['text_genshin_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.genshinImpact))
# out['text_nintendo_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.nintendogames))
# out['text_crypto_count'] = preprocessed_text_series.map(do_count_all_str(text_vars.crypto))
