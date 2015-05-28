import os

from tfidf import TF_IDF
from ICTCLAS2015 import nlpir

def maxent_predeal(item_info):
    def load_dictionary():
        file_dir = os.path.dirname(os.path.realpath(__file__))
        lexicon_dir = os.path.join(file_dir, 'lexicon')
        lexicon = {}

        # weibo lexicon
        lexicon['weibo_lexicon'] = {}
        f = open(os.path.join(lexicon_dir, 'InputLexicon.txt'))
        for line in f.readlines():
            line = line.strip().split()
            if not len(line) == 2:
                continue
            lexicon['weibo_lexicon'][line[0]] = {
                    'class':'weibo',
                    'polarity':int(line[1])
                    }
        f.close()

        #emotion lexicon
        lexicon['emotion_lexicon'] = {}
        f = open(os.path.join(lexicon_dir, 'EmotionLexicon.txt'))
        for line in f.readlines():
            line = line.strip().split('\t')
            if not len(line) == 3:
                continue
            lexicon['emotion_lexicon'][line[0]] = {
                    'class':int(line[1]),
                    'polarity':int(line[2])
                    }
        f.close()

        #emotion lexicon
        lexicon['negation_lexicon'] = {}
        f = open(os.path.join(lexicon_dir, 'NegationOperatorLexicon.txt'))
        for line in f.readlines():
            line = line.strip().split()
            if not len(line) == 1:
                continue
            lexicon['negation_lexicon'][line[0]] = {
                    'class':'negation',
                    'polarity':1
                    }
        f.close()

        #adverb_lexicon
        lexicon['adverb_lexicon'] = {}
        f = open(os.path.join(lexicon_dir, 'StrengthenerAdverbLexicon.txt'))
        for line in f.readlines():
            line = line.strip().split()
            if not len(line) == 1:
                continue
            lexicon['adverb_lexicon'][line[0]] = {
                    'class':'adverb',
                    'polarity':1
                    }
        f.close()

        #ask_lexicon
        lexicon['ask_lexicon']= {}
        f = open(os.path.join(lexicon_dir, 'asked_word.txt'))
        for line in f.readlines():
            line = line.strip().split()
            if not len(line) == 1:
                continue
            lexicon['ask_lexicon'][line[0]] = {
                    'class':'ask',
                    'polarity':1
                    }
        f.close()

        lexicon['stop_word']= {}
        f = open(os.path.join(lexicon_dir, 'stop_word.txt'))
        for line in f.readlines():
            line = line.strip().split()
            if not len(line) == 1:
                continue
            lexicon['stop_word'][line[0]] = {
                    'class':'stop',
                    'polarity':1
                    }
        f.close()
        return lexicon

    def add_word_class_polarity(word, lexicon):
        w, pos, c, polarity = word[0], word[1], 'other', 1 
        for dictionary in [
                lexicon['adverb_lexicon'], lexicon['negation_lexicon'],
                lexicon['emotion_lexicon'], lexicon['weibo_lexicon'],
                lexicon['ask_lexicon']]:
            if w in dictionary:
                c = dictionary[w]['class']
                polarity = dictionary[w]['polarity']
                break
        return {'word':w, 'pos':pos, 'class':c, 'polarity':polarity}

    def ictclas2015(item_info, lexicon):
        result = []
        for item_id, value in item_info.iteritems():
            content = []
            try:
                for word in nlpir.Seg(value['content']):
                    content.append(add_word_class_polarity(word, lexicon))
            except Exception as e:
                print '!!!'
                print value
                print value['content']
                print e
                break
            value['content'] = content
            result.append((item_id, value))
        return result

    def get_field_word(segment_content, lexicon):
        texts = []
        stext = []
        term_pos_dict = {}
        for item_id, value in segment_content:
            if value['tag'] == '0':
                continue
            text = []
            for word in value['content']:
                w = word['word']
                text.append(w)
                stext.append(w)
                if w not in term_pos_dict:
                    term_pos_dict[w] = {}
                pos = word['pos']
                if pos not in term_pos_dict[w]:
                    term_pos_dict[w][pos] = 0
                term_pos_dict[w][pos] += 1
            texts.append(text)

        pos_dict = {}
        for w, value in term_pos_dict.iteritems():
            p_list = value.items()
            p_list.sort(key=lambda x:x[1], reverse=True)
            pos_dict[w] = p_list[0][0]
            
        tfidf = TF_IDF(texts)
        term_list = []
        for term in set(stext):
            term_list.append( (term, tfidf.tf_idf(term, stext)) )

        term_list.sort(key=lambda x:x[1], reverse=True)
        term_list = [term for term in term_list[:int(0.005*len(term_list))] 
                if term[0] not in lexicon['stop_word'] 
                and term[0] not in lexicon['emotion_lexicon']
                and term[0] not in lexicon['weibo_lexicon']
                and pos_dict[term[0]].startswith('n')]

        #print '\n'.join(['_'.join(map(str, list(term) + [pos_dict[term[0]]])) for term in term_list 
            #if pos_dict[term[0]].startswith('n')])
        of = open('field', 'w')
        for field in term_list:
            of.write(field[0] + '\n')
        of.close()
        return term_list

    lexicon = load_dictionary()
    segment_content = ictclas2015(item_info, lexicon)
    field_word_list = get_field_word(segment_content, lexicon)
    return (lexicon, segment_content, field_word_list)
 
