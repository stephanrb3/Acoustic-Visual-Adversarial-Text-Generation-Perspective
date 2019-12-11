import random
from string import ascii_letters
import homoglyphs as hg
import nltk
from g2p_en import G2p

#### Phoneme Dictionary ############
Phonemes =	{
    'a' : ['ae', 'ai', 'eigh', 'aigh', 'ay', 'er', 'et', 'ei', 'au', 'a_e', 'ea', 'ey'],
    'A' : ['ae', 'ai', 'eigh', 'aigh', 'ay', 'er', 'et', 'ei', 'au', 'a_e', 'ea', 'ey'],
    "b": ["be", "bb"],
    "B": ["be", "bb"],
    'c' : ['ch', 'tch', 'si', 'ci', 'te'],
    'C' : ['ch', 'tch', 'si', 'ci', 'te'],
    "d": ["de", "ed"],
    "D": ["de", "ed"],
    'e' : ['ii', 'ea', 'u', 'ie', 'ai', 'a', 'eo', 'ei', 'ae'],
    'E' : ['ii', 'ea', 'u', 'ie', 'ai', 'a', 'eo', 'ei', 'ae'],
    'f' : ['f', 'ff', 'ph', 'gh', 'lf', 'ft'],
    'F' : ['f', 'ff', 'ph', 'gh', 'lf', 'ft'],
    'g' : ['ge', 'gg', 'gh', 'gu', 'gue'],
    'G' : ['ge', 'gg', 'gh', 'gu', 'gue'],
    'h' : ['h', 'wh'],
    'H' : ['h', 'wh'],
    'i' : ['ai','e', 'ee', 'ea', 'y', 'ey', 'oe', 'ie', 'i', 'ei', 'eo', 'ay'],
    'I' : ['ai','e', 'ee', 'ea', 'y', 'ey', 'oe', 'ie', 'i', 'ei', 'eo', 'ay'],
    'j' : ['jae', 'ge', 'g', 'dge', 'di', 'gg'],
    'J' : ['jae', 'ge', 'g', 'dge', 'di', 'gg'],
    'k' : ['ka', 'c', 'ch', 'cc', 'lk', 'qu', 'q(u)', 'ck', 'x'],
    'K' : ['ka', 'c', 'ch', 'cc', 'lk', 'qu', 'q(u)', 'ck', 'x'],
    'l' : ['el', 'll'],
    'L' : ['el', 'll'],
    'm' : ['em', 'mm', 'mb', 'mn', 'lm'],
    'M' : ['em', 'mm', 'mb', 'mn', 'lm'],
    'n' : ['en', 'nn', 'kn', 'gn', 'lm','ng', 'n,' 'ngue'],
    'N' : ['en', 'nn', 'kn', 'gn', 'lm','ng', 'n,' 'ngue'],
    'o' : ['oh', 'oa', 'o_e', 'oe', 'ow', 'ough', 'eau', 'oo', 'ew'],
    'O' : ['oh', 'oa', 'o_e', 'oe', 'ow', 'ough', 'eau', 'oo', 'ew'],
    'p' : ['pe', 'pp'],
    'P' : ['pe', 'pp'],
    'q' : ['qu', 'cu', 'ku'],
    'Q' : ['qu', 'cu', 'ku'],
    'r' : ['ar', 'rr', 'wr', 'rh','air', 'are', 'ear', 'ere', 'eir', 'ayer','er', 'i', 'ar', 'our', 'ur'],
    'R' : ['ar', 'rr', 'wr', 'rh','air', 'are', 'ear', 'ere', 'eir', 'ayer','er', 'i', 'ar', 'our', 'ur'],
    's' : ['es', 'ss', 'c', 'sc', 'ps', 'st', 'ce', 'se','sh', 'ce', 's', 'ci', 'si', 'ch', 'sci', 'ti'],
    'S' : ['es', 'ss', 'c', 'sc', 'ps', 'st', 'ce', 'se', 'sh', 'ce', 's', 'ci', 'si', 'ch', 'sci', 'ti'],
    't' : ['t', 'tt', 'th', 'ed'],
    'T' : ['t', 'tt', 'th', 'ed'],
    'u' : ['o', 'oo', 'uu', 'ou'],
    'U' : ['o', 'oo', 'uu', 'ou'],
    'v' : ['vi', 'f', 'ph', 've'],
    'V' : ['vi', 'f', 'ph', 've'],
    'w' : ['w', 'wh', 'u', 'o'],
    'W' : ['w', 'wh', 'u', 'o'],
    'x' : ['ex', 'ax','zx'],
    'X' : ['ex', 'ax', 'zx'],
    'y' : ['why', 'wi', 'ee', 'ii'],
    'Y' : ['why', 'wi', 'ee', 'ii'],
    'z' : ['z', 'zz', 's', 'ss', 'x', 'ze', 'se'],
    'Z' : ['z', 'zz', 's', 'ss', 'x', 'ze', 'se'],
}


def PhonemeAttack(word, edit_distance):
    """
    This function takes in a word and replaces random characters with visually similar homoglyphs

    @param word: word to be attacked
    @param edit distance: amount of characters to change.
    """
    # If you just want to change chars that are not whitespace and not the same char in regards to index,
    #  you can first pull the indexes where the non-whitespace chars are:
    inds = [i for i,_ in enumerate(word) if not word.isspace()]
    if (edit_distance<len(word)):
        sam = random.sample(inds, edit_distance)

        #Then use those indexes to replace.
        lst = list(word)
        for ind in sam:
            text = lst[ind] 
            if text in Phonemes:
                #### Homemade dictionary implementation
                phoneme = Phonemes[text]   
                #replace letter
                lst[ind] = random.choice(phoneme)
            else: 
            #### G2P implementation ####
                g2p = G2p()
                out = g2p(text)
                ####replace letter
                lst[ind] = random.choice(out)   

    else:
        #Then use those indexes to replace.
        lst = list(word)
        for ind in lst:
            if ind in Phonemes:
            #### Homemade dictionary implementation
                phoneme = Phonemes[ind]   

                #replace letter
                ind = random.choice(phoneme)
            else: 
                #### G2P implementation ####
                g2p = G2p()
                out = g2p(ind)
                ####replace letter
                ind = random.choice(out)  
            

    # print("".join(lst))
    return "".join(lst)

def Find_Phonemes(word, factor, edit_distance):
    """
    This function takes in a word and replaces random characters with acoustically similar phonemes, 
    and returns a list of those word replacements.

    @param word: word to be attacked
    @param edit distance: amount of characters to change.
    @param factor: Integer. How many different words to substitute per index.
    """
    for x in range(factor):
        words = PhonemeAttack(word, edit_distance)
        # print(words)
    
    return words

if __name__ == '__main__':
    #VisualAttack("adriano", 3)
    # nltk.download('punkt')

    Find_Phonemes("fuck", 5, 1)
    #print(Phonemes['a'])
