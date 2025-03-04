from LQPR import *

init_pattern_vecs()

# print("初始化完成")

sentence = "The system shall be able to add product to shopping cart in less than 2ms. "

vec = sentence_vector("in under 100 minutes")
vec2 = sentence_vector(" in under 100")
print(cos_sim(vec, vec2))

# word_list = get_word(sentence)
# split_sentence_vecs = get_split_sentence_vec(word_list)

# for i in range(len(pattern_vecs)):
#     print(' '.join(languagePatterns[i]), semantic_smilarity(split_sentence_vecs, pattern_vecs[i]))