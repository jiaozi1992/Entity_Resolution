import difflib
import zlib
import bz2
import logging

disagree_weight = 0
threshold = 0


def exact(val1, val2):
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    elif (val1 == val2):
        return 1.0

    else:
        return 0.0

def tokenset(val1,val2):
    # val1 = self.val1
    # val2 = self.val2

    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Remove all stop words - - - - - - - - - - - - - - - - - - - - - - - - - -
    #
    clean_val1 = val1
    clean_val2 = val2

    set1 = set(clean_val1.split())  # Make the cleaned values a set of tokens
    set2 = set(clean_val2.split())

    num_token1 = len(set1)
    num_token2 = len(set2)

    if ((num_token1 == 0) or (num_token2 == 0)):  # No tokens in one of the sets
        w = disagree_weight

    else:
        num_common_token = len(set1.intersection(set2))

        if (num_common_token == 0):
            w = disagree_weight  # No common tokens

        else:
            # Calculate the divisor
            divisor = 0.5 * (num_token1 + num_token2)
            w = float(num_common_token) / float(divisor)

            assert (w >= 0.0), 'Token comparison: Similarity weight < 0.0'
            assert (w <= 1.0), 'Token comparison: Similarity weight > 1.0'

    return w


def winkler(val1, val2):
    JARO_MARKER_CHAR = chr(1)  # Special character used to mark assigned
    # characters

    # Taken from US Census Bureau BigMatch C code 'stringcmp'
    #
    sim_char_pairs = frozenset([('a', 'e'), ('e', 'a'), ('a', 'i'), ('i', 'a'),
                                     ('a', 'o'), ('o', 'a'), ('a', 'u'), ('u', 'a'),
                                     ('b', 'v'), ('v', 'b'), ('e', 'i'), ('i', 'e'),
                                     ('e', 'o'), ('o', 'e'), ('e', 'u'), ('u', 'e'),
                                     ('i', 'o'), ('o', 'i'), ('i', 'u'), ('u', 'i'),
                                     ('o', 'u'), ('u', 'o'), ('i', 'y'), ('y', 'i'),
                                     ('e', 'y'), ('y', 'e'), ('c', 'g'), ('g', 'c'),
                                     ('e', 'f'), ('f', 'e'), ('w', 'u'), ('u', 'w'),
                                     ('w', 'v'), ('v', 'w'), ('x', 'k'), ('k', 'x'),
                                     ('s', 'z'), ('z', 's'), ('x', 's'), ('s', 'x'),
                                     ('q', 'c'), ('c', 'q'), ('u', 'v'), ('v', 'u'),
                                     ('m', 'n'), ('n', 'm'), ('l', 'i'), ('i', 'l'),
                                     ('q', 'o'), ('o', 'q'), ('p', 'r'), ('r', 'p'),
                                     ('i', 'j'), ('j', 'i'), ('2', 'z'), ('z', '2'),
                                     ('5', 's'), ('s', '5'), ('8', 'b'), ('b', '8'),
                                     ('1', 'i'), ('i', '1'), ('1', 'l'), ('l', '1'),
                                     ('0', 'o'), ('o', '0'), ('0', 'q'), ('q', 'o'),
                                     ('c', 'k'), ('k', 'c'), ('g', 'j'), ('j', 'g'),
                                     ('e', ' '), (' ', 'e'), ('y', ' '), (' ', 'y'),
                                     ('s', ' '), (' ', 's')])




    len1, len2 = len(val1), len(val2)

    if (len1 < 4) or (len2 < 4):  # Both strings must be at least 4 chars long
        return disagree_weight

    halflen = max(len1, len2) / 2 - 1  # Or + 1?? PC 12/03/2009

    ass1, ass2 = '', ''  # Characters assigned in string 1 and string 2
    workstr1, workstr2 = val1, val2  # Copies of the original strings

    common1, common2 = 0.0, 0.0  # Number of common characters

    # Analyse the first string  - - - - - - - - - - - - - - - - - - - - - - - -
    #
    for i in range(len1):
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len2)
        index = workstr2.find(val1[i], start, end)
        if (index > -1):  # Found common character
            common1 += 1
            ass1 = ass1 + val1[i]
            workstr2 = workstr2[:index] + JARO_MARKER_CHAR + workstr2[index + 1:]

    # Analyse the second string - - - - - - - - - - - - - - - - - - - - - - - -
    #
    for i in range(len2):
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len1)
        index = workstr1.find(val2[i], start, end)
        if (index > -1):  # Found common character
            common2 += 1
            ass2 = ass2 + val2[i]
            workstr1 = workstr1[:index] + JARO_MARKER_CHAR + workstr1[index + 1:]

    assert (common1 == common2), 'Winkler: Different "common" values'

    if (common1 == 0.0):  # No characters in common
        return disagree_weight

    # Compute number of transpositions  - - - - - - - - - - - - - - - - - - - -
    #
    transp = 0.0
    for i in range(len(ass1)):
        if (ass1[i] != ass2[i]):
            transp += 0.5

    # Check for similarities in non-matched characters - - - - - - - - - - - -
    #

    check_sim = True
    check_init = True
    check_long = True

    if (check_sim == True):

        sim_weight = 0.0

        workstr1 = workstr1.replace(JARO_MARKER_CHAR, '')  # Remove assigned
        workstr2 = workstr2.replace(JARO_MARKER_CHAR, '')  # characters

        for c1 in workstr1:
            for j in range(len(workstr2)):
                if (c1, workstr2[j]) in sim_char_pairs:
                    sim_weight += 3
                    workstr2 = workstr2[:j] + JARO_MARKER_CHAR + workstr2[j + 1:]
                    break  # Mark character as used

        common1 += sim_weight / 10.0

    # Calculate basic (Jaro) weight
    #
    w = 1. / 3. * (common1 / float(len1) + common1 / float(len2) + \
                   (common1 - transp) / common1)

    assert (w > 0.0), 'Winkler: Basic weight is smaller than 0.0: %f' % (w)
    assert (w <= 1.0), 'Winkler: Basic weight is larger than 1.0: %f' % (w)

    # Check for same characters at the beginning - - - - - - - - - - - - - - -
    #
    same_init = 0  # Variable needed later on

    if (check_init == True):

        minlen = min(len1, len2, 4)

        for same_init in range(1, minlen + 1):
            if (val1[:same_init] != val2[:same_init]):
                break
        same_init -= 1

        assert (same_init >= 0), 'Winkler: "same_init" value smaller than 0'
        assert (same_init <= 4), 'Winkler: "same_init" value larger than 4'

        w += same_init * 0.1 * (1.0 - w)

    # Check for long strings and possibly adjust weight - - - - - - - - - - - -
    #
    if (check_long == True):

        if ((min(len1, len2) > 4) and (common1 > same_init + 1) and \
                (common1 >= min(len1, len2) + same_init)):
            w_mod = w + (1.0 - w) * (common1 - same_init - 1) / \
                        (float(len1) + float(len2) - same_init * 2 + 2)
            # Fixed -2 => +2 PC 12/03/2009

            assert (w_mod >= w), 'Winkler: Long string adjustment decreases weight'
            assert (w_mod < 1.0), 'Winkler: Long strings adjustment weight > 1.0'

            w = w_mod

    return w


def QGram(val1, val2):
    # Check if one of the values is a missing value
    #

    padded = True
    common_divisor = 'average'
    QGRAM_START_CHAR = chr(1)
    QGRAM_END_CHAR = chr(2)

    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate q-gram similarity value - - - - - - - - - - - - - - - - - - - -
    #
    q = 3  # Faster access

    # Calculate number of q-grams in strings (plus start and end characters)
    #
    if (padded == True):
        num_qgram1 = len(val1) + q - 1
        num_qgram2 = len(val2) + q - 1
    else:
        num_qgram1 = max(len(val1) - (q - 1), 0)  # Make sure its not negative
        num_qgram2 = max(len(val2) - (q - 1), 0)

    # Check if there are q-grams at all from both strings - - - - - - - - - - -
    # (no q-grams if length of a string is less than q)
    #
    if ((padded == False) and (min(num_qgram1, num_qgram2) == 0)):
        w = disagree_weight

    else:
        if (common_divisor == 'average'):  # Calculate the divisor
            divisor = 0.5 * (num_qgram1 + num_qgram2)
        elif (common_divisor == 'shortest'):
            divisor = min(num_qgram1, num_qgram2)
        else:  # Longest
            divisor = max(num_qgram1, num_qgram2)

        # Use number of q-grams to quickly check if below threshold - - - - - - -
        #
        max_common_qgram = min(num_qgram1, num_qgram2)  # Max possible q-grams
        w = float(max_common_qgram) / float(divisor)  # in common

        if (w < threshold):  # Similariy is smaller than threshold
            w = disagree_weight

        else:

            # Add start and end characters (padding) - - - - - - - - - - - - - - -
            #
            if (padded == True):
                qgram_str1 = (q - 1) * QGRAM_START_CHAR + val1 + (q - 1) * \
                                                                      QGRAM_END_CHAR
                qgram_str2 = (q - 1) * QGRAM_START_CHAR + val2 + (q - 1) * \
                                                                      QGRAM_END_CHAR
            else:
                qgram_str1 = val1
                qgram_str2 = val2

            # Make a list of q-grams for both strings - - - - - - - - - - - - - - -
            #
            qgram_list1 = [qgram_str1[i:i + q] for i in range(len(qgram_str1) - (q - 1))]
            qgram_list2 = [qgram_str2[i:i + q] for i in range(len(qgram_str2) - (q - 1))]

            # Get common q-grams  - - - - - - - - - - - - - - - - - - - - - - - - -
            #
            common = 0

            if (num_qgram1 < num_qgram2):  # Count using the shorter q-gram list
                short_qgram_list = qgram_list1
                long_qgram_list = qgram_list2
            else:
                short_qgram_list = qgram_list2
                long_qgram_list = qgram_list1

            for q_gram in short_qgram_list:
                if (q_gram in long_qgram_list):
                    common += 1
                    long_qgram_list.remove(q_gram)  # Remove the counted q-gram

            w = float(common) / float(divisor)

            assert (w >= 0.0), 'Q-gram: Similarity weight < 0.0'
            assert (w <= 1.0), 'Q-gram: Similarity weight > 1.0'



    return w


def Jaro(val1, val2):
    # Check if one of the values is a missing value
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate Jaro similarity value - - - - - - - - - - - - - - - - - - - - -
    #
    len1, len2 = len(val1), len(val2)
    halflen = max(len1, len2) / 2 - 1  # Or + 1?? PC 12/03/2009

    ass1, ass2 = '', ''  # Characters assigned in string 1 and string 2
    workstr1, workstr2 = val1, val2  # Copies of the original strings

    common1, common2 = 0.0, 0.0  # Number of common characters

    JARO_MARKER_CHAR = chr(1)

    for i in range(len1):  # Analyse the first string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len2)
        index = workstr2.find(val1[i], start, end)
        if (index > -1):  # Found common character
            common1 += 1
            ass1 = ass1 + val1[i]
            workstr2 = workstr2[:index] + JARO_MARKER_CHAR + workstr2[index + 1:]

    for i in range(len2):  # Analyse the second string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len1)
        index = workstr1.find(val2[i], start, end)
        if (index > -1):  # Found common character
            common2 += 1
            ass2 = ass2 + val2[i]
            workstr1 = workstr1[:index] + JARO_MARKER_CHAR + workstr1[index + 1:]

    assert (common1 == common2), 'Jaro: Different "common" values'

    if (common1 == 0.0):  # No characters in common
        w = disagree_weight

    else:  # Compute number of transpositions  - - - - - - - - - - - - - - - -

        transp = 0.0
        for i in range(len(ass1)):
            if (ass1[i] != ass2[i]):
                transp += 0.5

        w = 1. / 3. * (common1 / float(len1) + common1 / float(len2) + \
                       (common1 - transp) / common1)

        assert (w > 0.0), 'Jaro: Weight is smaller than 0.0: %f' % (w)
        assert (w <= 1.0), 'Jaro: Weight is larger than 1.0: %f' % (w)

    return w


def SWDist(val1, val2):
    """Compare two field values using the Smith-Waterman distance approximate
       string comparator.
    """

    # Scores used for Smith-Waterman algorithm - - - - - - - - - - - - - - - -
    #
    match_score = 5
    approx_score = 2
    mismatch_score = -5
    gap_penalty = 5
    extension_penalty = 1

    approx_matches = {'a': 0, 'b': 5, 'd': 1, 'e': 0, 'g': 2, 'i': 0, 'j': 2, 'l': 3,
                           'm': 4, 'n': 4, 'o': 0, 'p': 5, 'r': 3, 't': 1, 'u': 0, 'v': 5}

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    common_divisor = 'average'

    # Calculate Smith-Waterman distance similarity value - - - - - - - - - - -
    #
    n = len(val1)
    m = len(val2)

    if (common_divisor == 'average'):
        divisor = 0.5 * (n + m) * match_score  # Average maximum score
    elif (common_divisor == 'shortest'):
        divisor = min(n, m) * match_score
    else:  # Longest
        divisor = max(n, m) * match_score

    best_score = 0  # Keep the best score while calculating table

    d = []  # Table with the full distance matrix

    for i in range(n + 1):  # Initalise table
        d.append([0.0] * (m + 1))

    for i in range(1, n + 1):
        vali1 = val1[i - 1]
        approx_match1 = approx_matches.get(vali1, -1)

        for j in range(1, m + 1):
            valj2 = val2[j - 1]

            match = d[i - 1][j - 1]

            if (vali1 == valj2):
                match += match_score
            else:
                approx_match2 = approx_matches.get(valj2, -1)

                if (approx_match1 >= 0) and (approx_match2 >= 0) and \
                    (approx_match1 == approx_match2):
                    match += approx_score
                else:
                    match += mismatch_score

            insert = 0
            for k in range(1, i):
                score = d[i - k][j] - gap_penalty - k * extension_penalty
                insert = max(insert, score)

            delete = 0
            for l in range(1, j):
                score = d[i][j - l] - gap_penalty - l * extension_penalty
                delete = max(delete, score)

            d[i][j] = max(match, insert, delete, 0)
            best_score = max(d[i][j], best_score)

    # best_score can be min(len(str1),len)str2))*match_score (if one string is
    # a sub-string of the other string)
    #
    # The lower best_score the less similar the sequences are.
    #
    w = float(best_score) / float(divisor)

    assert (w >= 0.0), 'Smith-Waterman distance: Similarity weight < 0.0'
    assert (w <= 1.0), 'Smith-Waterman distance: Similarity weight > 1.0'

    return w


def __do_lcs__(str1, str2):
    """Method to extract longest common substring from the two input strings.
       Returns the common substring, its length, and the two input strings with
       the common substring removed.

       Should not be used from outside the module.
    """

    n = len(str1)
    m = len(str2)

    if (n > m):  # Make sure n <= m, to use O(min(n,m)) space
        str1, str2 = str2, str1
        n, m = m, n
        swapped = True
    else:
        swapped = False

    current = (n + 1) * [0]

    com_len = 0
    com_ans1 = -1
    com_ans2 = -1

    for i in range(m):
        previous = current
        current = (n + 1) * [0]

        for j in range(n):
            if (str1[j] != str2[i]):
                current[j] = 0
            else:
                current[j] = previous[j - 1] + 1
                if (current[j] > com_len):
                    com_len = current[j]
                    com_ans1 = j
                    com_ans2 = i

    com1 = str1[com_ans1 - com_len + 1:com_ans1 + 1]
    com2 = str2[com_ans2 - com_len + 1:com_ans2 + 1]


    # Remove common substring from input strings
    #
    str1 = str1[:com_ans1 - com_len + 1] + str1[1 + com_ans1:]
    str2 = str2[:com_ans2 - com_len + 1] + str2[1 + com_ans2:]

    if (swapped == True):
        return com1, com_len, str2, str1
    else:
        return com1, com_len, str1, str2


def OntoLCS(val1, val2):
    """Compare two field values using the longest common substring approximate
       string comparator.
    """

    # Check if one of the values is a missing value
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    common_divisor = None
    min_common_len = 2
    p = 0.6

    common_divisor = 'average'

    # Calculate longest common substring similarity value - - - - - - - - - - -
    #
    len1, len2 = len(val1), len(val2)

    if (common_divisor == 'average'):
        divisor = 0.5 * (len1 + len2)  # Compute average string length
    elif (common_divisor == 'shortest'):
        divisor = min(len1, len2)
    else:  # Longest
        divisor = max(len1, len2)

    # Quick check if below threshold - - - - - - - - - - - - - - - - - - - -
    #
    max_common_len = min(len1, len2)

    w = float(max_common_len) / float(divisor)

    if (w < threshold):  # Similariy is smaller than threshold
        w = disagree_weight

    else:
        # Iterative calculation of longest common substring until strings to s
        #
        w = 0.0

        for (s1, s2) in [(val1, val2), (val2, val1)]:

            com_str, com_len, s1, s2 = __do_lcs__(s1, s2)  # Find initial LCS

            total_com_str = com_str
            total_com_len = com_len

            while (com_len >= min_common_len):
                com_str, com_len, s1n, s2n = __do_lcs__(s1, s2)

                if (com_len >= min_common_len):
                    total_com_str += com_str
                    total_com_len += com_len
                    s1, s2 = s1n, s2n

            w += float(total_com_len) / float(divisor)

        w /= 2.0

        assert (w >= 0.0), 'Longest common substring: Similarity weight < 0.0'
        assert (w <= 1.0), 'Longest common substring: Similarity weight > 1.0'

    return w


def SeqMatch(val1, val2):
    """Compare two field values using the Python standard library 'difflib'
       sequence matcher approximate string comparator.
    """

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate sequence matcher similarity value - - - - - - - - - - - - - - -
    #
    seq_matcher_1 = difflib.SequenceMatcher(None, val1, val2)
    seq_matcher_2 = difflib.SequenceMatcher(None, val2, val1)

    w = (seq_matcher_1.ratio() + seq_matcher_2.ratio()) / 2.0  # Calc average

    assert (w >= 0.0), 'Python sequence matcher: Similarity weight < 0.0'
    assert (w <= 1.0), 'Python sequence matcher: Similarity weight > 1.0'

    return w


def PosQGram(val1, val2):
    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    q = 3
    max_dist = 5
    common_divisor = 'average'
    padded = True
    QGRAM_START_CHAR = chr(1)
    QGRAM_END_CHAR = chr(2)

    # Calculate number of q-grams in strings (plus start and end characters)
    #
    if (padded == True):
        num_qgram1 = len(val1) + q - 1
        num_qgram2 = len(val2) + q - 1
    else:
        num_qgram1 = max(len(val1) - (q - 1), 0)  # Make sure its not negative
        num_qgram2 = max(len(val2) - (q - 1), 0)

    # Check if there are q-grams at all from both strings - - - - - - - - - - -
    # (no q-grams if length of a string is less than q)
    #
    if ((padded == False) and (min(num_qgram1, num_qgram2) == 0)):
        w = disagree_weight

    else:
        if (common_divisor == 'average'):  # Calculate the divisor
            divisor = 0.5 * (num_qgram1 + num_qgram2)
        elif (common_divisor == 'shortest'):
            divisor = min(num_qgram1, num_qgram2)
        else:  # Longest
            divisor = max(num_qgram1, num_qgram2)

        # Use number of q-grams to quickly check if below threshold - - - - - - -
        #
        max_common_qgram = min(num_qgram1, num_qgram2)  # Max possible q-grams
        w = float(max_common_qgram) / float(divisor)  # ... in common

        if (w < threshold):  # Similariy is smaller than threshold
            w = disagree_weight

        else:

            # Add start and end characters (padding) - - - - - - - - - - - - - - -
            #
            if (padded == True):
                qgram_str1 = (q - 1) * QGRAM_START_CHAR + val1 + (q - 1) * \
                                                                      QGRAM_END_CHAR
                qgram_str2 = (q - 1) * QGRAM_START_CHAR + val2 + (q - 1) * \
                                                                      QGRAM_END_CHAR
            else:
                qgram_str1 = val1
                qgram_str2 = val2

            # Make a list of q-grams for both strings - - - - - - - - - - - - - - -
            #
            qgram_list1 = \
                [(qgram_str1[i:i + q], i) for i in range(len(qgram_str1) - (q - 1))]
            qgram_list2 = \
                [(qgram_str2[i:i + q], i) for i in range(len(qgram_str2) - (q - 1))]

            # Get common q-grams  - - - - - - - - - - - - - - - - - - - - - - - - -
            #
            common = 0

            if (num_qgram1 < num_qgram2):  # Count using the shorter q-gram list
                short_qgram_list = qgram_list1
                long_qgram_list = qgram_list2
            else:
                short_qgram_list = qgram_list2
                long_qgram_list = qgram_list1

            for pos_q_gram in short_qgram_list:
                (q_gram, pos) = pos_q_gram

                pos_range = range(max(pos - max_dist, 0), pos + max_dist + 1)

                for test_pos in pos_range:
                    test_pos_q_gram = (q_gram, test_pos)
                    if (test_pos_q_gram in long_qgram_list):
                        common += 1
                        long_qgram_list.remove(test_pos_q_gram)  # Remove counted q-gram
                        break

            w = float(common) / float(divisor)

            assert (w >= 0.0), 'Positional Q-gram: Similarity weight < 0.0'
            assert (w <= 1.0), 'Positional Q-gram: Similarity weight > 1.0'

    return w


def __delete_cost__(char1, char2):
    groupsof_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 0, 'f': 1, 'g': 2, 'h': 7,
                     'i': 0, 'j': 2, 'k': 2, 'l': 4, 'm': 5, 'n': 5, 'o': 0, 'p': 1,
                     'q': 2, 'r': 6, 's': 2, 't': 3, 'u': 0, 'v': 1, 'w': 7, 'x': 2,
                     'y': 0, 'z': 2, '{': 7}

    if (char1 == char2):
        return 0

    code1 = groupsof_dict.get(char1, -1)  # -1 is not a char
    code2 = groupsof_dict.get(char2, -2)  # -2 if not a char

    if (code1 == code2) or (code2 == 7):  # Same or silent
        return 2  # Small difference costs
    else:
        return 3


def Editex(val1, val2):
    """Compare two field values using the editex approximate string comparator.
    """
    groupsof_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 0, 'f': 1, 'g': 2, 'h': 7,
                     'i': 0, 'j': 2, 'k': 2, 'l': 4, 'm': 5, 'n': 5, 'o': 0, 'p': 1,
                     'q': 2, 'r': 6, 's': 2, 't': 3, 'u': 0, 'v': 1, 'w': 7, 'x': 2,
                     'y': 0, 'z': 2, '{': 7}

    BIG_COSTS = 3  # If characters are not in same group
    SML_COSTS = 2

    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate editex similarity value - - - - - - - - - - - - - - - - - - - -
    #
    n, m = len(val1), len(val2)

    if (n > m):  # Make sure n <= m, to use O(min(n,m)) space
        str1 = val2.lower()
        str2 = val1.lower()
        n, m = m, n
    else:
        str1 = val1.lower()
        str2 = val2.lower()

    if (' ' in str1):
        str1 = str1.replace(' ', '{')
    if (' ' in str2):
        str2 = str2.replace(' ', '{')

    row = [0] * (m + 1)  # Generate empty cost matrix
    F = []
    for i in range(n + 1):
        F.append(row[:])

    F[1][0] = BIG_COSTS  # Initialise first row and first column of
    F[0][1] = BIG_COSTS  # cost matrix

    sum = BIG_COSTS
    for i in range(2, n + 1):
        sum += __delete_cost__(str1[i - 2], str1[i - 1])
        F[i][0] = sum

    sum = BIG_COSTS
    for j in range(2, m + 1):
        sum += __delete_cost__(str2[j - 2], str2[j - 1])
        F[0][j] = sum

    for i in range(1, n + 1):

        if (i == 1):
            inc1 = BIG_COSTS
        else:
            inc1 = __delete_cost__(str1[i - 2], str1[i - 1])

        for j in range(1, m + 1):

            if (j == 1):
                inc2 = BIG_COSTS
            else:
                inc2 = __delete_cost__(str2[j - 2], str2[j - 1])

            if (str1[i - 1] == str2[j - 1]):
                diag = 0
            else:
                code1 = groupsof_dict.get(str1[i - 1], -1)  # -1 is not a char
                code2 = groupsof_dict.get(str2[j - 1], -2)  # -2 if not a char

                if (code1 == code2):  # Same phonetic group
                    diag = SML_COSTS
                else:
                    diag = BIG_COSTS

            F[i][j] = min(F[i - 1][j] + inc1, F[i][j - 1] + inc2, F[i - 1][j - 1] + diag)

    w = 1.0 - float(F[n][m]) / float(max(F[0][m], F[n][0]))

    if (w < 0.0):
        w = 0.0

    assert (w >= 0.0), 'Editex: Similarity weight < 0.0'
    assert (w <= 1.0), 'Editex: Similarity weight > 1.0'

    return w


def bagDist(val1, val2):
    """Compare two field values using the bag distance approximate string
       comparator.
    """

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate bag distance similarity value - - - - - - - - - - - - - - - - -
    #
    n = len(val1)
    m = len(val2)

    list1 = list(val1)
    list2 = list(val2)

    for ch in val1:
        if (ch in list2):
            list2.remove(ch)

    for ch in val2:
        if (ch in list1):
            list1.remove(ch)

    b = max(len(list1), len(list2))

    w = 1.0 - float(b) / float(max(n, m))

    assert (w >= 0.0), 'Bag distance: Similarity weight < 0.0'
    assert (w <= 1.0), 'Bag distance: Similarity weight > 1.0'

    # if (self.do_caching == True):  # Put values pair into the cache
    #     self.__put_into_cache__(val1, val2, w)

    return w


def compress(val1, val2):
    """Compare two field values using the compressor approximate string
       comparator.
    """

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate the compressor similarity value - - - - - - - - - - - - - - - -
    #
    compressor = 'zlib'

    if (compressor == 'zlib'):
        c1 = float(len(zlib.compress(val1)))
        c2 = float(len(zlib.compress(val2)))
        c12 = 0.5 * (len(zlib.compress(val1 + val2)) + len(zlib.compress(val2 + val1)))

    elif (compressor == 'bz2'):
        c1 = float(len(bz2.compress(val1)))
        c2 = float(len(bz2.compress(val2)))
        c12 = 0.5 * (len(bz2.compress(val1 + val2)) + len(bz2.compress(val2 + val1)))

    # else:  # More to be added later

    if (c12 == 0.0):
        w = disagree_weight  # Maximal distance

    else:  # Calculate normalised compression distance
        w = 1.0 - (c12 - min(c1, c2)) / max(c1, c2)

        if (w < 0.0):
            logging.warning('%s Compression based comparison smaller than 0.0 ' % \
                            (compressor) + 'with strings "%s" and "%s": %.3f' \
                            % (val1, val2, w))
            w = 0.0

        assert (w >= 0.0), 'Compression: Similarity weight < 0.0'
        assert (w <= 1.0), 'Compression: Similarity weight > 1.0'


    return w


def compressbz2(val1, val2):
    """Compare two field values using the compressor approximate string
       comparator.
    """

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate the compressor similarity value - - - - - - - - - - - - - - - -
    #
    compressor = 'bz2'

    if (compressor == 'zlib'):
        c1 = float(len(zlib.compress(val1)))
        c2 = float(len(zlib.compress(val2)))
        c12 = 0.5 * (len(zlib.compress(val1 + val2)) + len(zlib.compress(val2 + val1)))

    elif (compressor == 'bz2'):
        c1 = float(len(bz2.compress(val1)))
        c2 = float(len(bz2.compress(val2)))
        c12 = 0.5 * (len(bz2.compress(val1 + val2)) + len(bz2.compress(val2 + val1)))

    # else:  # More to be added later

    if (c12 == 0.0):
        w = disagree_weight  # Maximal distance

    else:  # Calculate normalised compression distance
        w = 1.0 - (c12 - min(c1, c2)) / max(c1, c2)

        if (w < 0.0):
            logging.warning('%s Compression based comparison smaller than 0.0 ' % \
                            (compressor) + 'with strings "%s" and "%s": %.3f' \
                            % (val1, val2, w))
            w = 0.0

        assert (w >= 0.0), 'Compression: Similarity weight < 0.0'
        assert (w <= 1.0), 'Compression: Similarity weight > 1.0'


    return w


def editDistance(val1, val2):
    """Compare two field values using the edit-distance (or Levenshtein)
       approximate string comparator.
    """

    # Check if one of the values is a missing value
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate edit distance similarity value - - - - - - - - - - - - - - - -
    #
    n = len(val1)
    m = len(val2)
    max_len = max(n, m)

    # Quick check if edit distance is below threshold - - - - - - - - - - - - -
    #
    len_diff = abs(n - m)
    w = 1.0 - float(len_diff) / float(max_len)

    if (w < threshold):  # Similariy is smaller than threshold
        w = disagree_weight

    else:  # Calculate the maximum distance possible with this threshold
        max_dist = (1.0 - threshold) * max_len

        if (n > m):  # Make sure n <= m, to use O(min(n,m)) space
            str1 = val2
            str2 = val1
            n, m = m, n
        else:
            str1 = val1
            str2 = val2

        current = range(n + 1)

        w = -1  # Set weight to an illegal value (so it can be chacked later)

        for i in range(1, m + 1):
            previous = current
            current = [i] + n * [0]
            str2char = str2[i - 1]

            for j in range(1, n + 1):
                substitute = previous[j - 1]
                if (str1[j - 1] != str2char):
                    substitute += 1

                # Get minimum of insert, delete and substitute
                #
                current[j] = min(previous[j] + 1, current[j - 1] + 1, substitute)

            if (min(current) > max_dist):  # Distance is already too large
                w = max(1.0 - float(max_dist + 1) / float(max_len), 0.0)
                break  # Exit loop

        if (w == -1):  # Weight has not been calculated
            w = 1.0 - float(current[n]) / float(max_len)

        assert (w >= 0.0), 'Edit distance: Similarity weight < 0.0'
        assert (w <= 1.0), 'Edit distance: Similarity weight > 1.0'

    return w


def KeyDiff(val1, val2):
    """Compare two field values using the key difference field comparator.
    """

    # Check if one of the values is a missing value
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    # Calculate the key difference - - - - - - - - - - - - - - - - - - - - - -
    #
    str1 = str(val1)  # Make sure values are strings
    str2 = str(val2)
    len1 = len(str1)
    len2 = len(str2)
    max_key_diff = 5

    # The initial number of errors is the difference in the string lengths
    #
    num_err = abs(len1 - len2)

    if (num_err > max_key_diff):
        return disagree_weight  # Too many different characters

    check_len = min(len1, len2)

    for i in range(check_len):  # Loop over positions in strings
        if (str1[i] != str2[i]):
            num_err += 1

    if (num_err >  max_key_diff):
        return disagree_weight  # Too many different characters

    # Get general or frequency based agreement weight
    #
    # agree_weight = self.__calc_freq_weights__(val1, val2)

    # Calculate partial agreement weight
    #
    return 1 - (float(num_err) / (max_key_diff + 1.0))


def ContainsString(val1, val2):
    """Compare two field values checking if the shorter value is contained in
       the longer value (both assumed to be strings).
    """
    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    len1 = len(val1)
    len2 = len(val2)

    if (len1 < len2):
        is_contained = (val1 in val2)
    else:
        is_contained = (val2 in val1)

    if (is_contained == True):
        return 1

    else:
        return 0

def TruncateString(val1, val2):
    """Compare two field values using exact string comparator with truncated
       strings.
    """

    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    elif (val1 == val2):
        return 1

    str1 = str(val1)  # Make sure values are strings
    str2 = str(val2)

    num_char_compared = 10

    if (str1[:num_char_compared] == str2[:num_char_compared]):
        return 1

    else:
        return 0


def NumericAbs(val1, val2):
    """Compare two numerical field values and tolerate an absolute difference.

       If at least one of the two values to be compared is not a number, then
       the disagreement value is returned.
    """
    max_abs_diff = 60

    # Check if one of the values is a missing value
    #
    if (val1 == None) or (val1 == None):
        return 0

    elif (val1 == val2):
        return 1

    # Calculate the absolute difference - - - - - - - - - - - - - - - - - - - -
    #
    try:  # Check if the field values are numbers
        float_val1 = float(val1)
    except:
        return disagree_weight
    try:
        float_val2 = float(val2)
    except:
        return disagree_weight

    if (float_val1 == float_val2):
        return 1

    elif (max_abs_diff == 0.0):  # No absolute difference tolerated
        return 0  # Because values are different

    # Calculate absolute difference and weight  - - - - - - - - - - - - - - - -
    #
    abs_diff = abs(float_val1 - float_val2)

    if (abs_diff > max_abs_diff):  # Absolute difference too large
        return disagree_weight

    # Get general or frequency based agreement weight
    #
    agree_weight = 1

    # Calculate partial agreement weight
    #
    return agree_weight - (abs_diff / (max_abs_diff + 1.0)) * \
                          (agree_weight + abs(disagree_weight))


def NumericPerc(val1, val2):
    """Compare two numerical field values and tolerate a percentage difference.

       If at least one of the two values to be compared is not a number, then
       the disagreement value is returned.
    """
    max_perc_diff = 30
    # Check if one of the values is a missing value
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    elif (val1 == val2):
        return 1

    # Calculate the percentage difference - - - - - - - - - - - - - - - - - - -
    #
    try:  # Check if the field values are numbers, if not return disagreement
        float_val1 = float(val1)
    except:
        return disagree_weight
    try:
        float_val2 = float(val2)
    except:
        return disagree_weight

    if (float_val1 == float_val2):
        return 1

    elif (max_perc_diff == 0.0):  # No percentage difference tolerated
        return disagree_weight  # Because values are different

    # Calculate percentage difference and weight  - - - - - - - - - - - - - - -
    #
    perc_diff = 100.0 * abs(float_val1 - float_val2) / \
                max(abs(float_val1), abs(float_val2))

    if (perc_diff > max_perc_diff):  # Percentage difference too large
        return disagree_weight

    # Get general or frequency based agreement weight
    #
    agree_weight = 1

    # Calculate partial agreement weight
    #
    return agree_weight - (perc_diff / (max_perc_diff + 1.0)) * \
                          (agree_weight + abs(disagree_weight))


def __do_lcs2__(str1, str2):
    """Method to extract longest common substring from the two input strings.
       Returns the common substring, its length, and the two input strings with
       the common substring removed.

       Should not be used from outside the module.
    """

    n = len(str1)
    m = len(str2)

    if (n > m):  # Make sure n <= m, to use O(min(n,m)) space
        str1, str2 = str2, str1
        n, m = m, n
        swapped = True
    else:
        swapped = False

    current = (n + 1) * [0]

    com_len = 0
    com_ans1 = -1
    com_ans2 = -1

    for i in range(m):
        previous = current
        current = (n + 1) * [0]

        for j in range(n):
            if (str1[j] != str2[i]):
                current[j] = 0
            else:
                current[j] = previous[j - 1] + 1
                if (current[j] > com_len):
                    com_len = current[j]
                    com_ans1 = j
                    com_ans2 = i

    com1 = str1[com_ans1 - com_len + 1:com_ans1 + 1]
    com2 = str2[com_ans2 - com_len + 1:com_ans2 + 1]

    if (com1 != com2):
        logging.exception('LCS: Different common substrings: %s / %s in ' % \
                          (com1, com2) + 'original strings: %s / %s' % \
                          (str1, str2))
        raise Exception

    # Remove common substring from input strings
    #
    str1 = str1[:com_ans1 - com_len + 1] + str1[1 + com_ans1:]
    str2 = str2[:com_ans2 - com_len + 1] + str2[1 + com_ans2:]

    if (swapped == True):
        return com1, com_len, str2, str1
    else:
        return com1, com_len, str1, str2

def longcommonSeq(val1, val2):
    """Compare two field values using the longest common substring approximate
       string comparator.
    """
    min_common_len = 10
    # Check if one of the values is a missing value
    if (len(val1) == 0) or (len(val2) == 0):
        return 0

    if (val1 == val2):
        return 1

    # Calculate longest common substring similarity value - - - - - - - - - - -
    #
    len1, len2 = len(val1), len(val2)
    common_divisor = 'average'

    if (common_divisor == 'average'):
        divisor = 0.5 * (len1 + len2)  # Compute average string length
    elif (common_divisor == 'shortest'):
        divisor = min(len1, len2)
    else:  # Longest
        divisor = max(len1, len2)

    # Quick check if below threshold - - - - - - - - - - - - - - - - - - - -
    #
    max_common_len = min(len1, len2)

    w = float(max_common_len) / float(divisor)

    if (w < threshold):  # Similariy is smaller than threshold
        w = disagree_weight

    else:
        # Iterative calculation of longest common substring until strings to s
        #
        w = 0.0

        for (s1, s2) in [(val1, val2), (val2, val1)]:

            com_str, com_len, s1, s2 = __do_lcs2__(s1, s2)  # Find initial LCS

            total_com_str = com_str
            total_com_len = com_len

            while (com_len >= min_common_len):
                com_str, com_len, s1n, s2n = __do_lcs2__(s1, s2)

                if (com_len >= min_common_len):
                    total_com_str += com_str
                    total_com_len += com_len
                    s1, s2 = s1n, s2n

            w += float(total_com_len) / float(divisor)

        w /= 2.0

        assert (w >= 0.0), 'Longest common substring: Similarity weight < 0.0'
        assert (w <= 1.0), 'Longest common substring: Similarity weight > 1.0'


    return w

