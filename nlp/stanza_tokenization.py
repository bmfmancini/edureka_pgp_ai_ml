#!/usr/bin/env python3

import stanza

# Load English pipeline
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,depparse,constituency",
    tokenize_pretokenized=False
)

def parsing_by_stanza(text):
    """
    Process input text using Stanza NLP
    """
    doc = nlp(text)

    for sent in doc.sentences:
        print("\n--- Current Sentence ---")
        print(sent.text)

        print("\nTokens:")
        for word in sent.words:
            print(
                f"  text={word.text:<12} "
                f"lemma={word.lemma:<12} "
                f"upos={word.upos:<6} "
                f"head={word.head:<3} "
                f"deprel={word.deprel}"
            )

        print("\nDependency Parse:")
        for word in sent.words:
            if word.head != 0:
                head_word = sent.words[word.head - 1].text
            else:
                head_word = "ROOT"
            print(f"  {word.text} --> {head_word} ({word.deprel})")

        print("\nConstituency Parse:")
        print(sent.constituency)

# Example usage
if __name__ == "__main__":
    text = "%LINEPROTO-5-UPDOWN: Line protocol on Interface GigabitEthernet1/0/20, changed state to down"
    parsing_by_stanza(text)
