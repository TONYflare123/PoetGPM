import spacy
from collections import defaultdict, Counter
import random

class PoetGPM:
    """
    A creative poetry generator using a Hybrid NG-GPM model.
    - Creative Layer: Bigram model for maximum novelty and unpredictability.
    - Grammar Layer: Trigram GPM to ensure grammatical coherence.
    """
    # Use (en_core_web_sm) instead, if you want to
    def __init__(self, spacy_model: str = "en_core_web_md"):
        self.nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
        # Creative Layer (Bigram: n=2)
        self.bigram_counts = defaultdict(Counter)
        # Grammar Layer (Trigram GPM: n=3 for POS)
        self.pos_trigram_model = defaultdict(Counter) # Stores counts
        self.pos_probability_model = {} # Stores probabilities

    def train(self, corpus_text: str):
        """Trains both the creative bigram model and the grammatical trigram GPM."""
        doc = self.nlp(corpus_text)
        
        for sent in doc.sents:
            words = [token.text for token in sent] # Keep case for poetry
            pos_tags = [token.tag_ for token in sent]
            
            # Train Creative Bigram Model
            for i in range(len(words) - 1):
                context = words[i]
                next_word = words[i+1]
                self.bigram_counts[context][next_word] += 1
                
            # Train Grammatical Trigram GPM
            for i in range(len(pos_tags) - 2):
                context = (pos_tags[i], pos_tags[i+1])
                next_pos = pos_tags[i+2]
                self.pos_trigram_model[context][next_pos] += 1

        # Convert GPM counts to probabilities
        for context, next_tags in self.pos_trigram_model.items():
            total = sum(next_tags.values())
            self.pos_probability_model[context] = {tag: count / total for tag, count in next_tags.items()}

    def get_creative_candidates(self, word: str, top_n: int = 15):
        """Gets the top N most frequent next words from the bigram model.
        Returns a list of (candidate_word, frequency) tuples."""
        candidates = self.bigram_counts.get(word, Counter())
        return candidates.most_common(top_n)

    def score_grammar(self, prev_word: str, prev_pos: str, candidate_word: str) -> float:
        """Scores a candidate word based on the POS context of the previous two words."""
        # Get POS context: We need the POS of the previous word and the one before it.
        # For the first word in a line, context might be weak. We'll handle that.
        doc = self.nlp(candidate_word)
        candidate_pos = doc[0].tag_
        
        # The POS context for the trigram is (POS of word_{i-2}, POS of word_{i-1})
        # We need to get the POS of the previous word. We'll accept it as passed in.
        pos_context = (prev_pos, candidate_pos)
        
        # Now, what's the probability of the candidate's POS given this context?
        # Note: This is a slightly different use than before but leverages the same model.
        # We are checking P( POS_candidate | (POS_prev, POS_prev_prev) )
        prob = self.pos_probability_model.get(pos_context, {}).get(candidate_pos, 0.0)
        return prob

    def generate_poetic_line(self, seed_word: str, line_length: int = 6):
        """Generates a single line of poetry."""
        current_word = seed_word
        line = [current_word]
        
        # Get POS of the seed word to start context
        seed_doc = self.nlp(seed_word)
        prev_pos = seed_doc[0].tag_
        prev_prev_pos = None  # We won't have this for the first word

        for _ in range(line_length - 1):
            # Step 1: Get creative suggestions from Bigram model
            candidates = self.get_creative_candidates(current_word, top_n=20)
            if not candidates:
                break
                
            # Step 2: Score each candidate based on grammatical probability
            scored_candidates = []
            for cand, freq in candidates:
                # For the GPM, we need the POS of the previous word and the one before it.
                # For the first iteration, prev_prev_pos is None, so we skip GPM?
                # Let's use a simpler check for the second word: use only the previous POS.
                grammar_score = self.score_grammar(current_word, prev_pos, cand)
                # Combine score: Bigram frequency * Grammatical probability
                total_score = freq * (grammar_score + 0.01)  # Add small smoothing factor
                scored_candidates.append((cand, total_score, freq, grammar_score))
            
            # Step 3: Choose a candidate. We can choose the highest score,
            # or introduce randomness for more creativity.
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            # Take a random choice from the top 3 to balance creativity and grammar
            if len(scored_candidates) > 3:
                best_candidates = scored_candidates[:3]
            else:
                best_candidates = scored_candidates
                
            if not best_candidates:
                break
                
            next_word, _, _, _ = random.choice(best_candidates)
            
            # Step 4: Update state for next iteration
            # Get POS of the chosen word for the next GPM context
            next_doc = self.nlp(next_word)
            next_pos = next_doc[0].tag_
            # For the next round, the "previous POS" becomes our current candidate's POS
            prev_prev_pos = prev_pos
            prev_pos = next_pos
            
            line.append(next_word)
            current_word = next_word

        return " ".join(line)

# Example Usage & Training on a Poetic Corpus
corpus = """
The sun did rise so clear and bright, upon the dewy morn.
The birds they sang a merry tune, from branches they adorn.
My heart it beats a hopeful drum, for love that is new-born.
A shadow falls across the land, a warning of a storm.
The waves did crash upon the shore, a wild and fierce form.
Of golden fields and endless skies, a world that feels so warm.
Remember me when I am gone, to lands you know not where.
I give my heart without a sigh, to you I do declare.
"""

print("Training the Poet-GPM on a lyrical corpus...")
poet = PoetGPM()
poet.train(corpus)

print("\nGenerating poetic lines:")
print(f"1. {poet.generate_poetic_line('The', 200)}")



#print(f"2. {poet.generate_poetic_line('My', 20)}")
#print(f"3. {poet.generate_poetic_line('Remember', 20)}")
#print(f"4. {poet.generate_poetic_line('A', 20)}")
