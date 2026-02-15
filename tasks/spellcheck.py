"""
Task intended to teach nanochat spelling correction in English sentences.

The task generates synthetic sentences with misspelled words and asks the model to correct the entire sentence.
This helps build spelling knowledge and correction instincts in context.

Example:
User: "Correct the spelling: I like to eat aple."
Assistant: "I like to eat apple."
"""

import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A list of 370K English words of large variety
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
# A number bigger than 370K to separate train and test random seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000

# Sentence templates for embedding misspelled words (using {} for positional formatting)
SENTENCE_TEMPLATES = [
    "The word is {}.",
    "I like {}.",
    "What is {}?",
    "This is a {}.",
    "He said {}.",
    "She loves {}.",
    "We need {}.",
    "They have {}.",
    "My {} is here.",
    "Your {} looks good.",
    "Can you see the {}?",
    "The {} is big.",
    "I eat {} every day.",
    "The {} runs fast.",
    "She sings {}.",
    "He plays {}.",
    "We go to {}.",
    "They build {}.",
    "My favorite is {}.",
    "The {} is blue.",
    "I like {} and {}.",
    "The {} is {}.",
    "She has {} and {}.",
    "We eat {} with {}.",
    "He plays {} on {}.",
]

# User message templates for data augmentation (English only)
USER_MSG_TEMPLATES = [
    "Correct the spelling: {sentence}",
    "Fix the spelling: {sentence}",
    "Correct this spelling: {sentence}",
    "Spell this correctly: {sentence}",
    "What's the correct spelling in: {sentence}",
    "Correct the sentence: {sentence}",
    "Fix this sentence: {sentence}",
    "Correctly spell: {sentence}",
    "The correct version is: {sentence}",  # Note: this is a trick, but model should still correct
    "Correct spelling: {sentence}",
]


def generate_misspelling(word):
    """
    Generate a simple misspelling of the word by introducing a random error.
    Returns the misspelled word, or the original if no change.
    """
    if len(word) <= 3:
        # Too short to misspell meaningfully
        return word

    word_list = list(word)
    error_type = random.choice(['swap', 'delete', 'insert'])

    if error_type == 'swap' and len(word) > 1:
        # Swap two adjacent letters
        pos = random.randint(0, len(word) - 2)
        word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
    elif error_type == 'delete' and len(word) > 1:
        # Delete a random letter
        pos = random.randint(0, len(word) - 1)
        word_list.pop(pos)
    elif error_type == 'insert':
        # Insert a random letter
        pos = random.randint(0, len(word))
        letter = random.choice(LETTERS)
        word_list.insert(pos, letter)

    misspelled = ''.join(word_list)
    # If somehow the same, return original (rare case)
    return misspelled if misspelled != word else word


class SpellCheck(Task):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellCheck split must be train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        # Filter to words that can be misspelled (length > 3)
        self.words = [w for w in words if len(w) > 3]

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # Pick a sentence template
        sentence_template = rng.choice(SENTENCE_TEMPLATES)

        # Count how many {} placeholders
        num_words = sentence_template.count('{}')

        # Pick words
        words = [rng.choice(self.words) for _ in range(num_words)]
        misspelled_words = []
        for word in words:
            misspelled = generate_misspelling(word)
            if misspelled == word:
                misspelled = generate_misspelling(word)
            misspelled_words.append(misspelled)

        # Create the misspelled sentence
        misspelled_sentence = sentence_template.format(*misspelled_words)
        # Create the correct sentence
        correct_sentence = sentence_template.format(*words)

        # Create user message with variations
        user_template = rng.choice(USER_MSG_TEMPLATES)
        # 30% chance to lowercase the template
        if rng.random() < 0.3:
            user_template = user_template.lower()
        user_msg = user_template.format(sentence=misspelled_sentence)
        if rng.random() < 0.5:  # 50% chance to add question mark
            user_msg += "?"

        # Create assistant response parts
        assistant_parts = []
        # Split sentences into words
        misspelled_words_list = misspelled_sentence.split()
        correct_words_list = correct_sentence.split()
        # Corrections for each word
        for misp, corr in zip(misspelled_words_list, correct_words_list):
            assistant_parts.append({"type": "text", "text": f"{misp}: {corr}\n"})
        # Final sentence
        assistant_parts.append({"type": "text", "text": f"\nThe corrected sentence is: {correct_sentence}"})

        # Return the conversation
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Check if the assistant response matches the corrected sentence.
        Similar to GSM8K's evaluation.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        # The last text part contains the final corrected sentence
        last_text_part = assistant_message['content'][-1]['text']
        # Extract the corrected sentence after "The corrected sentence is: "
        if "The corrected sentence is: " in last_text_part:
            correct_sentence = last_text_part.split("The corrected sentence is: ", 1)[1].strip()
        else:
            correct_sentence = last_text_part.strip()
        # Compare
        return assistant_response.strip().lower() == correct_sentence.lower()

    def reward(self, conversation, assistant_response):
        """Simple 0-1 reward."""
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)


if __name__ == "__main__":
    # Preview the SpellCheck task, first 10 examples
    task = SpellCheck()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 50)
        print(ex['messages'][0]['content'])
        print("-" * 50)
        # Assistant content is now a list of parts
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
        print()
        print("-" * 50)
