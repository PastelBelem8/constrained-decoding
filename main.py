# Text generation @ HuggingFace
# https://huggingface.co/blog/constrained-beam-search#constrained-beam-search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

encoder_input_str = "translate English to German: How old are you?"
input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

# ------------------------------------------------------------
# Beam search decoding @ HuggingFace
# ------------------------------------------------------------
outputs = model.generate(
    input_ids,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# ------------------------------------------------------------
# Constrained decoding @ HuggingFace
# ------------------------------------------------------------
# issue: https://github.com/huggingface/transformers/issues/14081#issuecomment-1004479944
# based on Li et al. (2020) on Guided Generation of Cause and Effect
# accepts: positive constrained decoding
# HuggingFace's algorithm
#
force_words = ["Sie"]
force_words_ids = tokenizer(force_words, add_special_tokens=False).input_ids

outputs = model.generate(
    input_ids,
    force_words_ids=force_words_ids,
    num_beams=10,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)


print("Output:\n" + 100 * '-')
print(tokenizer.decode(outputs[0], skip_special_tokens=True))