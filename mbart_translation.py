import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from tqdm import tqdm

# Load and prepare data
data = pd.read_csv("../poetry_data.csv")
data.dropna(inplace=True)
data = data[['Hindi Poetry', 'English Poetry']]
data.columns = ['input_text', 'target_text']

# Add start/end tokens
data['input_text'] = data['input_text'].apply(lambda x: x.strip())
data['target_text'] = data['target_text'].apply(lambda x: x.strip())

# Load mBART model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set tokenizer for Hindi as source and English as target
tokenizer.src_lang = "hi_IN"
target_lang = "en_XX"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Translate in batches
translated_texts = []
for sentence in tqdm(data['input_text'], desc="Translating"):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
        max_length=50
    )
    translation = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    translated_texts.append(translation)

# Save results
data['mbart_translation'] = translated_texts
data.to_csv("mbart_translated_poetry.csv", index=False)

print("✅ Translation complete. Saved to mbart_translated_poetry.csv")


sample_hindi = "चाँदनी रात में सपने बुनता हूँ।"

inputs = tokenizer(sample_hindi, return_tensors="pt").to(device)
output_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id[target_lang],
    max_length=50
)
sample_translation = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("\n🔍 Test Translation:")
print("Hindi: ", sample_hindi)
print("English:", sample_translation)