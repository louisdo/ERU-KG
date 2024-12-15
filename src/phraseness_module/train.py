from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BertTokenizer
import torch, json
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.optim import Adam
from transformers import AdamW
from tqdm import tqdm

BERT_CHECKPOINT = "google/bert_uncased_L-4_H-512_A-8" # "google-bert/bert-base-uncased"
TRAINING_DATA_PATH = "/scratch/lamdo/phraseness_module/data/data500k.json"
CHECKPOINT_FOLDER = "/scratch/lamdo/phraseness_module/checkpoints/phraseness_module_500k_v2"

NUM_EPOCHS=1
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WARMUP_STEPS = 5000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_text = """Unsupervised Open-domain Keyphrase Generation
In this work, we study the problem of unsupervised open-domain keyphrase generation,
where the objective is a keyphrase generation model that can be built without using
human-labeled data and can perform consistently across domains. To solve this problem,
we propose a seq2seq model that consists of
two modules, namely phraseness and informativeness module, both of which can be built in
an unsupervised and open-domain fashion. The
phraseness module generates phrases, while the
informativeness module guides the generation
towards those that represent the core concepts
of the text. We thoroughly evaluate our proposed method using eight benchmark datasets
from different domains. Results on in-domain
datasets show that our approach achieves stateof-the-art results compared with existing unsupervised models, and overall narrows the gap
between supervised and unsupervised methods down to about 16%. Furthermore, we
demonstrate that our model performs consistently across domains, as it overall surpasses
the baselines on out-of-domain datasets""".lower()


with open(TRAINING_DATA_PATH) as f:
    training_data = json.load(f)[:6000000]
    print("Training data length", len(training_data))

def mask_pad_tokens(labels, pad_token_id):
    # Create a mask where pad tokens are True
    mask = labels == pad_token_id
    
    # Replace pad tokens with -100
    labels = labels.masked_fill(mask, -100)
    
    return labels

def generate_noun_phrase(text, model, tokenizer, max_length=50):
    # Tokenize input text
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    # print(input_ids)
    
    with torch.no_grad():
        # Generate output
        output_ids = model.generate(input_ids, 
                                    max_new_tokens=50,
                                    num_beams = 20, 
                                    num_return_sequences=20,
                                    no_repeat_ngram_size=1)
    print(output_ids)
    
    # Decode the output
    noun_phrase = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return noun_phrase

# def lr_lambda(current_step):
#     if current_step < WARMUP_STEPS:
#         # Linear warm-up
#         return float(current_step) / float(max(1, WARMUP_STEPS))
#     else:
#         # Cosine annealing
#         return 0.5 * (1 + torch.cos(torch.pi * (current_step - WARMUP_STEPS) / (TOTAL_STEPS - WARMUP_STEPS)))

# Define custom dataset
class NounPhraseDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, noun_phrase = self.data[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True, return_tensors="pt", padding='max_length'
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()
        
        label_encoding = self.tokenizer(
            noun_phrase, max_length=16, truncation=True, return_tensors="pt", padding='max_length'
        )
        label_ids = label_encoding.input_ids.squeeze()
        label_attention_mask = label_encoding.attention_mask.squeeze()
        
        # Prepare decoder inputs
        decoder_input_ids = label_ids[:-1]
        decoder_attention_mask = label_attention_mask[:-1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_ids,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask
        }

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_CHECKPOINT)

print(tokenizer.pad_token_id, "Padding")

# Prepare data
train_dataset = NounPhraseDataset(training_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

TOTAL_STEPS = (len(train_dataset) / BATCH_SIZE) *NUM_EPOCHS

print("total training step", TOTAL_STEPS)


# Initialize the model
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(BERT_CHECKPOINT, BERT_CHECKPOINT).to(DEVICE)

bert2bert.config.bos_token_id = 101
bert2bert.config.decoder_start_token_id = 101
bert2bert.config.eos_token_id = 102
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

# Prepare optimizer
optimizer = AdamW(bert2bert.parameters(), lr=LEARNING_RATE)

# Create the learning rate scheduler
scheduler = CosineAnnealingLR(optimizer, T_max = TOTAL_STEPS, eta_min=2e-5)

# Training loop
bert2bert.train()
for epoch in range(NUM_EPOCHS):  # Number of epochs
    pbar = tqdm(train_loader, desc = f"Epoch {epoch + 1}")
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        decoder_input_ids = batch["decoder_input_ids"].to(DEVICE)
        decoder_attention_mask = batch["decoder_attention_mask"].to(DEVICE)

        labels = mask_pad_tokens(labels, tokenizer.pad_token_id)

        # print(labels[:1], decoder_attention_mask[:1])
        # print(input_ids.shape, attention_mask.shape, decoder_input_ids.shape, decoder_attention_mask.shape, labels.shape)
        
        # Forward pass with attention masks
        encoder_outputs = bert2bert.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        decoder_labels = labels[:, 1:].clone()
        decoder_labels[decoder_labels == tokenizer.pad_token_id] = -100

        decoder_outputs = bert2bert.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask
        )

        logits = decoder_outputs.logits

        # Compute loss manually
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), decoder_labels.view(-1), ignore_index=-100)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pbar.set_postfix({"Loss": loss.item()})

        if i % 200 == 0:
            with torch.no_grad():
                test_gen = generate_noun_phrase(test_text, model = bert2bert, tokenizer=tokenizer)
                print(test_gen)
        


bert2bert.save_pretrained(CHECKPOINT_FOLDER)
tokenizer.save_pretrained(CHECKPOINT_FOLDER)

print("Training completed.")