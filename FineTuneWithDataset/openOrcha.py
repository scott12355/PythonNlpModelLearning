# Import necessary libraries
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from datasets import load_dataset
from evaluate import load
import numpy as np
from flask import Flask, request, jsonify


# Load the OpenOrca dataset
dataset = load_dataset("Open-Orca/OpenOrca")
# Select only 10000 rows
dataset = dataset['train'].select(range(10000))


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

def preprocess_function(examples):
    inputs = ["question: " + example for example in examples["question"]]
    targets = [example if len(example)> 0 else "I don't know" for example in examples["response"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding=True, return_tensors="np")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding=True, return_tensors="np")
    model_inputs["labels"] = labels["input_ids"].astype(np.int32)
    model_inputs["decoder_input_ids"] = labels["input_ids"].astype(np.int32)
    model_inputs["input_ids"] = model_inputs["input_ids"].astype(np.int32)
    model_inputs["attention_mask"] = model_inputs["attention_mask"].astype(np.int32)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format("tf")

# Prepare the TensorFlow dataset
from transformers import DataCollatorForSeq2Seq
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

tf_dataset = tokenized_dataset.to_tf_dataset(
    columns=["input_ids", "attention_mask", "labels", "decoder_input_ids"],
    shuffle=True,
    batch_size=128,
    drop_remainder=True  # ensures all batches have the same size
)

for batch in tf_dataset.take(1):
    print(batch['input_ids'].shape)
    print(batch['attention_mask'].shape)
    print(batch['decoder_input_ids'].shape)
    print(batch['labels'].shape)


# Freeze all layers except the final head
for layer in model.layers[:-1]:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()

model.fit(tf_dataset, epochs=2) 

model.save_pretrained("./flan_t5_finetuned")