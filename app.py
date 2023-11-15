#!/usr/bin/env python
# coding: utf-8

# In[28]:


# imports
import os

# To supress the warnings and to print only the error logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import json

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os,time

tokenizer = ""

# Create lookup tables
word2idx, idx2word = "",""
sampled_captions = ""
cap_model = ""

model_weights_path = './img-cap-weights.h5'
# if os.path.exists(model_path) and os.path.exists('tokenizer.pickle') and os.path.exists('idx2word.pickle') and os.path.exists('sampled_captions.pickle'):
#     print("Loading the pre-trained model and components.")
#     cap_model = load_model(model_path)
    
#     with open('tokenizer.pickle', 'rb') as handle:
#         tokenizer = pickle.load(handle)
    
#     with open('idx2word.pickle', 'rb') as handle:
#         idx2word = pickle.load(handle)
#     with open('sampled_captions.pickle', 'rb') as handle:
#         sampled_captions = pickle.load(handle)
# else:
print("Training a new model.")
# cap_model = initialize_and_compile_model(EMBEDDING_DIM, UNITS, image_augmentation)
# history = fit_and_plot_model(cap_model, train_dataset, val_dataset, EPOCHS)
# cap_model.save(model_path)


# In[29]:


# Configurable variables
FOLDER_BASE_PATH = "./coco"
JSON_FILE_NAME = "captions_train2017.json"
JSON_FILE_PATH = f'{FOLDER_BASE_PATH}/annotations/{JSON_FILE_NAME}'
SAMPLES_SIZE = 500040
TOKEN_LENGTH = 40
VOCAB_LENGTH = 150000
# VOCAB_LENGTH = 15000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 9
PREDICTION_IMAGE_URL=[
    "https://res.cloudinary.com/cloudinary-marketing/images/w_1999,h_1333/f_auto,q_auto/v1686858145/Web_Assets/blog/blog_ai-image-captioning-2/blog_ai-image-captioning-2-png?_i=AA",
    "https://ca-times.brightspotcdn.com/dims4/default/5eebfa2/2147483647/strip/true/crop/2048x1152+0+0/resize/1200x675!/quality/75/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2F26%2Ffd%2F74151099e6dacfe365d58f751dde%2Fla-1548963786-oi2mpj9yrl-snap-image",
    "https://image.petmd.com/files/styles/863x625/public/petmd-eating-quirks.jpg",
    ]


# In[30]:


def load_and_prepare_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
        data = data['annotations']

    img_cap_pairs = []
    for sample in data:
        img_name = '%012d.jpg' % sample['image_id']
        img_path = f'{FOLDER_BASE_PATH}/images/train2017/{img_name}'
        
        # Check if the image file exists before adding it to the list
        if os.path.exists(img_path):
            img_cap_pairs.append([img_name, sample['caption']])
        else:
            print(f"Image not found for {img_name}. Skipping this sample.")

    captions_df = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
    captions_df['image'] = captions_df['image'].apply(lambda x: f'{FOLDER_BASE_PATH}/images/train2017/{x}')

    return captions_df


# In[31]:


def sample_data(captions_df, num_samples):
    sampled_captions = captions_df.sample(num_samples)
    return sampled_captions.reset_index(drop=True)


# In[32]:





import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images_with_captions_one_by_one(sampled_captions, num_images=10):
    sampled_data_for_display = sampled_captions.sample(num_images)
    
    for _ in range(num_images):
        sample = random.choice(sampled_captions.index)
        img_path, cap = sampled_captions.loc[sample, 'image'], sampled_captions.loc[sample, 'caption']
        
        plt.figure(figsize=(10, 10))
        img = mpimg.imread(img_path)
        plt.title(cap)
        plt.axis("off")
        plt.imshow(img)
        plt.show()
        
        input("Press Enter to display the next random image...") # Wait for user to press Enter before showing the next image


# Call the function to display images with captions
# display_images_with_captions_one_by_one(sampled_captions)
# return
# exit(1)
# exit
# In[33]:


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text

def preprocess_captions(captions_df):
    captions_df['caption'] = captions_df['caption'].apply(preprocess)
    return captions_df

def sample_random_row(captions_df):
    random_row = captions_df.sample(1).iloc[0]
    return random_row


# In[34]:





# In[35]:


# Sample a random row



# In[36]:


def create_and_adapt_tokenizer(captions_df, vocabulary_size, max_size):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=None,
        output_sequence_length=max_size)
    tokenizer.adapt(captions_df['caption'])

    return tokenizer


# In[37]:


def create_lookup_tables(tokenizer):
    vocabulary = tokenizer.get_vocabulary()
    word2idx = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocabulary)
    idx2word = tf.keras.layers.StringLookup(mask_token="", vocabulary=vocabulary, invert=True)

    return word2idx, idx2word


# In[38]:


# if os.path.exists(model_weights_path):
#     # Create and adapt tokenizer
#     # with open('tokenizer_state.pickle', 'rb') as handle:
#     #     tokenizer = pickle.load(handle)
#     with open('tokenizer_vocab.pkl', 'rb') as f:
#         vocab = pickle.load(f)

#         # Create a new TextVectorization layer
#         tokenizer = TextVectorization(max_tokens=len(vocab))
#         tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["placeholder"]).batch(1))

#         # Set the vocabulary
#         tokenizer.set_vocabulary(vocab)
#         vocab_length = len(tokenizer.get_vocabulary())
#         print(f"Vocabulary Length: {vocab_length}")
# else:



# In[39]:


def split_data_into_train_val(captions_df):
    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(captions_df['image'], captions_df['caption']):
        img_to_cap_vector[img].append(cap)

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    train_imgs, train_captions, val_imgs, val_captions = [], [], [], []

    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        train_imgs.extend([imgt] * capt_len)
        train_captions.extend(img_to_cap_vector[imgt])

    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        val_imgs.extend([imgv] * capv_len)
        val_captions.extend(img_to_cap_vector[imgv])

    return train_imgs, train_captions, val_imgs, val_captions


# In[40]:


# Split the data into train and validation sets



# In[41]:


def load_and_preprocess_data(img_path, caption, tokenizer):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    caption = tokenizer(caption)
    return img, caption


# In[42]:


def create_tf_datasets(imgs, captions, buffer_size, batch_size, tokenizer):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, captions))
    dataset = dataset.map(lambda item1, item2: load_and_preprocess_data(item1, item2, tokenizer),
                        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    return dataset


# In[43]:




# In[44]:


# Image augmentation
image_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomContrast(0.3),
])


# In[45]:


def create_encoder():
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    reshaped = tf.keras.layers.Reshape((-1, model.output.shape[-1]))(model.output)
    return tf.keras.models.Model(inputs=model.input, outputs=reshaped)


# In[46]:


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim)
        self.dense = tf.keras.layers.Dense(dim, activation="relu")

    def call(self, inp, training):
        normalized_input = self.norm1(inp)
        dense_output = self.dense(normalized_input)

        attn_output = self.att(query=dense_output, value=dense_output, key=dense_output, training=training)

        return self.norm2(dense_output + attn_output)


# In[47]:


class TokenPositionEmbed(tf.keras.layers.Layer):

    def __init__(self, vocab, dim, maxlen):
        super().__init__()
        self.token_embed = tf.keras.layers.Embedding(vocab, dim)
        self.pos_embed = tf.keras.layers.Embedding(maxlen, dim, input_shape=(None, maxlen))

    def call(self, input_ids):
        pos_ids = tf.expand_dims(tf.range(tf.shape(input_ids)[-1]), 0)
        return self.token_embed(input_ids) + self.pos_embed(pos_ids)


# In[48]:


class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, dims, units, heads_num):
        super().__init__()
        # Token and Positional Embedding: This layer maps each token to a vector of 'dims' dimensions. 
        # It also adds positional encoding to embed information about the position of each token in the sequence.
        self.embed = TokenPositionEmbed(tokenizer.vocabulary_size(), dims, TOKEN_LENGTH)

        # MultiHeadAttention layers: These layers allow the model to focus on different positions in the input sequence,
        # and they are crucial for understanding the context of each word in a sentence.
        self.attn1 = tf.keras.layers.MultiHeadAttention(num_heads=heads_num, key_dim=dims, dropout=0.1)
        self.attn2 = tf.keras.layers.MultiHeadAttention(num_heads=heads_num, key_dim=dims, dropout=0.1)

        # Layer Normalization: These layers normalize the inputs (subtracts the mean and divides by the standard deviation),
        # maintaining the mean and variance constant for each input across the layers.
        self.lnorm1 = tf.keras.layers.LayerNormalization()
        
        # Normalization layer 2
        self.lnorm2 = tf.keras.layers.LayerNormalization()

        # Normalization layer 3
        self.lnorm3 = tf.keras.layers.LayerNormalization()

        # Feed Forward Network (FFN): These are standard Dense layers which follow the MultiHeadAttention layers.
        self.ffn1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(dims)

        # Output layer: This is the final Dense layer with a softmax activation. It will output a probability distribution over the vocabulary for the next word prediction.
        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        # Dropout layers: These layers are used for regularization. They "drop out" a random set of activations in that layer by setting them to zero during training
        self.drop1 = tf.keras.layers.Dropout(0.3)
        self.drop2 = tf.keras.layers.Dropout(0.5)

    def call(self, in_ids, enc_out, training, mask=None):
        embeds = self.embed(in_ids)
        combined_mask = None
        padding_mask = None

        if mask is not None:
            combined_mask = tf.minimum(tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32), self.get_mask(embeds))
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)

        attn1 = self.attn1(query=embeds, value=embeds, key=embeds, attention_mask=combined_mask, training=training)
        out1 = self.lnorm1(embeds + attn1)

        attn2 = self.attn2(query=out1, value=enc_out, key=enc_out, attention_mask=padding_mask, training=training)
        out2 = self.lnorm2(out1 + attn2)

        ffn_out = self.drop1(self.ffn1(out2), training=training)
        ffn_out = self.lnorm3(self.ffn2(ffn_out) + out2)
        preds = self.out(self.drop2(ffn_out, training=training))

        return preds


    def get_mask(self, inputs):
        shape = tf.shape(inputs)
        mask = tf.cast(tf.range(shape[1])[:, tf.newaxis] >= tf.range(shape[1]), dtype="int32")
        return tf.tile(tf.reshape(mask, (1, shape[1], shape[1])), [shape[0], 1, 1])


# In[49]:


class ImgCapModel(tf.keras.Model):

    def __init__(self, cnn, enc, dec, img_aug=None):
        super().__init__()
        self.cnn = cnn
        self.enc = enc
        self.dec = dec
        self.img_aug = img_aug
        self.loss_tr = tf.keras.metrics.Mean(name="loss")
        self.acc_tr = tf.keras.metrics.Mean(name="accuracy")

    def call(self, inputs, training=False):
        # Assuming inputs is a list: [image_input, caption_input]
        img_input, caption_input = inputs
        print(caption_input)

        img_embed = self.cnn(img_input)
        encoder_output = self.enc(img_embed, training=training)

        # You need to include how the decoder uses caption_input and encoder_output
        # For example:
        decoder_output = self.dec(caption_input, encoder_output, training=training)

        # Return the final output
        return decoder_output


    def compute_loss_acc(self, img_embed, captions, training=True):
        encoder_output = self.enc(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.dec(
            y_input, encoder_output, training=True, mask=mask
        )

        # Calculate loss
        loss = self.loss(y_true, y_pred)
        loss_mask = tf.cast(mask, dtype=loss.dtype)
        loss *= loss_mask
        final_loss = tf.reduce_sum(loss) / tf.reduce_sum(loss_mask)

        # Calculate accuracy
        acc = tf.cast(tf.math.logical_and(mask, tf.equal(y_true, tf.argmax(y_pred, axis=2))), dtype=tf.float32)
        acc_mask = tf.cast(mask, dtype=tf.float32)
        final_acc = tf.reduce_sum(acc) / tf.reduce_sum(acc_mask)

        return final_loss, final_acc

    def compute_step(self, batch, training):
        imgs, captions = batch
        if self.img_aug and training:
            imgs = self.img_aug(imgs)
        loss, acc = self.compute_loss_acc(self.cnn(imgs), captions, training)
        return loss, acc

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            loss, acc = self.compute_step(batch, training=True)
            grads = tape.gradient(loss, self.enc.trainable_variables + self.dec.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.enc.trainable_variables + self.dec.trainable_variables))
        self.loss_tr.update_state(loss)
        self.acc_tr.update_state(acc)
        return {"loss": self.loss_tr.result(), "acc": self.acc_tr.result()}

    def test_step(self, batch):
        loss, acc = self.compute_step(batch, training=False)
        self.loss_tr.update_state(loss)
        self.acc_tr.update_state(acc)
        return {"loss": self.loss_tr.result(), "acc": self.acc_tr.result()}

    @property
    def metrics(self):
        return [self.loss_tr, self.acc_tr]


# In[50]:


def initialize_and_compile_model(embed_dim, units, img_aug):
    enc = EncoderLayer(embed_dim, 1)
    dec = DecoderLayer(embed_dim, units, 8)

    cnn = create_encoder()
    cap_model = ImgCapModel(cnn, enc, dec, img_aug)

    cap_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")
    )

    return cap_model


# In[51]:


def fit_and_plot_model(model, train_dataset, val_dataset, epochs):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping]
    )

    # # model.save("./img-cap")
    # vocab_length = len(tokenizer.get_vocabulary())
    # print(f"Vocabulary Length: {vocab_length}")
    # model.save_weights('./img-cap-weights.h5')

    # # with open('tokenizer_state.pickle', 'wb') as handle:
    # #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # # model.save_pretrained("./ic")
    # vocab = tokenizer.get_vocabulary()

    # # Save the vocabulary
    # with open('tokenizer_vocab.pkl', 'wb') as f:
    #     pickle.dump(vocab, f)
    # tokenizer_json = tokenizer.to_json()
    # with open('tokenizer.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.savefig("./img_caption/loss_graph.jpg")
    plt.show()

    plt.plot(history.history['acc'], label='train accuracy')
    plt.plot(history.history['val_acc'], label='validation Accuracy')
    plt.legend()
    plt.savefig("./img_caption/acc_graph.jpg")
    plt.show()

    return history


# In[52]:


# Initialize and compile model



# In[53]:

# if os.path.exists(model_weights_path):
#     # Define input shapes
#     image_input_shape = (299, 299, 3)  # Input shape for InceptionV3
#     caption_input_length = 40 #tokenizer.vocabulary_size() # Adjust based on your model's expected caption length

#     # Create dummy inputs
#     vocab_length = len(tokenizer.get_vocabulary())
#     print(f"Vocabulary Length: {vocab_length}")
#     dummy_img_input = tf.random.uniform(shape=[1, *image_input_shape], dtype=tf.float32)
#     dummy_caption_input = tf.random.uniform(shape=[1, 40], maxval=10322, dtype=tf.int64)

#     # Build the image augmentation model if it's part of your ImgCapModel
#     if cap_model.img_aug is not None:
#         dummy_aug_input = tf.random.uniform(shape=[1, *image_input_shape], dtype=tf.float32)
#         cap_model.img_aug(dummy_aug_input)

#     # Run a forward pass with dummy data to build the entire ImgCapModel
#     cap_model([dummy_img_input, dummy_caption_input], training=False)

#     # Load the weights
#     cap_model.load_weights(model_weights_path)
#     # cap_model.from_pretrained("./ic")
# else:
    # Fit the model and save the weights

    # cap_model.save_weights(model_weights_path)


# In[54]:

# # In[57]:
# with open('sampled_captions.pickle', 'wb') as handle:
#     pickle.dump(sampled_captions, handle, protocol=pickle.HIGHEST_PROTOCOL)
# # Save the tokenizer and idx2word
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('idx2word.pickle', 'wb') as handle:
#     pickle.dump(idx2word, handle, protocol=pickle.HIGHEST_PROTOCOL)



def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


# In[55]:


def gen_caption(path, model, tokenizer, idx2word, add_noise=False):
    img = load_image_from_path(path)

    if add_noise:
        img = img + tf.random.normal(img.shape) * 0.1
        img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))

    img_embed = model.cnn(img[None, ...])
    img_encoded = model.enc(img_embed, training=False)

    caption = '[start]'
    for i in range(TOKEN_LENGTH - 1):
        tokens = tokenizer([caption])[:, :-1]
        pred = model.dec(tokens, img_encoded, training=False, mask=tokens!=0)
        
        pred_word = idx2word(tf.argmax(pred[0, i])).numpy().decode('utf-8')
        if pred_word == '[end]':
            break

        caption += ' ' + pred_word

    return caption.replace('[start] ', '')


# In[56]:



def plot_prediction(predictionImageURL, model, tokenizer, idx2word, captions):

    for i in range(10):
        idx = random.randrange(0, len(captions))
        img_path = captions.iloc[idx].image

        pred_caption = gen_caption(img_path, model, tokenizer, idx2word)
        im = Image.open(img_path)
        plt.imshow(im)
        plt.title('Predicted Caption: ' + pred_caption)
        plt.axis('off')
        plt.savefig(f'./img_caption/prediction_on_train_data_{i}.jpg')
        plt.show()

    for i, url in enumerate(predictionImageURL):
        im = Image.open(requests.get(url, stream=True).raw)
        im = im.convert('RGB')
        im.save('imageFromURL_{}.jpg'.format(i))

        pred_caption = gen_caption('imageFromURL_{}.jpg'.format(i), model, tokenizer, idx2word)
        plt.imshow(im)
        plt.title('Predicted Caption: ' + pred_caption)
        plt.axis('off')
        plt.savefig('./img_caption/URL_predicted_caption_{}.png'.format(i))
        plt.show()




# In[58]:





# In[59]:


def plot_URL_prediction(predictionImageURL, model, tokenizer, idx2word, captions):
        
    for i, url in enumerate(predictionImageURL):
        im = Image.open(requests.get(url, stream=True).raw)
        im = im.convert('RGB')
        im.save('imageFromURL_{}.jpg'.format(i))

        pred_caption = gen_caption('imageFromURL_{}.jpg'.format(i), model, tokenizer, idx2word)
        plt.imshow(im)
        plt.title('Predicted Caption: ' + pred_caption)
        plt.axis('off')
        #plt.savefig('./img_caption/URL_predicted_caption_{}.png'.format(i))
        plt.show()


# In[60]:




def generate_caption_from_file(filepath, model, tokenizer, idx2word):
    # Load image from the file
    im = Image.open(filepath)
    im = im.convert('RGB')

    # Generate caption
    pred_caption = gen_caption(filepath, model, tokenizer, idx2word)

    return pred_caption


# In[61]:




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    # if file:
    #     filename = secure_filename(file.filename)
    #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     # Process the image here and generate response
    #     time.sleep(3)
    #     response_text = "A flow chart showing the database schema on a black canvas"

    #     return jsonify({"message": response_text})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate the caption for the uploaded image
        pred_caption = generate_caption_from_file(filepath, cap_model, tokenizer, idx2word)

        return jsonify({"message": pred_caption})

if __name__ == '__main__':
    # captions_df = load_and_prepare_data(JSON_FILE_PATH)
    # sampled_captions = sample_data(captions_df, SAMPLES_SIZE)
    # print(sampled_captions.head())
    # print(len(captions_df), len(sampled_captions))
    # # Preprocess captions
    # preprocessed_captions = preprocess_captions(sampled_captions)
    # print(preprocessed_captions.head())
    # random_sample = sample_random_row(preprocessed_captions)
    # tokenizer = create_and_adapt_tokenizer(preprocessed_captions, VOCAB_LENGTH, TOKEN_LENGTH)


    # # Create lookup tables
    # word2idx, idx2word = create_lookup_tables(tokenizer)
    # train_imgs, train_captions, val_imgs, val_captions = split_data_into_train_val(preprocessed_captions)
    # # Create TensorFlow Datasets for training and validation
    # train_dataset = create_tf_datasets(train_imgs, train_captions, BUFFER_SIZE, BATCH_SIZE, tokenizer)
    # val_dataset = create_tf_datasets(val_imgs, val_captions, BUFFER_SIZE, BATCH_SIZE, tokenizer)
    # cap_model = initialize_and_compile_model(EMBEDDING_DIM, UNITS, image_augmentation)
    # history = fit_and_plot_model(cap_model, train_dataset, val_dataset, EPOCHS)
    # url = ["https://images.pexels.com/photos/5555112/pexels-photo-5555112.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"]

    # plot_prediction(PREDICTION_IMAGE_URL, cap_model, tokenizer, idx2word, sampled_captions)
    # plot_URL_prediction(url, cap_model, tokenizer, idx2word, sampled_captions)
    # print(len(captions_df))

    app.run(debug=True, use_reloader=False)


# In[ ]:




