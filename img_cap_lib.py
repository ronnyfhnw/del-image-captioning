# imports
import torch
import torchvision
import torchtext
from torchtext.vocab import vocab, GloVe, Vectors
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from PIL import Image
import string
from collections import OrderedDict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import pickle
import subprocess
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def data_download(target_folder:str="flickr8k"):
    '''
    Function for downloading the Flickr8k dataset. 

    Params:
    -------
        target_folder: str = flickr8k - the folder to download the dataset to.
    '''
    # check if flickr8k is in the current directory
    if os.path.exists(target_folder) and os.path.exists(f'{target_folder}/captions.txt') and os.path.exists(f'{target_folder}/images'):
        print('Data already exi sts at {}'.format(target_folder))
    else:
        print(f'{target_folder} is not in the current directory')
        print('starting download...')
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        
        # download
        subprocess.run("curl -L0 -o captions.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip".split(" "))
        subprocess.run("curl -L0 -o images.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip".split(" "))

        # captions transformation
        subprocess.run("mkdir captions".split(" "))
        subprocess.run("unzip captions.zip -d captions".split(" "))

        captions = pd.read_csv('captions/Flickr8k.token.txt', delimiter='\t', header=None)
        captions.columns = ['image', 'caption']
        captions.image = captions.image.apply(lambda x: x.split('#')[0])
        captions.to_csv(f'{target_folder}/captions.txt', index=False)

        # images transformation
        subprocess.run("mkdir images".split(" "))
        subprocess.run("unzip images.zip -d images".split(" "))
        subprocess.run(f"mv images/Flickr8k_Dataset {target_folder}/images".split(" "))

        # delete files
        subprocess.run(f"mv images/Flicker8k_Dataset/ {target_folder}/images".split(" "))
        subprocess.run("rm -r images".split(" "))
        subprocess.run("rm -r captions.zip".split(" "))
        subprocess.run("rm -r images.zip".split(" "))
        subprocess.run("rm -r captions".split(" "))
        
        print('download complete')


class CaptionPreprocessor:
    '''
    Class for preprocessing captions. 

    - Tokenizes captions
    - adds tokens and padding
    - creates word_set (word)
    - creates vocabulary (word, word_frequency)
    - filters out words by min_frequency
    - creates embedding matrix using GloVe

    '''
    def __init__(self, captions:pd.DataFrame, embedding=None, vocabulary=None, embedding_dim:int=300, max_length:int=22, min_frequency:int=1, sos_token='<SOS>', eos_token='<EOS>', pad_token='<PAD>', unk_token='<UNK>'):
        assert captions.columns[1] == 'caption', 'Captions are assumed to be in the first column of the dataframe'
        self.captions = captions
        self.embedding_dim = embedding_dim
        self.min_frequency = min_frequency
        self.max_length = max_length
        self.embedding = embedding
        self.vocabulary = vocabulary
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}
        self.n_words = 0

    def filter_captions(self):
        '''
        Filters captions by caption length. Removes punctuation and special characters.
        '''
        print("Shape captions:", self.captions.shape)
        self.captions['caption_word_length'] = self.captions.caption.apply(lambda x: len(x.rstrip(" .").split(" ")))
        old_shape = self.captions.shape
        # filter captions with less or equal to 20 words
        self.captions = self.captions[self.captions.caption_word_length <= self.max_length - 2]

        # remove all punctuation
        self.captions.caption = self.captions.caption.apply(lambda x: x.strip("."))
        self.captions = self.captions.dropna()
        self.captions.caption = self.captions.caption.apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))
        self.captions = self.captions.dropna()

        # strip image up until ".jpg" and add ".jpg" to the end
        self.captions.image = self.captions.image.apply(lambda x: x.split(".jpg")[0] + ".jpg")
        
        print("Shape captions after filtering:", self.captions.shape)
        print("Removed Captions: ", old_shape[0] - self.captions.shape[0], ", in Percent: ", round(((old_shape[0] - self.captions.shape[0]) / old_shape[0]) * 100, 2))

        # check if for all iamge_paths a image exists
        image_paths = np.array(os.listdir("flickr8k/images"))
        self.captions = self.captions[self.captions.image.isin(image_paths)]

    def tokenize_captions(self, caption_length:int):
        '''
        Function for tokenizing captions from a pandas dataframe. Captions are tokenized and padded up to caption_length.

        Params:
        -------
        captions: pd.DataFrame - pandas dataframe containing captions, captions are assumed to be in the first column
        caption_length: int - length of captions before padding

        Returns:
        --------
        captions: pd.DataFrame - pandas dataframe where captions are tokenized and padded
        vocabulary: set - set of unique words in the captions

        '''
        # create tokenizer
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    
        # create word_set 
        word_set = set()
        for i, caption in enumerate(self.captions.values):
            # tokenize caption
            tokenized = tokenizer(caption[1])

            # cut off tokenized caption at caption_length
            tokenized = tokenized[:caption_length]
            
            # update word_set
            word_set.update(tokenized)

            # add tokens
            tokenized = [self.sos_token] + tokenized + [self.eos_token]
            
            # padding
            if len(tokenized) < caption_length + 2:
                tokenized += [self.pad_token] * (caption_length + 2 - len(tokenized))

            # update caption
            self.captions.iloc[i, 1] = str(tokenized)

        # store variables
        self.word_set = word_set

    def create_vocabulary(self):
        '''
        Function for creating a vocabulary from the word_set. The vocabulary is a dictionary with words as keys and word frequencies as values.
        '''
        # count word occurences
        word_count = Counter()
        for caption in self.captions.caption.values:
            # turn string into list
            caption = eval(caption)
            # count words
            word_count.update(caption)

        # sort words inside of counter 
        sort_by_occurence = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        
        # transform into ordered dict for vocab class
        sorted_word_occurences = OrderedDict(sort_by_occurence)

        # create vocabulary, add words and set specials
        vocabulary = vocab(ordered_dict=sorted_word_occurences, min_freq=self.min_frequency, special_first=True, specials=[self.unk_token]) # special_first places the tokens or words specified in specials at the begining of the vocabulary

        # set self.unk_token as standart for unknown words
        vocabulary.set_default_index(vocabulary[self.unk_token])

        # test unknown words
        assert vocabulary['this word does not exist'] == vocabulary[self.unk_token]

        # store variables
        self.vocabulary = vocabulary

    def vectorize_captions(self):
        '''
        Function for vectorizing the captions using the vocabulary. Vectorized captions are required in the loss function of the model.
        Creates a new column 'vectorized_caption' in the captions dataframe.
        '''
        vectorized_captions = []
        for caption in self.captions.caption.values:
            vector = [self.vocabulary[word] for word in eval(caption)]
            assert len(vector) == 22
            vectorized_captions.append(vector)

        self.captions['vectorized_caption'] = vectorized_captions

    def create_embedding(self):
        '''
        Creates embedding matrix using GloVE. Filters out words, which are not in the vocabulary.
        '''
        # create embedding matrix
        glove = GloVe(name='6B', dim=self.embedding_dim)

        # reduce size
        glove = glove.get_vecs_by_tokens(self.vocabulary.vocab.itos_) # tensor of shape (vocab_size, embedding_dim)

        # store variables
        self.embedding = glove

    def preprocess(self):
        '''
        Combines all preprocessing functions.
        '''
        self.filter_captions()
        self.tokenize_captions(caption_length=self.max_length - 2)

        if self.embedding is None:
            self.create_vocabulary()
            self.create_embedding()
        
        self.vectorize_captions()

class Embedding:
    '''
    This class contains the embedding matrix for the captions. The forward function turns a word into a embedding vector.
    
    Params:
    -------
        embedding_matrix torch.Tensor: The embedding matrix that is used for the forward function.
        vocabulary torchtext.vocab.Vocab: The vocabulary that is used for the forward function.
    '''
    def __init__(self, embedding_matrix, vocabulary):
        # store variables
        self.embedding_matrix = embedding_matrix
        self.vocabulary = vocabulary
        self.words = self.vocabulary.get_itos()
    
    def caption_to_embedding(self, caption):
        '''
        Params:
        -------
            caption list: A list of words that is turned into a list of embedding vectors.

        Returns:
        --------
            embedding_vectors torch.Tensor: A tensor of embedding vectors. Shape: (len(caption), embedding_size)
        '''
        # turn caption into list of embedding vectors
        embedding = [self.forward(torch.tensor(self.vocabulary[word])) for word in caption]
        # stack embedding vectors
        embedding = torch.stack(embedding)
        # return embedding
        return embedding

    def forward(self, x):
        '''
        Returns embedding vector for a index. The index is the index of the word in the vocabulary. 

        Params:
        -------
            x torch.Tensor: The input tensor that is used for embedding. Shape: (batch_size, max_caption_length)
        
        Returns:
        --------
            embedding_vector torch.Tensor: A tensor of a embedding vector.
        '''
        return self.embedding_matrix[x]

    def index_to_caption(self, indexes:torch.Tensor):
        '''
        Turns a tensor of indexes into a list of words.
        Params:
        -------
            indexes torch.Tensor: A tensor of embedding vectors. Shape: (seq_len, batch_size)

        Returns:
        --------
            caption list: A list of words.
        '''

        # turn embedding vectors into list of words
        caption = [[self.words[idx] for idx in indexes[:, i]] for i in range(indexes.shape[1])]
        
        # return caption
        return caption

    def embedding_vectors_to_caption(self, embedding_vectors:torch.Tensor):
        ''' 
        Turn embedding vectors back in to a list of words.
        '''
        caption = [self.words[index] for index in [torch.eq(self.embedding_matrix, embedding_vector).sum(dim=1).argmax().item() for embedding_vector in embedding_vectors]]
        return caption

    def embedding_to_one_hot(self, embedding_vectors:torch.Tensor) -> torch.Tensor:
        '''
        Turns embedding vectors into one hot vectors for calculating loss with CrossEntropyLoss.

        Params:
        -------
            embedding_vectors torch.Tensor: A tensor of embedding vectors. Shape: (batch_size, seq_len, embedding_size)

        Returns:
        --------
            one_hot_vectors torch.Tensor: A tensor of one hot vectors. Shape: (batch_size, seq_len, vocab_size)
        '''
        # turn embedding vectors into one hot vectors
        one_hot_vectors = torch.stack([torch.eq(self.embedding_matrix, embedding_vector).sum(dim=1).argmax() for embedding_vector in embedding_vectors])
        # return one hot vectors
        return one_hot_vectors


class EncoderCNN(torch.nn.Module):
    '''
    This class contains the CNN that is used for feature extraction from the images. 
    
    Params:
    -------
        pretrained_net torchvision.models: Determines the pretrained network that is used for feature extraction.
        output_size int: The Outputsize of the last layer of the CNN. This is determined by the dimensions of the embedding matrix. 
    '''
    def __init__(self, net=None, pretrained_weights=None, output_size:int=300):
        super(EncoderCNN, self).__init__()
        self.output_size = output_size
        self.starting_net = net
        # load net
        self.cnn = net(weights=pretrained_weights)
        # replace last layer
        self.cnn.fc = torch.nn.Linear(in_features=self.cnn.fc.in_features, out_features=self.output_size, bias=True)
        # create linear stack for net
        self.net = torch.nn.Sequential(
            self.cnn
        )
        # send to device
        self.net = self.net.to(device)

    def forward(self, x):
        '''
        Params:
        -------
            x torch.Tensor: The input tensor that is used for feature extraction. Shape: (batch_size, channels, height, width)
        '''
        return self.net(x).unsqueeze(1)


class FlickrDataset(Dataset):
    '''
    This class creates a dataset for the flickr8k dataset.

    Params:
    -------
        captions pd.DataFrame: A dataframe that contains the captions for the images.
        embedding Embedding: The embedding class for the belonging to the captions. 

    Returns:
    --------
        image torch.Tensor: A tensor of the image. Shape: (batch_size, channels, height, width)
        caption torch.Tensor: A tensor of the caption. Shape: (batch_size, max_caption_length)
        lengths torch.Tensor: A tensor of the lengths of the captions. Shape: (batch_size)
        vectorized_caption torch.Tensor: A tensor of the vectorized caption. Shape: (batch_size, max_caption_length, embedding_size)
    '''
    def __init__(self, captions, embedding):
        self.captions = captions
        self.embedding = embedding


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        vectorized_caption = torch.from_numpy(np.array(self.captions.vectorized_caption.values[idx]))
        # get image
        image = torchvision.io.read_image("flickr8k/transformed_images/" + self.captions.iloc[idx, 0]).float()
        # get caption
        caption = self.captions.caption.values[idx]
        # turn string into list
        caption = self.embedding.caption_to_embedding(eval(caption))
        # get length
        length = self.captions.iloc[idx, 2] + 2

        return image, caption, length, vectorized_caption

class DecoderRNN(torch.nn.Module):
    '''
    Class for decoding a feature vector into a sequence of words. 
    
    Params:
    -------
        input_size int: The input size of the first layer of the RNN. This is determined by the dimensions of the embedding matrix. 
        hidden_size int: The hidden size of the RNN. Is also determined by the dimensions of the embedding matrix. 
        num_layers int: The number of layers of the RNN. 
        dropout float: The dropout rate of the RNN. 
    '''
    def __init__(self, input_size:int, hidden_size:int, len_vocab:int, num_layers:int=1, dropout:float=0.0, len_subtract:int=1):
        super(DecoderRNN, self).__init__()
        self.len_subtract = len_subtract
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.len_vocab = len_vocab
        # create RNN
        self.rnn = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, self.len_vocab)

    def forward(self, x, lengths):
        '''
        Params:
        -------
            x torch.Tensor: The input tensor that is used for feature extraction. Shape: (seq_len, batch_size, input_size)
            hidden tuple(torch.Tensor, torch.Tensor): The hidden state of the RNN. Shape: (seq_len, batch_size, input_size)

        Returns:
        --------
            output torch.Tensor: The output of the RNN. Shape: (seq_len, batch_size)
        '''
        # subtract 1 from lengths to remove the <EOS> token --> this teaches the model to stop predicting
        lengths = lengths - self.len_subtract
        
        # pack data
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        output, _ = self.rnn(packed)
        # pad data
        padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        output = self.fc(padded)
        
        return output

class ImageCaptioning(torch.nn.Module):
    '''
    Class for the image captioning model.
    
    Params:
    -------
        encoder EncoderCNN: The encoder that is used for feature extraction.
        decoder DecoderRNN: The decoder that is used for decoding the feature vector into a sequence of words. 
        embedding Embedding: The embedding matrix that is used for embedding the words. 
    '''
    def __init__(self, encoder, decoder, embedding, max_caption_length:int=22, batch_size:int=32):
        super(ImageCaptioning, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = embedding
        self.max_caption_length = max_caption_length

        # send submodels to device
        self.encoder.net.to(device)
        self.decoder.rnn.to(device)
        self.decoder.fc.to(device)
        self.embedding.embedding_matrix.to(device)

        # create padding vector for decoder
        self.padding_vector = torch.zeros((max_caption_length, batch_size, self.embedding.embedding_matrix.shape[0])).to(device)
        self.padding_vector[:, :, 1] = 1

        self.words = np.array(self.embedding.words)

    def forward_train(self, images, captions, lengths):
        '''
        This function is used during training time. 
        Forward pass through model while using the embedding vectors of the caption as input for the decoder.

        Params:
        -------
            images torch.Tensor: The input tensor that is used for feature extraction. Shape: (batch_size, channels, height, width)

        Returns:
        --------
            output torch.Tensor: The padded output of the RNN. Shape: (seq_len, batch_size)
        '''
        # extract features
        features = self.encoder.forward(images)
        
        # turn caption into embedding vectors
        input = torch.cat((features, captions), 1)
        # change shape to (seq_len, batch_size, input_size) for DecoderRNN
        input = input.permute(1, 0, 2)

        # forward pass through decoder
        output = self.decoder.forward(input, lengths)
        

        # pad output with 1's until max_caption_length (seq_len, batch_size) --> outputs have to be the same length for loss calculation, network only needs to predict '<EOS>'
        output = torch.cat((output, self.padding_vector[:self.max_caption_length - output.shape[0], :, :]), 0)

        
        return output

    def forward(self, images, max_length=22):
        '''
        This function is used for inference time. It takes a batch of images and returns a np.array containing the predicted captions. 

        Params:
        -------
            images torch.Tensor: A batch of images. Shape: (batch_size, 3, 224, 224)

        Returns:
        --------
            captions list: A list with the length of batch_size containing the predicted captions.
        '''
        # optimize inference speed
        self.eval()
        input = self.encoder.forward(images)
        hidden = None
        captions = None

        for i in range(max_length):
            output, hidden = self.decoder.rnn(input, hidden)
            indexes = self.decoder.fc(output)
            indexes = indexes.argmax(dim=2)
            input = self.embedding.embedding_matrix[indexes]
            words = self.words[indexes]
            if captions is None:
                captions = words
            else:
                captions = np.hstack((captions, words))

        return captions

        ############################ old version of forward function
        # # extract features from images
        # input = self.encoder.forward(images)

        # # create tensor for storing indexes
        # indexes = torch.tensor([], dtype=torch.long).to(device)

        # # TODO: implement for a whole batch of images

        # # loop over max_length 
        # with torch.no_grad():
        #     hidden = None
        #     for i in range(max_length):
        #         output, hidden = self.decoder.rnn(input, hidden)
        #         index = self.decoder.fc(output)
        #         indexes = torch.cat((indexes, index.argmax(2).T), 0)
        #         # get embedding vector of last predicted word
        #         input = self.embedding.embedding_matrix[indexes[-1, :]].unsqueeze(0)
        #         input = input.to(device)
        #         # break when <EOS> is predicted and fill up the rest of the caption with <PAD>
        #         if index.argmax(2).item() == 4:
        #             break

        #     # pad output with 1's until max_caption_length (seq_len, batch_size)
        #     if indexes.shape[0] < max_length:
        #         indexes = torch.cat((indexes, torch.ones((max_length - indexes.shape[0], 1), dtype=torch.long).to(device)), 0)
        #     elif indexes.shape[0] > max_length:
        #         indexes = indexes[:max_length, :]

        # captions = self.embedding.index_to_caption(indexes)

        # return captions

    # create training function for ImageCaptioning model
    def train_model(self, loader, optimizer, criterion, epochs:int=200, print_every:int=100):
        '''
        Function for training ImageCaptioning models with different loaders, optimizers and criterions.

        Params:
        -------
            model ImageCaptioning:      A fully initialized ImageCaptioning object.
            loader FlickrLoader:        An instance of the FlickrLoader class.
            optimizer torch.optim:      A pytorch optimizer.
            criterion torch.nn:         A pytorch loss function.
            epochs int:                 The number of epochs that the model should be trained for.
            print_every int:            The number of batches that should be trained before printing the loss.

        Returns:
        --------
            losses list:                A list containing the loss for each batch.
            model_state_dict dict:      The state_dict of the trained model.
            model_stats dict:           A dictionary containing the training stats and general information about the model.
        '''
        # setup variables
        losses = []
        model_stats = {}
        model_stats['epochs'] = epochs
        model_stats['batch_size'] = loader.batch_size
        model_stats['optimizer'] = optimizer
        model_stats['criterion'] = criterion
        model_stats['start_time'] = time.time()
        model_stats['end_time'] = None
        model_stats['total_time'] = None
        model_stats['encoder_starting_net'] = self.encoder.starting_net
        model_stats['max_caption_length'] = self.max_caption_length

        # decoder
        model_stats['decoder_input_size'] = self.decoder.input_size
        model_stats['decoder_hidden_size'] = self.decoder.hidden_size
        model_stats['decoder_len_vocab'] = self.decoder.len_vocab
        model_stats['decoder_num_layers'] = self.decoder.num_layers
        model_stats['decoder_dropout'] = self.decoder.dropout

        # send model to device
        self.to(device)
        self.encoder.to(device)
        self.decoder.to(device)

        # set model to training mode
        self.train()

        average_epoch_losses = []

        # loop over epochs
        for epoch in range(epochs):
            epoch_loss = 0
            # loop over batches
            for i, (images, captions, lengths, vectorized_captions) in enumerate(loader):
                # send data to device
                images = images.to(device)
                captions = captions.to(device)
                lengths = lengths.to('cpu')
                vectorized_captions = vectorized_captions.to(device)

                # zero out gradients
                optimizer.zero_grad()

                # forward pass through model
                output = self.forward_train(images=images, captions=captions, lengths=lengths)
                # shape for crossentropy (batch_size, num_classes, seq_len)
                output = output.permute(1, 2, 0)
                
                # calculate loss
                try:
                    loss = criterion(output, vectorized_captions)
                except:
                    print(output.shape, vectorized_captions.shape)
                    print(self.embedding.index_to_caption(vectorized_captions.permute(1, 0))[0])
                    raise
                epoch_loss += loss.item()

                # backpropagate loss
                loss.backward()

                # update weights
                optimizer.step()

                # print loss
                if i % print_every == 0:
                    print(f'Epoch: {epoch+1}/{epochs} | Batch: {i+1}/{len(loader)} | Loss: {loss.item()}')
                losses.append(loss.item())
            print(f'Epoch: {epoch+1}/{epochs} | Average Epoch Loss: {epoch_loss/len(loader)}')
            average_epoch_losses.append(epoch_loss/len(loader))

        model_stats['end_time'] = time.time()
        model_stats['total_time'] = model_stats['end_time'] - model_stats['start_time']
        model_stats['encoder_state_dict'] = self.encoder.state_dict()
        model_stats['decoder_state_dict'] = self.decoder.state_dict()
        model_stats['model_state_dict'] = self.state_dict()
        model_stats['embedding'] = self.embedding
        model_stats['losses'] = losses
        model_stats['average_epoch_losses'] = average_epoch_losses

        return model_stats
    
    
def load_captioning_model(model_stats:dict):
    '''
    Function for loading a trained ImageCaptioning model from a dictionary containing the model stats. model_stats are created by the train function. 
    '''
    # load encoder
    encoder = EncoderCNN(net=model_stats['encoder_starting_net'])
    encoder.load_state_dict(model_stats['encoder_state_dict'])
    # load decoder
    decoder = DecoderRNN(input_size=model_stats['decoder_input_size'], hidden_size=model_stats['decoder_hidden_size'], num_layers=model_stats['decoder_num_layers'], dropout=model_stats['decoder_dropout'], len_vocab=model_stats['decoder_len_vocab'])
    decoder.load_state_dict(model_stats['decoder_state_dict'])
    # create model
    model = ImageCaptioning(encoder=encoder, decoder=decoder, embedding=model_stats['embedding'], max_caption_length=model_stats['max_caption_length'], batch_size=model_stats['batch_size'])
    model.load_state_dict(model_stats['model_state_dict'])

    # send model to device
    model.to(device)
    model.encoder.to(device)
    model.decoder.to(device)

    return model

class FlickrLoader(torch.utils.data.DataLoader):
    '''
    Simple DataLoader for the Flickr8k dataset. 
    drop_last is set to True to avoid problems with batches that are smaller than the batch_size. --> The batch_size is used when padding during training time. If a smaller batch is used, an error occurs. 
    '''
    def __init__(self, dataset, batch_size=64, shuffle=False, drop_last=True):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    def __iter__(self):
        return super().__iter__()

    def __next__(self):
        return super().__next__()


class ImagePreprocessor:
    def __init__(self, image_size:tuple=(224, 224), normalize:bool=True, image_folder_path:str=None):
        '''
        Class for preprocessing images. The image_folder_path has to contain the folder 'images', where the original images are stored. Creates a new folder 'transformed_images' within the image_folder_path.
        '''
        self.image_size = image_size
        self.normalize = normalize
        self.image_folder_path = image_folder_path

    def preprocess_images(self):
        # setup variables
        image_paths = os.listdir(self.image_folder_path + '/images')
        image_dimensions = pd.DataFrame({'image_path':[], 'dimension':[]}, index=None)

        # get image dimensions
        for image_path in image_paths:
            # load image
            image = Image.open(self.image_folder_path + "/images/" + image_path)
            # get image dimensions
            image_dimensions = pd.concat([image_dimensions, pd.DataFrame({'image_path':[image_path], 'dimension':[image.size]})], axis=0)

        # generate height and width columns
        image_dimensions['height'] = image_dimensions.dimension.apply(lambda x: x[1])
        image_dimensions['width'] = image_dimensions.dimension.apply(lambda x: x[0])
        image_dimensions['area'] = image_dimensions['height'] * image_dimensions['width']

        # get max_img
        max_img = image_dimensions[image_dimensions.area == image_dimensions.area.max()]

        # build transform
        if self.normalize:
            transform = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = T.Compose([
                T.CenterCrop((max_img.height.values[0], max_img.width.values[0])),
                T.Resize((224, 224)),
                T.ToTensor()
            ])

        # check if transformed_images folder exists
        if not os.path.exists(self.image_folder_path + '/transformed_images/'):
            os.mkdir(self.image_folder_path + '/transformed_images/')
            print("starting image preprocessing ...")
            for image_path in image_paths:
                # load image
                image = Image.open(self.image_folder_path + "/images/" + image_path)
                # transform image
                image = transform(image)
                # check shape
                assert image.shape == (3, 224, 224)
                # save image to disk
                torchvision.utils.save_image(image, self.image_folder_path + "/transformed_images/" + image_path)
            print("image preprocessing finished")
        else:
            print("transformed_images folder already exists. No preprocessing necessary.")

